#!/usr/bin/env python3
"""Convert the media side of a Tilde Omni checkpoint into an mmproj GGUF (vision + audio, for tilde-server and mtmd).

- vision (projector type tilde_omni_v): TLive-Omni-4B's Qwen3.5 ViT (visual.*, frozen) up to its merger LayerNorm,
  i.e. the pre-merger cells the trainer's online encoder produces (tilda.qwen35.model.online.TLiveVision), then per
  image token (blank cells merged, tilda.vision.merge): the frozen merger MLP as the projector, and Tilde's trained
  bridge (tilda.vision.model.VisionBridge: linear + residual MLP + position encoding of each token's geometry). The
  point head's sub-patch keys (LayerNorm + linear over each token's 2x2 sub-patch features) are computed here too.
- audio (projector type tilde_omni_a): TLive's AuT (audio_tower.*, Qwen3-Omni AuT shape, output 2560 = the LM width)
  and Tilde's trained audio bridge (tilda.audio.model.AudioModule.linear; native: no rescaling).

Both encoders output 3072 floats per token: the 2560-wide LM input embedding, then 512 point-head values (the 4 x 128
sub-patch keys for vision, zeros for audio); the LM GGUF reads the extra 512 as side data (n_embd_inp 3072).

    python tilde/convert_tilde_omni_mmproj.py --base TLive-Omni-4B/ --checkpoint final_folded/ \\
        --outtype f16 --outfile mmproj-tilde-4b-omni-F16.gguf
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gguf-py"))

import torch  # noqa: E402

import gguf  # noqa: E402
from conversion.base import LazyTorchTensor, ModelBase  # noqa: E402
from conversion.qwen3vl import Qwen3OmniMmprojModel, Qwen3VLVisionModel  # noqa: E402

logger = logging.getLogger("tilde-omni-mmproj")

POINT_WIDTH = 512  # 4 sub-patches x 128 (tilda.vision.point.SUB_KEY_WIDTH)
TOKEN_PX = 32
MERGE_TOLERANCE = 3


@ModelBase.register("TildeOmniMmprojModel")
class TildeOmniMmprojModel(Qwen3OmniMmprojModel):
    has_audio_encoder = True
    has_vision_encoder = True
    ckpt_dir: Path

    def get_vision_config(self):
        return self.global_config.get("vision_config")

    def get_audio_config(self):
        return self.global_config.get("audio_config")

    def set_gguf_parameters(self):
        # vision and (through the MRO) audio parameters, without the Qwen projector types
        Qwen3VLVisionModel.set_gguf_parameters(self)
        w = self.gguf_writer
        w.add_clip_vision_projector_type("tilde_omni_v")
        w.add_clip_audio_projector_type("tilde_omni_a")
        w.add_uint32("tilde_omni.point_width", POINT_WIDTH)
        w.add_uint32("tilde_omni.token_px", TOKEN_PX)
        w.add_uint32("tilde_omni.merge_tolerance", MERGE_TOLERANCE)

    def index_tensors(self, remote_hf_model_id=None):
        tensors = super().index_tensors(remote_hf_model_id)
        from safetensors import safe_open
        for file, prefix in (("vision.safetensors", "tilde_v."), ("audio.safetensors", "tilde_a.")):
            path = type(self).ckpt_dir / file
            with safe_open(str(path), framework="pt") as st:
                for key in st.keys():
                    tensors[prefix + key] = (lambda path=path, key=key: _load(path, key))
        return tensors

    @classmethod
    def filter_tensors(cls, item):
        if item[0].startswith(("tilde_v.", "tilde_a.")):
            return item
        return super().filter_tensors(item)

    # trained bridge / point tensors -> mmproj names (the rest of the point head goes into the LM GGUF)
    BRIDGE = {
        "tilde_v.bridge.linear.weight": "tilde.v.lin.weight", "tilde_v.bridge.linear.bias": "tilde.v.lin.bias",
        "tilde_v.bridge.mlp.0.weight": "tilde.v.mlp0.weight", "tilde_v.bridge.mlp.0.bias": "tilde.v.mlp0.bias",
        "tilde_v.bridge.mlp.2.weight": "tilde.v.mlp2.weight", "tilde_v.bridge.mlp.2.bias": "tilde.v.mlp2.bias",
        "tilde_v.bridge.position.weight": "tilde.v.pos.weight", "tilde_v.bridge.position.bias": "tilde.v.pos.bias",
        "tilde_v.point_head.sub_key.0.weight": "tilde.v.sub_key_norm.weight",
        "tilde_v.point_head.sub_key.0.bias": "tilde.v.sub_key_norm.bias",
        "tilde_v.point_head.sub_key.1.weight": "tilde.v.sub_key.weight",
        "tilde_a.linear.weight": "tilde.a.lin.weight", "tilde_a.linear.bias": "tilde.a.lin.bias",
    }

    def modify_tensors(self, data_torch, name, bid):
        if name.startswith(("tilde_v.", "tilde_a.")):
            if name in self.BRIDGE:
                yield self.BRIDGE[name], data_torch
            # markers (LM vocabulary rows), the rest of the point head (LM GGUF), frame_scale / speech_scale (unused
            # with a native store) are not part of the mmproj
            return
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if new_name.startswith("tilde."):
            return gguf.GGMLQuantizationType.F32  # trained bridge and point keys, small: full precision
        return super().tensor_force_quant(name, new_name, bid, n_dims)


def _load(path: Path, key: str):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as st:
        return st.get_tensor(key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="TLive-Omni-4B directory (HF layout)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Tilde Omni checkpoint (vision / audio.safetensors)")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--outtype", choices=("f16", "bf16", "f32"), default="f16")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    report = json.loads((args.checkpoint / "report.json").read_text())
    spec, audio_spec = report.get("vision_spec") or {}, report.get("audio_spec") or {}
    meta = spec.get("features_meta") or {}
    if (meta.get("encoder") != "tlive" or not meta.get("native_bridge") or spec.get("projector_trained")
            or spec.get("point_head") != "unified" or not spec.get("merged") or spec.get("order_features")
            or int(meta.get("token_px", TOKEN_PX)) != TOKEN_PX or int(meta.get("merge_tolerance", MERGE_TOLERANCE)) != MERGE_TOLERANCE):
        raise SystemExit(f"unsupported vision spec {spec}")
    if not audio_spec.get("native") or audio_spec.get("acif") or audio_spec.get("l12"):
        raise SystemExit(f"unsupported audio spec {audio_spec}")
    # the graph feeds the merger output to the bridge as is: no projector norm (tilda.vision.model.Bridge:
    # projector_l2norm / --vision-projector-norm mean keep a projected_scale buffer)
    from safetensors import safe_open
    with safe_open(str(args.checkpoint / "vision.safetensors"), framework="pt") as st:
        vision_keys = set(st.keys())
    if meta.get("projector_l2norm") or any(k.endswith("projected_scale") for k in vision_keys):
        raise SystemExit("checkpoints with a normalized projector output (projector_l2norm / projected_scale) are not supported")
    # the ViT position-embedding interpolation of the graph is bilinear with align_corners (transformers' Qwen3.5 ViT)
    vcfg = json.loads((args.base / "config.json").read_text()).get("vision_config") or {}
    if vcfg.get("interpolation_mode", "bilinear") != "bilinear" or not vcfg.get("interpolation_align_corners", True):
        raise SystemExit("the ViT position embeddings must interpolate bilinearly with align_corners")

    with tempfile.TemporaryDirectory(prefix="tilde-omni-mmproj-") as tmp:
        work = Path(tmp)
        config = json.loads((args.base / "config.json").read_text())
        config["architectures"] = ["TildeOmniMmprojModel"]
        config.pop("auto_map", None)
        (work / "config.json").write_text(json.dumps(config))
        for name in ("preprocessor_config.json", "model.safetensors.index.json"):
            shutil.copy(args.base / name, work / name)
        for shard in sorted(args.base.glob("model-*.safetensors")):
            (work / shard.name).symlink_to(shard.resolve())
        TildeOmniMmprojModel.ckpt_dir = args.checkpoint
        ftype = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
                 "bf16": gguf.LlamaFileType.MOSTLY_BF16}[args.outtype]
        hparams = ModelBase.load_hparams(work, False)
        model = TildeOmniMmprojModel(work, ftype, args.outfile, hparams=hparams, model_name="tilde-4b-omni",
                                     small_first_shard=False)
        model.write()
        logger.info(f"wrote {model.fname_out}")


if __name__ == "__main__":
    main()
