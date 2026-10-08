#!/usr/bin/env python3
"""Convert a Tilde checkpoint into one GGUF file.

Tilde is the t40 checkpoint: Ling-3.0-tiny-base (inclusionAI/Ling-3.0-tiny-base) fine-tuned with a joint decision
head (Cloudflare Clef's JointSchemaHead, "fast" variant), bidirectional MLA layers and 263 trained marker tokens. The
checkpoint directory holds only what training changed:

    dense_delta.safetensors      fp32 deltas of the trained dense linears (base + delta = trained weight)
    routed_experts.safetensors   the trained routed experts, w_gu [E, 2I, H] (gate rows first) and w_down [E, H, I]
    markers.safetensors          the 263 marker embeddings (ids marker_start .. marker_start + 262), fp32
    joint_head.safetensors       the joint decision head, fp32
    report.json, tokenizer/      training report (attention mode, noul format, ...) and the tokenizer with markers

The converter streams: tensors are read lazily from the base shards and the checkpoint and written one at a time, so
peak RAM stays at a few GB. The fitted temperature T is folded into the head (the three logit scales of the head are
stored as their inference values, and the prior and joint scales are divided by T), so the raw head output is the
calibrated logit. Everything the runtime needs to encode requests exactly as training did is written as `tilde.*`
metadata.

    python tilde/convert_tilde_to_gguf.py --base BASE_DIR --checkpoint CKPT_DIR --temperature 1.26 \\
        --outtype bf16 --outfile tilde-8b-a1b-BF16.gguf

--outtype q8_0 writes Q8_0 directly (no bf16 intermediate); other presets are made from the bf16 file with
llama-quantize (see tilde/quantize_presets.sh).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gguf-py"))

import torch  # noqa: E402

import gguf  # noqa: E402
from conversion.base import LazyTorchTensor, ModelBase  # noqa: E402
from conversion.bailingmoe3 import BailingMoeV3Model  # noqa: E402

logger = logging.getLogger("tilde-convert")

TILDE_FORMAT_VERSION = 1
STRUCTURE_TOKENS = ("<|tilda_start|>", "<|tilda_state|>", "<|tilda_state_end|>",
                    "<|tilda_question|>", "<|tilda_options|>", "<|tilda_decision|>")
OPTION_TOKENS = tuple(f"<|tilda_option_{index:03d}|>" for index in range(255))
NOUL_TOKENS = ("<|tilda_no|>", "<|tilda_yes|>")
ALL_TOKENS = STRUCTURE_TOKENS + OPTION_TOKENS + NOUL_TOKENS
CLEF_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
JOINT_HEAD_CONFIG = dict(hidden_size=1536, width=1024, routing_layers=2, layers=4, heads=16, feedforward=4096)
HEAD_NORM_EPS = 1e-5  # torch.nn.LayerNorm default, used by every norm of the head
LOGIT_SCALE_CLAMP = math.log(100.0)

# Request limits written into the file (the server reads them; it rejects, never truncates).
MAX_REQUEST_TOKENS = 65536    # state + every question's schema block
MAX_QUESTION_TOKENS = 32768   # state + one question's schema block (the record as if it had only that question)
MAX_QUESTIONS = 32            # TypeSafe System One's documented limit; the server may raise it
MAX_OPTIONS = 255

ORG = "Resident Labs"
ORG_URL = "https://residentlabs.dev/"
REPO_URL = "https://github.com/resident-labs/tilde.cpp"
HF_MODEL = "resident-labs/tilde-8b-a1b"
HF_GGUF = "resident-labs/tilde-8b-a1b-GGUF"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SafetensorsIndex:
    """Name -> lazy loader over a list of safetensors files (kept open for the whole conversion)."""

    def __init__(self, files: Iterable[Path]):
        self.loaders: dict[str, Callable[[], torch.Tensor]] = {}
        self.dtypes: dict[str, str] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self._open = []
        for path in files:
            st = gguf.utility.SafetensorsLocal(path)
            tensors = st.__enter__()
            self._open.append(st)
            for name in tensors.keys():
                data = tensors[name]
                self.loaders[name] = (lambda data=data: LazyTorchTensor.from_local_tensor(data))
                self.dtypes[name] = data.dtype
                self.shapes[name] = tuple(data.shape)

    def __contains__(self, name: str) -> bool:
        return name in self.loaders

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.loaders[name]()

    def keys(self):
        return self.loaders.keys()


@ModelBase.register("TildeModel")
class TildeModel(BailingMoeV3Model):
    model_arch = gguf.MODEL_ARCH.TILDE
    no_mtp = True  # the MTP layer of the base is not used (no text is generated)

    # set by main() before construction
    base_dir: Path
    ckpt_dir: Path
    report: dict
    temperature: float
    marker_start: int
    head_type: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.F32
    lexical_type: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.BF16
    extra_metadata: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # head blocks are named dec.blk.N, routing blocks first, then the joint blocks
        n_head_blocks = JOINT_HEAD_CONFIG["routing_layers"] + JOINT_HEAD_CONFIG["layers"]
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, max(self.block_count, n_head_blocks))
        self._scales: dict[str, float] = {}

    # ---------------------------------------------------------------- tensors

    def index_tensors(self, remote_hf_model_id: str | None = None):
        del remote_hf_model_id
        cls = type(self)
        n_layers = self.hparams["num_hidden_layers"]
        cls._main_layers = n_layers
        base_index = json.loads((cls.base_dir / "model.safetensors.index.json").read_text())["weight_map"]
        base = SafetensorsIndex(sorted({cls.base_dir / part for part in base_index.values()}))
        delta = SafetensorsIndex([cls.ckpt_dir / "dense_delta.safetensors"])
        experts = SafetensorsIndex([cls.ckpt_dir / "routed_experts.safetensors"])
        markers = SafetensorsIndex([cls.ckpt_dir / "markers.safetensors"])
        head = SafetensorsIndex([cls.ckpt_dir / "joint_head.safetensors"])
        self._sources = (base, delta, experts, markers, head)

        start = cls.marker_start
        n_markers = markers.shapes["special_embeddings.weight"][0]
        if n_markers != len(ALL_TOKENS):
            raise ValueError(f"expected {len(ALL_TOKENS)} marker rows, found {n_markers}")
        inter = self.hparams["moe_intermediate_size"]
        tensors: dict[str, Callable[[], torch.Tensor]] = {}
        used_delta: set[str] = set()

        for name in base.keys():
            layer = None
            if name.startswith("model.layers."):
                layer = int(name.split(".")[2])
                if layer >= n_layers:
                    continue  # MTP layer
            if ".mlp.experts." in name:
                continue  # replaced by the trained experts below
            short = name.removeprefix("model.")
            if short in delta:
                if tuple(delta.shapes[short]) != tuple(base.shapes[name]):
                    raise ValueError(f"delta shape mismatch for {name}")
                used_delta.add(short)
                # fp32 master = fp32(base) + fp32 delta, exactly as tilda.model.apply_dense_delta builds it
                tensors[name] = (lambda name=name, short=short: base[name].float() + delta[short].float())
            elif name == "model.word_embeddings.weight":
                # base rows below the marker block, then the trained markers (fp32)
                tensors[name] = (lambda name=name: torch.cat(
                    [base[name][:start].float(), markers["special_embeddings.weight"].float()], 0))
            elif name == "lm_head.weight":
                # the frozen original output table: the head reads it for the lexical vector of each option token.
                # Rows past the base vocabulary are zero (option text never tokenizes to a marker).
                rows = start + n_markers
                vocab = base.shapes[name][0]
                if rows >= vocab:
                    tensors[name] = (lambda name=name, rows=rows, vocab=vocab: torch.nn.functional.pad(
                        base[name], (0, 0, 0, rows - vocab)))
                else:
                    tensors[name] = (lambda name=name, rows=rows: base[name][:rows])
            else:
                tensors[name] = (lambda name=name: base[name])
        unused = set(delta.keys()) - used_delta
        if unused:
            raise ValueError(f"dense delta keys not in the base: {sorted(unused)[:8]}")

        for key in experts.keys():
            # layers.N.mlp.w_gu [E, 2I, H] / layers.N.mlp.w_down [E, H, I]
            layer = int(key.split(".")[1])
            prefix = f"model.layers.{layer}.mlp.experts."
            if key.endswith(".w_gu"):
                tensors[prefix + "gate_proj.weight"] = (lambda key=key: experts[key][:, :inter, :])
                tensors[prefix + "up_proj.weight"] = (lambda key=key: experts[key][:, inter:, :])
            elif key.endswith(".w_down"):
                tensors[prefix + "down_proj.weight"] = (lambda key=key: experts[key])
            else:
                raise ValueError(f"unexpected routed expert tensor {key}")

        for key in head.keys():
            tensors["joint_head." + key] = (lambda key=key: head[key])

        out = {}
        for name, gen in tensors.items():
            if titem := self.filter_tensors((name, gen)):
                out[titem[0]] = titem[1]
        return out

    def modify_tensors(self, data_torch, name: str, bid: int | None):
        if name.startswith("joint_head."):
            yield from self._modify_head(data_torch, name)
            return
        if ".mlp.experts." in name:
            # already stacked [E, ...]: skip BailingMoeV3Model's per-expert buffering
            yield from super(BailingMoeV3Model, self).modify_tensors(data_torch, name, bid)
            return
        if name.endswith(".attention.kv_b_proj.weight"):
            # keep kv_b whole ([n_head * (qk_nope + v), kv_lora]); the runtime decompresses K and V for the
            # non-causal prefill instead of attending in the latent space
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_KV_B, bid), data_torch
            return
        yield from super().modify_tensors(data_torch, name, bid)

    def _modify_head(self, data_torch, name: str):
        parts = name.split(".")
        # learned scalars, stored as the values used at inference, with the temperature folded in
        if len(parts) == 2 and parts[1] in ("prior_logit_scale", "joint_logit_scale", "residual_gate"):
            value = float(LazyTorchTensor.to_eager(data_torch))
            if parts[1] == "residual_gate":
                self._scales[parts[1]] = 1.0 / (1.0 + math.exp(-value))
            else:
                self._scales[parts[1]] = math.exp(min(value, LOGIT_SCALE_CLAMP))
            if len(self._scales) == 3:
                T = type(self).temperature
                # logit = prior_scale * prior + gate * (joint_scale * cosine + residual); logit / T is exact with
                # prior_scale / T and gate / T
                scales = [self._scales["prior_logit_scale"] / T, self._scales["joint_logit_scale"],
                          self._scales["residual_gate"] / T]
                logger.info(f"head scales (T={T} folded): {scales}")
                yield self.format_tensor_name(gguf.MODEL_TENSOR.DECISION_SCALES, suffix=""), \
                    torch.tensor(scales, dtype=torch.float32)
            return
        if parts[1] == "layers":
            parts[2] = str(int(parts[2]) + JOINT_HEAD_CONFIG["routing_layers"])
        name = ".".join(parts)
        # nn.MultiheadAttention keeps q, k, v in one tensor
        for suffix in ("weight", "bias"):
            if name.endswith(".in_proj_" + suffix):
                prefix = name[:-len("in_proj_" + suffix)]
                for x, data in zip("qkv", data_torch.chunk(3, dim=0)):
                    yield self.map_tensor_name(prefix + x + "." + suffix), data
                return
        yield self.map_tensor_name(name), data_torch

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if name.startswith("joint_head.") or new_name.startswith(("decision.", "dec.", "token_types")):
            return type(self).head_type
        if new_name == "output.weight":
            return type(self).lexical_type
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    # ---------------------------------------------------------------- metadata

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        w = self.gguf_writer
        cls = type(self)
        report = cls.report
        # route the number of experts the checkpoint was trained with (the base config's value is only a default)
        k = report.get("experts_per_token") or self.hparams["num_experts_per_tok"]
        w.add_expert_used_count(int(k))
        w.add_decision_type(gguf.DecisionType.TILDE)
        w.add_decision_routing_block_count(JOINT_HEAD_CONFIG["routing_layers"])
        w.add_decision_block_count(JOINT_HEAD_CONFIG["layers"])
        w.add_decision_head_count(JOINT_HEAD_CONFIG["heads"])
        w.add_layer_norm_eps(HEAD_NORM_EPS)
        w.add_causal_attention(False)

        def s(key, value):
            w.add_string("tilde." + key, value)

        def u(key, value):
            w.add_uint32("tilde." + key, value)

        u("format_version", TILDE_FORMAT_VERSION)
        s("serialization", report["serialization"])
        s("attention_mode", report["attention_mode"])
        s("input_layout", report.get("input_layout", "state_first"))
        s("noul_format", report["noul_format"])
        s("head_variant", report["head_variant"]["head_variant"])
        s("head_implementation", report.get("head_implementation", "FastJointSchemaHead"))
        s("clef_revision", report["clef_revision"])
        u("experts_per_token", report.get("experts_per_token") or self.hparams["num_experts_per_tok"])
        u("marker_start", cls.marker_start)
        u("marker_count", len(ALL_TOKENS))
        w.add_array("tilde.structure_token_ids", [cls.marker_start + i for i in range(len(STRUCTURE_TOKENS))])
        u("max_request_tokens", MAX_REQUEST_TOKENS)
        u("max_question_tokens", MAX_QUESTION_TOKENS)
        u("max_questions", MAX_QUESTIONS)
        u("max_options", MAX_OPTIONS)
        w.add_float32("tilde.temperature_folded", cls.temperature)
        for key, value in cls.extra_metadata.items():
            if isinstance(value, str):
                s(key, value)
            elif isinstance(value, bool):
                w.add_bool("tilde." + key, value)
            elif isinstance(value, int):
                u(key, value)
            elif isinstance(value, float):
                w.add_float32("tilde." + key, value)

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only)
        w = self.gguf_writer
        w.add_organization(ORG)
        w.add_url(ORG_URL)
        w.add_source_url(REPO_URL)
        w.add_source_repo_url(f"https://huggingface.co/{HF_MODEL}")
        w.add_base_model_count(1)
        w.add_base_model_name(0, "Ling-3.0-tiny-base")
        w.add_base_model_organization(0, "inclusionAI")
        w.add_base_model_repo_url(0, "https://huggingface.co/inclusionAI/Ling-3.0-tiny-base")


def stage_dir(base: Path, ckpt: Path, report: dict, work: Path) -> Path:
    """A directory with what the converter reads besides the weights: the base config (vocabulary grown by the
    marker block, no MTP layer, architecture renamed) and the checkpoint's tokenizer."""
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads((base / "config.json").read_text())
    start = int(report["marker_start"])
    config["vocab_size"] = start + len(ALL_TOKENS)
    config["architectures"] = ["TildeModel"]
    config.pop("auto_map", None)
    (work / "config.json").write_text(json.dumps(config, indent=1))
    for path in (ckpt / "tokenizer").iterdir():
        shutil.copy(path, work / path.name)
    if not (work / "special_tokens_map.json").exists() and (base / "special_tokens_map.json").exists():
        shutil.copy(base / "special_tokens_map.json", work / "special_tokens_map.json")
    return work


def check_tokenizer(work: Path, start: int) -> dict:
    """The marker ids must follow the base vocabulary exactly (tilda.joint_serving.load_joint_checkpoint)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(work, trust_remote_code=False)
    ids = [tok.convert_tokens_to_ids(t) for t in ALL_TOKENS]
    if ids != list(range(start, start + len(ALL_TOKENS))):
        raise SystemExit("checkpoint marker ids do not match its tokenizer")
    return {"bos_token_id": -1 if tok.bos_token_id is None else int(tok.bos_token_id),
            "pad_token_id": -1 if tok.pad_token_id is None else int(tok.pad_token_id)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="Ling-3.0-tiny-base directory (HF layout)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Tilde checkpoint directory (stage1_final)")
    ap.add_argument("--temperature", type=float, required=True, help="fitted global temperature to fold (1.26 for t40)")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--outtype", choices=("bf16", "f16", "f32", "q8_0"), default="bf16")
    ap.add_argument("--head-type", choices=("f32", "f16"), default="f32", help="type of the joint head tensors")
    ap.add_argument("--lexical-type", choices=("bf16", "f16", "f32", "q8_0"), default="bf16",
                    help="type of the frozen output table the head reads option-token vectors from")
    ap.add_argument("--model-name", default="tilde-8b-a1b")
    ap.add_argument("--no-verify", action="store_true", help="skip the sha256 checks of the checkpoint files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not (math.isfinite(args.temperature) and args.temperature > 0):
        raise SystemExit("temperature must be finite and positive")
    report = json.loads((args.checkpoint / "report.json").read_text())
    if report.get("serialization") != "clef_joint_bare_v1" or report.get("clef_revision") != CLEF_REVISION:
        raise SystemExit("not a clef_joint_bare_v1 joint checkpoint")
    if report.get("head_config") != JOINT_HEAD_CONFIG:
        raise SystemExit(f"unexpected head config {report.get('head_config')}")
    if report.get("head_variant", {}).get("head_variant", "fast") != "fast":
        raise SystemExit("only the fast head variant is supported")
    if report.get("question_prefix") is not None or report.get("state_repeat"):
        raise SystemExit("question prefix / state repeat checkpoints are not supported")
    if report.get("input_layout", "state_first") != "state_first":
        raise SystemExit("only state_first checkpoints are supported")
    if report.get("attention_mode", "causal") != "bidirectional_mla":
        raise SystemExit("tilde.cpp serves bidirectional_mla checkpoints")
    if report.get("backbone_family") not in (None, "ling"):
        raise SystemExit(f"unsupported backbone family {report.get('backbone_family')}")
    base_rev = report.get("base_identity", {}).get("revision", "")

    verified = {}
    if not args.no_verify:
        for name, meta in report["files"].items():
            digest = sha256_file(args.checkpoint / name)
            if digest != meta["sha256"]:
                raise SystemExit(f"checkpoint hash mismatch: {name}")
            verified[name] = digest
        for meta in report.get("base_identity", {}).get("weights", []):
            if sha256_file(args.base / meta["file"]) != meta["sha256"]:
                raise SystemExit(f"base differs from the pinned clean base: {meta['file']}")
        logger.info("checkpoint and base hashes verified")

    with tempfile.TemporaryDirectory(prefix="tilde-convert-") as tmp:
        work = stage_dir(args.base, args.checkpoint, report, Path(tmp) / "model")
        tokinfo = check_tokenizer(work, int(report["marker_start"]))
        types = {"f32": gguf.GGMLQuantizationType.F32, "f16": gguf.GGMLQuantizationType.F16,
                 "bf16": gguf.GGMLQuantizationType.BF16, "q8_0": gguf.GGMLQuantizationType.Q8_0}
        TildeModel.base_dir = args.base
        TildeModel.ckpt_dir = args.checkpoint
        TildeModel.report = report
        TildeModel.temperature = args.temperature
        TildeModel.marker_start = int(report["marker_start"])
        TildeModel.head_type = types[args.head_type]
        TildeModel.lexical_type = types[args.lexical_type]
        TildeModel.extra_metadata = {
            "bos_token_id": tokinfo["bos_token_id"],
            "pad_token_id": tokinfo["pad_token_id"],
            "base_model": "inclusionAI/Ling-3.0-tiny-base",
            "base_revision": base_rev,
            "checkpoint": str(args.checkpoint.resolve().relative_to(args.checkpoint.resolve().parent.parent)),
            "checkpoint_report_sha256": sha256_file(args.checkpoint / "report.json"),
            "checkpoint_head_sha256": verified.get("joint_head.safetensors", ""),
            "hf_repo": HF_MODEL,
            "hf_gguf_repo": HF_GGUF,
        }
        ftype = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
                 "bf16": gguf.LlamaFileType.MOSTLY_BF16, "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}[args.outtype]
        hparams = ModelBase.load_hparams(work, False)
        model = TildeModel(work, ftype, args.outfile, hparams=hparams, model_name=args.model_name,
                           dry_run=args.dry_run, small_first_shard=False)
        model.write()
        logger.info(f"wrote {model.fname_out}")


if __name__ == "__main__":
    main()
