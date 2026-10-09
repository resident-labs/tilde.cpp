#!/usr/bin/env python3
"""Convert a Tilde Omni checkpoint (tilde-4b-omni) into one GGUF file (the text model, the joint head and the point
head; the vision and audio towers go into the mmproj file, see convert_tilde_omni_mmproj.py).

Tilde Omni is TLive-Omni-4B (TaoLiveAIGC/TLive-Omni-4B: a Qwen3.5-4B text decoder, the Qwen3.5 ViT and the AuT audio
tower) trained from base with the Tilde joint decision head (tilda.qwen35, checkpoint format qwen35_joint_v1):

    dense_delta.safetensors   bf16 deltas of the trained decoder linears (served weight = bf16(base + delta))
    markers.safetensors       the 263 Tilda marker embeddings (ids marker_start .. marker_start + 262), fp32
    joint_head.safetensors    the joint decision head, fp32
    vision.safetensors        the vision bridge (its <image start/end>, <point query/none> markers) and the point head
    audio.safetensors         the audio bridge (its <image start/end> markers for audio blocks)
    report.json, tokenizer/

The vocabulary of the GGUF is the checkpoint tokenizer's (base + markers) plus 7 internal ids the tokenizer never
produces: the 5 tilda-vl ids (image start, image end, image token, point query, point none; tilda.clef_joint.
vision_token_ids) and 2 ids for the start / end markers of audio blocks (the reference fills the same placeholder ids
with the audio module's markers). Their rows of the token table hold those marker embeddings; the row of point none in
the output table is the point head's none_lexical vector, and the token-type table gets a 4th row (point fields).

The temperature: --temperature is the total temperature to serve at. A folded checkpoint already holds part of it
(final_folded on HF: 1.28 until it is re-uploaded; a re-folded copy: 1.40), and the converter folds only the
remaining factor. The temperature already folded is measured, not read from the report (which a re-fold may leave
stale): --unfolded-head names the joint head of the unfolded checkpoint (final/joint_head.safetensors), and the
folded temperature is exp(joint_logit_scale(final) - joint_logit_scale(folded)), checked against the residual
scorer's output weights (folded = final / T). --checkpoint-temperature states it instead (1.0: an unfolded
checkpoint).

    python tilde/convert_tilde_omni_to_gguf.py --base TLive-Omni-4B/ --checkpoint final_folded/ --temperature 1.40 \\
        --unfolded-head final/joint_head.safetensors --outtype bf16 --outfile tilde-4b-omni-BF16.gguf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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
from conversion.qwen import Qwen3_5TextModel  # noqa: E402

logger = logging.getLogger("tilde-omni-convert")

TILDE_FORMAT_VERSION = 1
STRUCTURE_TOKENS = ("<|tilda_start|>", "<|tilda_state|>", "<|tilda_state_end|>",
                    "<|tilda_question|>", "<|tilda_options|>", "<|tilda_decision|>")
OPTION_TOKENS = tuple(f"<|tilda_option_{index:03d}|>" for index in range(255))
NOUL_TOKENS = ("<|tilda_no|>", "<|tilda_yes|>")
ALL_TOKENS = STRUCTURE_TOKENS + OPTION_TOKENS + NOUL_TOKENS
# tilda.vision.common.VISION_TOKENS, right after the marker block; then the two audio block markers (tilde.cpp only)
VISION_TOKENS = ("image_start", "image_end", "image_token", "point_query", "point_none")
AUDIO_TOKENS = ("audio_start", "audio_end")
VISION_MARKERS = ("image_start", "image_end", "point_query", "point_none")  # tilda.vision.model.MARKERS
# tilda.vision.point.PointHead module -> GGUF name (point.<name>.weight / .bias)
POINT_TENSORS = {"query.0": "query_norm", "query.1": "query", "key.0": "key_norm", "key.1": "key",
                 "pair_norm": "pair_norm", "sub_query": "sub_query", "size.0": "size_up", "size.2": "size_down",
                 "location.0": "loc_up", "location.2": "loc_down"}
CLEF_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
HIDDEN = 2560
JOINT_HEAD_CONFIG = dict(hidden_size=HIDDEN, width=1024, routing_layers=2, layers=4, heads=16, feedforward=4096)
HEAD_NORM_EPS = 1e-5  # torch.nn.LayerNorm default, used by every norm of the head
LOGIT_SCALE_CLAMP = math.log(100.0)
TEXT_PREFIX = "model."  # TLive-Omni keeps the text decoder under model.* (visual.*, audio_tower.* are the towers)

# Request limits written into the file (the server reads them; it rejects, never truncates).
MAX_REQUEST_TOKENS = 65536
MAX_QUESTION_TOKENS = 32768
MAX_QUESTIONS = 32
MAX_OPTIONS = 255

ORG = "Resident Labs"
ORG_URL = "https://residentlabs.dev/"
REPO_URL = "https://github.com/resident-labs/tilde.cpp"
BASE_REPO = "TaoLiveAIGC/TLive-Omni-4B"


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
        self.shapes: dict[str, tuple[int, ...]] = {}
        self._open = []
        for path in files:
            st = gguf.utility.SafetensorsLocal(path)
            tensors = st.__enter__()
            self._open.append(st)
            for name in tensors.keys():
                data = tensors[name]
                self.loaders[name] = (lambda data=data: LazyTorchTensor.from_local_tensor(data))
                self.shapes[name] = tuple(data.shape)

    def __contains__(self, name: str) -> bool:
        return name in self.loaders

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.loaders[name]()

    def keys(self):
        return self.loaders.keys()


def eager(t: torch.Tensor) -> torch.Tensor:
    return LazyTorchTensor.to_eager(t)


@ModelBase.register("TildeOmniModel")
class TildeOmniModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.TILDE_OMNI
    no_mtp = True  # no text is generated

    # set by main() before construction
    base_dir: Path
    ckpt_dir: Path
    report: dict
    extra_temperature: float
    total_temperature: float
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
        cls._original_block_count = n_layers
        cls.opt_num_mtp_layers = 0
        base_index = json.loads((cls.base_dir / "model.safetensors.index.json").read_text())["weight_map"]
        base = SafetensorsIndex(sorted({cls.base_dir / part for part in base_index.values()}))
        delta = SafetensorsIndex([cls.ckpt_dir / "dense_delta.safetensors"])
        markers = SafetensorsIndex([cls.ckpt_dir / "markers.safetensors"])
        head = SafetensorsIndex([cls.ckpt_dir / "joint_head.safetensors"])
        vision = SafetensorsIndex([cls.ckpt_dir / "vision.safetensors"])
        audio = SafetensorsIndex([cls.ckpt_dir / "audio.safetensors"])
        self._sources = (base, delta, markers, head, vision, audio)

        start = cls.marker_start
        n_markers = markers.shapes["special_embeddings.weight"][0]
        if n_markers != len(ALL_TOKENS):
            raise ValueError(f"expected {len(ALL_TOKENS)} marker rows, found {n_markers}")
        if vision.shapes["bridge.markers.weight"] != (len(VISION_MARKERS), HIDDEN):
            raise ValueError("unexpected vision markers")
        if audio.shapes["markers"] != (len(AUDIO_TOKENS), HIDDEN):
            raise ValueError("unexpected audio markers")
        n_vocab = start + n_markers + len(VISION_TOKENS) + len(AUDIO_TOKENS)
        none_id = start + n_markers + VISION_TOKENS.index("point_none")

        def media_rows() -> torch.Tensor:
            v = eager(vision["bridge.markers.weight"]).float()
            a = eager(audio["markers"]).float()
            rows = {"image_start": v[0], "image_end": v[1], "image_token": torch.zeros(HIDDEN), "point_query": v[2],
                    "point_none": v[3], "audio_start": a[0], "audio_end": a[1]}
            return torch.stack([rows[name] for name in VISION_TOKENS + AUDIO_TOKENS])

        tensors: dict[str, Callable[[], torch.Tensor]] = {}
        used_delta: set[str] = set()
        for name in base.keys():
            if not (name.startswith(TEXT_PREFIX) or name == "lm_head.weight"):
                continue  # visual.*, audio_tower.*: the mmproj file
            short = name[len(TEXT_PREFIX):]
            if short.startswith("layers.") and int(short.split(".")[1]) >= n_layers:
                continue
            if short in delta:
                if tuple(delta.shapes[short]) != tuple(base.shapes[name]):
                    raise ValueError(f"delta shape mismatch for {name}")
                used_delta.add(short)
                # tilda.qwen35.train.checkpointing.apply_dense_delta: bf16(fp32(base) + fp32(delta)), as served
                tensors[name] = (lambda name=name, short=short:
                                 (base[name].float() + delta[short].float()).to(torch.bfloat16))
            elif name == TEXT_PREFIX + "embed_tokens.weight":
                # base rows below the marker block, the trained markers, then the media markers
                tensors[name] = (lambda name=name: torch.cat(
                    [eager(base[name][:start]).float(), eager(markers["special_embeddings.weight"]).float(),
                     media_rows()], 0))
            elif name == "lm_head.weight":
                # the frozen original output table: the head reads the rows of the option tokens. Rows past the base
                # vocabulary are zero, except point none, whose lexical vector is the point head's (tilda.vision.joint)
                def lexical(name=name):
                    table = eager(base[name])
                    vocab = table.shape[0]
                    out = torch.zeros((n_vocab, table.shape[1]), dtype=table.dtype)
                    out[:min(vocab, n_vocab)] = table[:n_vocab]
                    # tilda.vision.joint: the none marker reads none_lexical only past the base table
                    if none_id >= vocab:
                        out[none_id] = eager(vision["point_head.none_lexical"]).to(table.dtype)
                    return out
                tensors[name] = lexical
            else:
                tensors[name] = (lambda name=name: base[name])
        if "lm_head.weight" not in tensors:
            # a tied base would read its (untrained) embedding table; the trainer's untied path is what this supports
            raise ValueError("the base has no lm_head.weight (tied embeddings are not supported)")
        unused = set(delta.keys()) - used_delta
        if unused:
            raise ValueError(f"dense delta keys not in the base: {sorted(unused)[:8]}")

        for key in head.keys():
            tensors["joint_head." + key] = (lambda key=key: head[key])

        # the point head (unified): everything but its sub-patch keys (mmproj) and its two field vectors (above)
        for src, dst in POINT_TENSORS.items():
            for suffix in ("weight", "bias"):
                if f"point_head.{src}.{suffix}" in vision:
                    tensors[f"point.{dst}.{suffix}"] = (lambda key=f"point_head.{src}.{suffix}": vision[key])

        out = {}
        for name, gen in tensors.items():
            if titem := self.filter_tensors((name, gen)):
                out[titem[0]] = titem[1]
        return out

    def modify_tensors(self, data_torch, name: str, bid: int | None):
        if name.startswith("point."):
            yield name, data_torch
            return
        if name.startswith("joint_head."):
            yield from self._modify_head(data_torch, name)
            return
        yield from super().modify_tensors(data_torch, name, bid)

    def _modify_head(self, data_torch, name: str):
        parts = name.split(".")
        # learned scalars, stored as the values used at inference, with the remaining temperature folded in
        if len(parts) == 2 and parts[1] in ("prior_logit_scale", "joint_logit_scale", "residual_gate"):
            value = float(eager(data_torch))
            if parts[1] == "residual_gate":
                self._scales[parts[1]] = 1.0 / (1.0 + math.exp(-value))
            else:
                self._scales[parts[1]] = math.exp(min(value, LOGIT_SCALE_CLAMP))
            if len(self._scales) == 3:
                T = type(self).extra_temperature
                # logit = prior_scale * prior + gate * (joint_scale * cosine + residual); logit / T is exact with
                # prior_scale / T and gate / T
                scales = [self._scales["prior_logit_scale"] / T, self._scales["joint_logit_scale"],
                          self._scales["residual_gate"] / T]
                logger.info(f"head scales (extra T={T} folded): {scales}")
                yield self.format_tensor_name(gguf.MODEL_TENSOR.DECISION_SCALES, suffix=""), \
                    torch.tensor(scales, dtype=torch.float32)
            return
        if parts[1] == "type_embedding":
            # the point type (tilda.vision.joint.POINT_TYPE = 3) is the point head's field_type row
            vision = self._sources[4]
            table = torch.cat([eager(data_torch).float(), eager(vision["point_head.field_type"]).float()[None]], 0)
            yield self.map_tensor_name(name), table
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
        if name.startswith("joint_head.") or new_name.startswith(("decision.", "dec.", "token_types", "point.")):
            return type(self).head_type
        if new_name == "output.weight":
            return type(self).lexical_type
        if new_name == "token_embd.weight":
            return gguf.GGMLQuantizationType.BF16  # the input table holds the trained markers: never quantized
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    # ---------------------------------------------------------------- metadata

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        w = self.gguf_writer
        cls = type(self)
        report = cls.report
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
        s("backbone_family", report["backbone_family"])
        u("marker_start", cls.marker_start)
        u("marker_count", len(ALL_TOKENS))
        w.add_array("tilde.structure_token_ids", [cls.marker_start + i for i in range(len(STRUCTURE_TOKENS))])
        media0 = cls.marker_start + len(ALL_TOKENS)
        w.add_array("tilde.vision_token_ids", [media0 + i for i in range(len(VISION_TOKENS))])
        w.add_array("tilde.audio_marker_ids", [media0 + len(VISION_TOKENS) + i for i in range(len(AUDIO_TOKENS))])
        u("max_request_tokens", MAX_REQUEST_TOKENS)
        u("max_question_tokens", MAX_QUESTION_TOKENS)
        u("max_questions", MAX_QUESTIONS)
        u("max_options", MAX_OPTIONS)
        w.add_float32("tilde.temperature_folded", cls.total_temperature)
        for key, value in cls.extra_metadata.items():
            if isinstance(value, str):
                s(key, value)
            elif isinstance(value, bool):
                w.add_bool("tilde." + key, value)
            elif isinstance(value, int):
                w.add_int32("tilde." + key, value) if value < 0 else u(key, value)
            elif isinstance(value, float):
                w.add_float32("tilde." + key, value)

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only)
        w = self.gguf_writer
        w.add_organization(ORG)
        w.add_url(ORG_URL)
        w.add_source_url(REPO_URL)
        w.add_base_model_count(1)
        w.add_base_model_name(0, "TLive-Omni-4B")
        w.add_base_model_organization(0, "TaoLiveAIGC")
        w.add_base_model_repo_url(0, f"https://huggingface.co/{BASE_REPO}")


def stage_dir(base: Path, ckpt: Path, report: dict, work: Path) -> Path:
    """A directory with what the converter reads besides the weights: the text config (vocabulary grown by the marker
    block and the media ids, no MTP layer, architecture renamed) and the checkpoint's tokenizer."""
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads((base / "config.json").read_text())
    text = dict(config["text_config"])
    if text.get("model_type") != "qwen3_5_text":
        raise SystemExit("the base has no Qwen3.5 text decoder")
    start = int(report["marker_start"])
    text["vocab_size"] = start + len(ALL_TOKENS) + len(VISION_TOKENS) + len(AUDIO_TOKENS)
    text["architectures"] = ["TildeOmniModel"]
    text["model_type"] = "qwen3_5_text"
    text["tie_word_embeddings"] = False
    text["mtp_num_hidden_layers"] = 0
    (work / "config.json").write_text(json.dumps(text, indent=1))
    for path in (ckpt / "tokenizer").iterdir():
        shutil.copy(path, work / path.name)
    return work


def check_tokenizer(work: Path, start: int) -> dict:
    """The marker ids must follow the base vocabulary exactly (tilda.joint_serving.load_joint_checkpoint)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(work, trust_remote_code=False)
    ids = [tok.convert_tokens_to_ids(t) for t in ALL_TOKENS]
    if ids != list(range(start, start + len(ALL_TOKENS))):
        raise SystemExit("checkpoint marker ids do not match its tokenizer")
    if len(tok) != start + len(ALL_TOKENS):
        raise SystemExit(f"tokenizer has {len(tok)} tokens, expected the marker block to end it")
    return {"bos_token_id": -1 if tok.bos_token_id is None else int(tok.bos_token_id),
            "pad_token_id": -1 if tok.pad_token_id is None else int(tok.pad_token_id)}


def measure_folded_temperature(unfolded: Path, folded: Path) -> float:
    """The temperature folded into a joint head: exp(joint_logit_scale(unfolded) - joint_logit_scale(folded)), with the
    residual scorer's output layer as a check (tilda's fold divides it by T)."""
    from safetensors import safe_open
    with safe_open(str(unfolded), "pt") as a, safe_open(str(folded), "pt") as b:
        if set(a.keys()) != set(b.keys()):
            raise SystemExit(f"{unfolded} and {folded} are not the same head")
        T = math.exp(float(a.get_tensor("joint_logit_scale")) - float(b.get_tensor("joint_logit_scale")))
        T_prior = math.exp(float(a.get_tensor("prior_logit_scale")) - float(b.get_tensor("prior_logit_scale")))
        if abs(T_prior / T - 1) > 1e-4:
            raise SystemExit(f"inconsistent fold: joint_logit_scale says T={T:.6f}, prior_logit_scale says {T_prior:.6f}")
        changed = []
        for name in sorted(a.keys()):
            x, y = a.get_tensor(name).double(), b.get_tensor(name).double()
            if x.shape != y.shape:
                raise SystemExit(f"{name}: shapes differ between the unfolded and folded heads")
            if not torch.equal(x, y):
                changed.append(name)
        w = "residual_scorer.3.weight"
        if abs(T - 1) > 1e-6:
            x, y = a.get_tensor(w).double(), b.get_tensor(w).double()
            ratio = float((x * y).sum() / (y * y).sum())
            if abs(ratio / T - 1) > 1e-3:
                raise SystemExit(f"inconsistent fold: joint_logit_scale says T={T:.6f}, {w} says {ratio:.6f}")
    logger.info(f"folded temperature measured from the heads: {T:.6f} (tensors changed by the fold: {changed})")
    return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="TLive-Omni-4B directory (HF layout)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Tilde Omni checkpoint directory (final_folded)")
    ap.add_argument("--temperature", type=float, required=True,
                    help="total temperature to serve at (1.40: the full-DEV fit of omni_full_tlive4b_20261007)")
    fold = ap.add_mutually_exclusive_group(required=True)
    fold.add_argument("--unfolded-head", type=Path,
                      help="joint_head.safetensors of the unfolded checkpoint (final/): measures the folded temperature")
    fold.add_argument("--checkpoint-temperature", type=float,
                      help="the temperature already folded into --checkpoint (1.0: not folded)")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--outtype", choices=("bf16", "f16", "f32", "q8_0"), default="bf16")
    ap.add_argument("--head-type", choices=("f32", "f16"), default="f32", help="type of the joint head tensors")
    ap.add_argument("--lexical-type", choices=("bf16", "f16", "f32", "q8_0"), default="bf16",
                    help="type of the frozen output table the head reads option-token vectors from")
    ap.add_argument("--model-name", default="tilde-4b-omni")
    ap.add_argument("--no-verify", action="store_true", help="skip the sha256 checks of the checkpoint and base files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not (math.isfinite(args.temperature) and args.temperature > 0):
        raise SystemExit("temperature must be finite and positive")
    report = json.loads((args.checkpoint / "report.json").read_text())
    if report.get("serialization") != "clef_joint_bare_v1" or report.get("clef_revision") != CLEF_REVISION:
        raise SystemExit("not a clef_joint_bare_v1 joint checkpoint")
    if report.get("backbone_family") != "qwen35" or report.get("checkpoint_format") != "qwen35_joint_v1":
        raise SystemExit("not a qwen35 joint checkpoint")
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
    spec = report.get("vision_spec") or {}
    if spec.get("point_head") != "unified" or spec.get("rope", "1d") != "1d" or spec.get("order_features"):
        raise SystemExit(f"unsupported vision spec {spec}")
    audio_spec = report.get("audio_spec") or {}
    if not audio_spec.get("native") or audio_spec.get("acif") or audio_spec.get("l12"):
        raise SystemExit(f"unsupported audio spec {audio_spec}")
    reported = float((report.get("temperature_fold") or {}).get("temperature") or 1.0)
    if args.unfolded_head is not None:
        folded = measure_folded_temperature(args.unfolded_head, args.checkpoint / "joint_head.safetensors")
    else:
        folded = args.checkpoint_temperature
        if not (math.isfinite(folded) and folded > 0):
            raise SystemExit("checkpoint temperature must be finite and positive")
    if abs(folded / reported - 1) > 1e-3:
        logger.warning(f"report.json says temperature {reported} is folded, the head holds {folded:.6f}: using the head")
    extra = args.temperature / folded
    if extra < 1 - 1e-6:
        logger.warning(f"the checkpoint holds temperature {folded:.6f} > --temperature {args.temperature}: "
                       f"folding {extra:.6f} < 1 (sharpening)")
    logger.info(f"checkpoint temperature {folded}, serving temperature {args.temperature}: folding {extra:.6f}")

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

    with tempfile.TemporaryDirectory(prefix="tilde-omni-convert-") as tmp:
        work = stage_dir(args.base, args.checkpoint, report, Path(tmp) / "model")
        tokinfo = check_tokenizer(work, int(report["marker_start"]))
        types = {"f32": gguf.GGMLQuantizationType.F32, "f16": gguf.GGMLQuantizationType.F16,
                 "bf16": gguf.GGMLQuantizationType.BF16, "q8_0": gguf.GGMLQuantizationType.Q8_0}
        TildeOmniModel.base_dir = args.base
        TildeOmniModel.ckpt_dir = args.checkpoint
        TildeOmniModel.report = report
        TildeOmniModel.extra_temperature = extra
        TildeOmniModel.total_temperature = args.temperature
        TildeOmniModel.marker_start = int(report["marker_start"])
        TildeOmniModel.head_type = types[args.head_type]
        TildeOmniModel.lexical_type = types[args.lexical_type]
        TildeOmniModel.extra_metadata = {
            "bos_token_id": tokinfo["bos_token_id"],
            "pad_token_id": tokinfo["pad_token_id"],
            "base_model": BASE_REPO,
            "base_revision": report.get("base_identity", {}).get("revision", ""),
            "checkpoint": str(args.checkpoint.resolve().relative_to(args.checkpoint.resolve().parent.parent)),
            "checkpoint_report_sha256": sha256_file(args.checkpoint / "report.json"),
            "checkpoint_head_sha256": verified.get("joint_head.safetensors", ""),
            "checkpoint_temperature": folded,
            "checkpoint_temperature_source": "measured" if args.unfolded_head is not None else "stated",
        }
        ftype = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
                 "bf16": gguf.LlamaFileType.MOSTLY_BF16, "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}[args.outtype]
        hparams = ModelBase.load_hparams(work, False)
        model = TildeOmniModel(work, ftype, args.outfile, hparams=hparams, model_name=args.model_name,
                               dry_run=args.dry_run, small_first_shard=False)
        model.write()
        logger.info(f"wrote {model.fname_out}")


if __name__ == "__main__":
    main()
