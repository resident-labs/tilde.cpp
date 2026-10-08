#!/usr/bin/env python3
"""Export a Tilde checkpoint as one merged Hugging Face directory (the `resident-labs/tilde-8b-a1b` layout).

    python tilde/export_hf.py --base BASE_DIR --checkpoint CKPT_DIR --temperature 1.26 --out tilde-8b-a1b/

Writes:
  model-0000N-of-0000M.safetensors   the full Ling-3.0 backbone in bf16: base + dense delta, the trained routed
                                     experts (per-expert gate/up/down, the base layout), word embeddings with the 263
                                     trained marker rows, and the frozen original output table (lm_head, padded with
                                     zero rows for the markers; the head reads option-token rows from it). No MTP layer.
  model.safetensors.index.json, config.json (vocab_size = marker_start + 263, num_nextn_predict_layers = 0)
  joint_head.safetensors             the joint decision head (fp32) with the temperature folded in:
                                     prior_logit_scale -= ln T, joint_logit_scale -= ln T, residual_scorer.3 /= T
                                     (exact; refused if a folded scale would cross the forward clamp at ln 100)
  tilde_config.json                  head config, attention mode, serialization, noul format, marker ids, limits, T
  tokenizer.json, tokenizer_config.json, special_tokens_map.json

Streams tensor by tensor (peak RAM about one MoE layer). The GGUF converter (convert_tilde_to_gguf.py) reads the same
inputs directly, so this directory is for PyTorch users; llama.cpp users take the GGUF.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ALL_TOKENS = 263
CLAMP = math.log(100.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    args = ap.parse_args()

    base, ckpt, out = args.base, args.checkpoint, args.out
    T = args.temperature
    if not (math.isfinite(T) and T > 0):
        raise SystemExit("temperature must be finite and positive")
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((ckpt / "report.json").read_text())
    config = json.loads((base / "config.json").read_text())
    start = int(report["marker_start"])
    n_layers = config["num_hidden_layers"]
    inter = config["moe_intermediate_size"]
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]

    delta = safe_open(ckpt / "dense_delta.safetensors", "pt")
    delta_keys = set(delta.keys())
    experts = safe_open(ckpt / "routed_experts.safetensors", "pt")
    markers = safe_open(ckpt / "markers.safetensors", "pt").get_tensor("special_embeddings.weight")
    if markers.shape[0] != ALL_TOKENS:
        raise SystemExit(f"expected {ALL_TOKENS} marker rows")

    pending, size, shard, names = {}, 0, 0, {}

    def flush():
        nonlocal pending, size, shard
        if pending:
            fn = f"model-{shard:05d}.safetensors"
            save_file(pending, out / fn, metadata={"format": "pt"})
            names.update({k: fn for k in pending})
            shard += 1
            pending, size = {}, 0

    def put(name, t):
        nonlocal size
        pending[name] = t.contiguous()
        size += t.numel() * t.element_size()
        if size > args.shard_gb * 2 ** 30:
            flush()

    used = set()
    for part in sorted(set(index.values())):
        with safe_open(base / part, "pt") as f:
            for key in f.keys():
                if key.startswith("model.layers.") and int(key.split(".")[2]) >= n_layers:
                    continue  # MTP layer
                if ".mlp.experts." in key:
                    continue  # replaced below
                t = f.get_tensor(key)
                short = key.removeprefix("model.")
                if short in delta_keys:
                    t = (t.float() + delta.get_tensor(short).float()).to(torch.bfloat16)
                    used.add(short)
                elif key == "model.word_embeddings.weight":
                    t = torch.cat([t[:start], markers.to(t.dtype)], 0)
                elif key == "lm_head.weight":
                    rows = start + ALL_TOKENS
                    t = torch.nn.functional.pad(t, (0, 0, 0, rows - t.shape[0])) if rows > t.shape[0] else t[:rows]
                put(key, t)
    if used != delta_keys:
        raise SystemExit(f"dense delta keys not in the base: {sorted(delta_keys - used)[:5]}")
    for key in sorted(experts.keys(), key=lambda k: (int(k.split(".")[1]), k)):
        layer = int(key.split(".")[1])
        t = experts.get_tensor(key)
        for e in range(t.shape[0]):
            p = f"model.layers.{layer}.mlp.experts.{e}."
            if key.endswith(".w_gu"):
                put(p + "gate_proj.weight", t[e, :inter].clone())
                put(p + "up_proj.weight", t[e, inter:].clone())
            else:
                put(p + "down_proj.weight", t[e].clone())
    flush()
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": names}, indent=1))

    config.update(vocab_size=start + ALL_TOKENS, num_nextn_predict_layers=0)
    if report.get("experts_per_token"):
        config["num_experts_per_tok"] = int(report["experts_per_token"])
    (out / "config.json").write_text(json.dumps(config, indent=1))
    for name in ("configuration_bailing_moe_v3.py", "modeling_bailing_moe_v3.py"):
        if (base / name).exists():
            shutil.copy(base / name, out / name)

    head = {}
    with safe_open(ckpt / "joint_head.safetensors", "pt") as f:
        for key in f.keys():
            head[key] = f.get_tensor(key)
    for key in ("prior_logit_scale", "joint_logit_scale"):
        value = head[key].double() - math.log(T)
        if float(head[key]) > CLAMP or float(value) > CLAMP:
            raise SystemExit(f"{key}: folding T={T} crosses the forward clamp at ln(100)")
        head[key] = value.to(head[key].dtype)
    for key in ("residual_scorer.3.weight", "residual_scorer.3.bias"):
        head[key] = (head[key].double() / T).to(head[key].dtype)
    save_file(head, out / "joint_head.safetensors", metadata={"format": "pt", "temperature_folded": str(T)})

    for path in (ckpt / "tokenizer").iterdir():
        shutil.copy(path, out / path.name)
    if (base / "special_tokens_map.json").exists() and not (out / "special_tokens_map.json").exists():
        shutil.copy(base / "special_tokens_map.json", out / "special_tokens_map.json")

    tilde = {
        "model": "tilde-8b-a1b",
        "base_model": "inclusionAI/Ling-3.0-tiny-base",
        "base_revision": report.get("base_identity", {}).get("revision"),
        "head_config": report["head_config"],
        "head_variant": report["head_variant"],
        "clef_revision": report["clef_revision"],
        "serialization": report["serialization"],
        "attention_mode": report["attention_mode"],
        "input_layout": report.get("input_layout", "state_first"),
        "noul_format": report["noul_format"],
        "question_prefix": report.get("question_prefix"),
        "experts_per_token": report.get("experts_per_token") or config["num_experts_per_tok"],
        "marker_start": start,
        "marker_count": ALL_TOKENS,
        "temperature_folded": T,
        "limits": {"max_request_tokens": 65536, "max_question_tokens": 32768, "max_questions": 32, "max_options": 255},
    }
    (out / "tilde_config.json").write_text(json.dumps(tilde, indent=1))
    print(f"wrote {out} ({shard} shards)")


if __name__ == "__main__":
    main()
