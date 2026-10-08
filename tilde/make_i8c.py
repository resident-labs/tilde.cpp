#!/usr/bin/env python3
"""Make the CUDA "I8C" Tilde GGUF: the dense matmul weights as int8 with one f32 scale per output row (W8 per channel),
for the int8 tensor-core GEMM path (per-token int8 activations, int32 accumulation, scales applied in the epilogue).
Everything else is copied from a source GGUF (normally the Q8_0 preset; the int8 rows are made from the BF16 file).

    python tilde/make_i8c.py --bf16 tilde-8b-a1b-BF16.gguf --base tilde-8b-a1b-Q8_0.gguf --out tilde-8b-a1b-I8C.gguf

A weight W [rows, K] becomes `<name>.weight` (I8, codes q = round(W / s), |q| <= 127) and `<name>.scale`
(F32 [rows], s = max|W_row| / 127). Only CUDA (Ampere or newer, built with CUTLASS) runs this file.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

# dense matmul weights of the backbone (not the experts, router, embeddings, output table, head, norms or conv; the
# 16-row beta and attention-gate projections stay Q8_0)
DENSE = re.compile(r"^blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|attn_q_a|attn_q_b|attn_kv_a_mqa|attn_kv_b|"
                   r"ssm_f_a|ssm_g_a|ffn_gate|ffn_up|ffn_down|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)\.weight$")


def to_f32(t) -> np.ndarray:
    return dequantize(t.data, t.tensor_type).astype(np.float32).reshape([int(x) for x in reversed(t.shape)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", type=Path, required=True, help="BF16 GGUF (source of the int8 rows)")
    ap.add_argument("--base", type=Path, required=True, help="GGUF whose other tensors are copied (Q8_0 preset)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ref = gguf.GGUFReader(args.bf16)
    src = {t.name: t for t in ref.tensors}
    base = gguf.GGUFReader(args.base)
    arch = base.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.out, arch=arch, endianess=base.endianess)
    for field in base.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_string("tilde.weight_format", "i8c")

    plan = []
    for t in base.tensors:
        if DENSE.match(t.name):
            if t.name not in src:
                raise SystemExit(f"{t.name} missing in {args.bf16}")
            rows = int(t.shape[1]) if len(t.shape) >= 2 else 1
            plan.append(("i8", t.name, src[t.name], rows))
            plan.append(("scale", t.name.replace(".weight", ".scale"), src[t.name], rows))
        else:
            plan.append(("copy", t.name, t, 0))

    for kind, name, t, rows in plan:
        if kind == "copy":
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        elif kind == "i8":
            k = int(t.shape[0])
            writer.add_tensor_info(name, [rows, k], np.dtype(np.int8), rows * k, gguf.GGMLQuantizationType.I8)
        else:
            writer.add_tensor_info(name, [rows], np.dtype(np.float32), rows * 4, gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    cache = {}
    n = 0
    for kind, name, t, rows in plan:
        if kind == "copy":
            writer.write_tensor_data(t.data, tensor_endianess=base.endianess)
            continue
        if t.name not in cache:
            w = to_f32(t)                                           # [rows, K]
            s = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127.0
            q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
            cache = {t.name: (q, s.astype(np.float32))}
            n += 1
        q, s = cache[t.name]
        writer.write_tensor_data(q if kind == "i8" else s)
    writer.close()
    print(f"wrote {args.out}: {n} dense weights as int8 + row scales")


if __name__ == "__main__":
    main()
