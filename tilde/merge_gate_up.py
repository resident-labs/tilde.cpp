#!/usr/bin/env python3
"""Rewrite a Tilde GGUF with the routed experts' gate and up merged into one ffn_gate_up_exps tensor per layer.

The merged tensor holds the gate rows then the up rows of every expert (llama.cpp's merged layout), byte for byte, so
any quant type is merged without requantizing and the weights are unchanged. The runtime then runs one expert matmul
for gate and up instead of two. tilde/convert_tilde_to_gguf.py writes the merged layout directly; this script converts
files made before that.

    python tilde/merge_gate_up.py IN.gguf OUT.gguf
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

GATE = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.input)
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    tensors = {t.name: t for t in reader.tensors}
    plan = []  # (name, [source tensors])
    for t in reader.tensors:
        m = GATE.match(t.name)
        if m:
            up = tensors.get(f"blk.{m.group(1)}.ffn_up_exps.weight")
            if up is None:
                raise SystemExit(f"{t.name} without an up tensor")
            if up.tensor_type != t.tensor_type or list(up.data.shape) != list(t.data.shape):
                raise SystemExit(f"layer {m.group(1)}: gate and up differ in type or shape; requantize instead")
            plan.append((f"blk.{m.group(1)}.ffn_gate_up_exps.weight", [t, up]))
        elif re.match(r"^blk\.\d+\.ffn_up_exps\.weight$", t.name):
            continue
        else:
            plan.append((t.name, [t]))
    if not any(len(src) == 2 for _, src in plan):
        raise SystemExit("no gate/up expert pairs found (already merged?)")

    for name, src in plan:
        shape = list(src[0].data.shape)
        if len(src) == 2:
            shape[-2] *= 2  # rows (numpy order: [experts, rows, row bytes or elements])
        nbytes = sum(s.data.nbytes for s in src)
        writer.add_tensor_info(name, shape, src[0].data.dtype, nbytes, src[0].tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, src in plan:
        data = src[0].data if len(src) == 1 else np.concatenate([s.data for s in src], axis=-2)
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {args.output}: {sum(len(s) == 2 for _, s in plan)} merged layers")


if __name__ == "__main__":
    main()
