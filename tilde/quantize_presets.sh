#!/usr/bin/env bash
# Make the Tilde quant presets from the bf16 GGUF (tilde/convert_tilde_to_gguf.py --outtype bf16).
#
#   tilde/quantize_presets.sh BIN_DIR IN-BF16.gguf OUT_DIR PRESET...
#
# Presets (every one keeps the joint head in f32 and the token embeddings in bf16; the embeddings stay in host
# memory, so they cost no VRAM):
#   Q8_0     everything else Q8_0, the head's lexical table bf16             (default: any GPU with >= 12 GB, CPUs)
#   Q6_KE    routed experts Q6_K, attention/dense/shared experts Q8_0          (16 GB cards with headroom)
#   Q4_KE    routed experts Q4_K, attention/dense/shared experts Q8_0          (8-12 GB cards, Macs)
#   MXFP4E   routed experts MXFP4, attention/dense/shared experts Q8_0         (experimental: fails the parity gates)
#   NVFP4E   routed experts NVFP4, attention/dense/shared experts Q8_0         (Blackwell sm_120: native FP4 MMA)
# The file names follow the release layout: tilde-8b-a1b-<PRESET>.gguf
set -euo pipefail

BIN=$1; IN=$2; OUT=$3; shift 3
mkdir -p "$OUT"

common=(--token-embedding-type bf16 --output-tensor-type bf16
        --tensor-type '^(decision\.|dec\.|token_types)=f32')

for preset in "$@"; do
    case $preset in
        Q8_0)   extra=() ;;
        Q6_KE)  extra=(--tensor-type 'ffn_(gate_up|gate|up|down)_exps=q6_K') ;;
        Q4_KE)  extra=(--tensor-type 'ffn_(gate_up|gate|up|down)_exps=q4_K') ;;
        MXFP4E) extra=(--tensor-type 'ffn_(gate_up|gate|up|down)_exps=mxfp4') ;;
        NVFP4E) extra=(--tensor-type 'ffn_(gate_up|gate|up|down)_exps=nvfp4') ;;
        *) echo "unknown preset $preset" >&2; exit 1 ;;
    esac
    "$BIN/llama-quantize" "${common[@]}" "${extra[@]}" "$IN" "$OUT/tilde-8b-a1b-$preset.gguf" Q8_0 8
done
