# tilde/: Tilde-specific tools

| path | what |
|---|---|
| `convert_tilde_to_gguf.py` | base model + Tilde checkpoint -> one GGUF (T folded into the head, `tilde.*` metadata) |
| `quantize_presets.sh` | bf16 GGUF -> the release presets (Q8_0, Q4_KE, MXFP4E, ...) |
| `tools/tilde-server.cpp` | `tilde-server`: llama-server with Tilde defaults picked from the hardware |
| `tools/tilde-eval.cpp` | runs pre-encoded records, prints per-option logits; `--profile` times every graph node |
| `tools/tilde-encode.cpp` | encodes `/v1/systemone` bodies exactly as llama-server does (encoder parity) |
| `parity/reference.py` | encode DEV records with the trainer's encoder, run the PyTorch reference, compare |
| `parity/encoder_parity.py` | llama-server's request encoder against `tilda.joint_serving.encode_request` |
| `bench/make_bench.py` | benchmark requests at exact token lengths from any long text |

See the top-level README.md for how to install and run Tilde.
