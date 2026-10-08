# tilde.cpp status

## What upstream llama.cpp already covers (base tag b11483) and what tilde.cpp adds

Upstream already has:
- the Ling-3.0 backbone (`bailingmoe3`: KDA through `ggml_gated_delta_net`, MLA, MoE);
- a Clef joint decision head (`clef` arch, the same Cloudflare JointSchemaHead that Tilde's "fast" head computes);
- the `/v1/systemone` endpoint in `llama-server` (`tools/server/server-decision.cpp`) with decision-order spans per token.

tilde.cpp adds:
- a `tilde` arch (`src/models/tilde.cpp`). It combines the bailingmoe3 backbone and the Clef head, with:
  - non-causal MLA over the whole request;
  - K and V decompressed from the latent (192/128 head dims instead of 576/512);
  - zero-state KDA, so no memory module is needed;
  - a head that pools spans without `[n_tokens, n_options]` matrices and uses flash attention for the head's
    cross-attention, so that 64k-token requests fit;
- a converter (`tilde/convert_tilde_to_gguf.py`) from base + checkpoint to one GGUF. It folds T into the head.
- still to do: a `tilde` decision type in `server-decision.cpp` (the Tilda serialization, noul format, limits and
  response extras) and the speed work (CUDA kernels).

## Phase

Phase 2 (model support, CPU parity) is in progress. Phase 1 (fork + converter) is done.

## What works

- Fork of llama.cpp b11483 (`upstream` remote = ggml-org/llama.cpp), branch `main`.
- Converter: base (`inclusionAI/Ling-3.0-tiny-base` @ bab7297) + t40 `stage1_final` produce `tilde-8b-a1b-BF16.gguf`
  (16.2 GB). It streams, verifies the checkpoint and base sha256, and folds T = 1.26 (head scales 0 and 2 divided by T).
- CPU: `tilde-eval` loads the GGUF and returns per-option logits for pre-encoded DEV records.

## Parity

Reference: the PyTorch t40 checkpoint (`load_joint_checkpoint` + `forward_joint`, batch 1, bf16 autocast, the serving
computation). Its logits are divided by T = 1.26, and the GGUF has T folded into the head. Inputs are DEV records from
tilda41_20261006 `dev_orig40_clean.jsonl`, encoded by the trainer's own DEV encoder (`encode_training_records`) and
selected as a hash-based 1/4 subset. These are private rows; the files live only on hermeslab in `~/tilde-work/data`.

| runtime | records | fields | accuracy (ref -> test) | log loss delta | mean abs dp | p99 abs dp | max abs dp | argmax agreement |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| CPU, BF16 GGUF (i7-12700K) | 318 | 405 | .8741 -> .8765 | +0.0002 | .0045 | .039 | .111 | 99.5% |
| CUDA, BF16 GGUF (3090 Ti) | 5,235 | 7,365 | .8846 -> .8847 | +0.0006 | .0044 | .047 | .268 | 99.62% |

Request encoder (llama-server's tilde type vs `tilda.joint_serving.encode_request`): token-exact and span-exact on all
3,015 accepted requests (3,000 DEV states plus edge cases). Rejected requests are refused by both encoders with the
same error texts.

## Benchmarks

Not yet measured.

## Open problems

- The 192/128 non-GQA flash-attention kernel is missing on CUDA (upstream only has the GQA-8/16 variant).

## Next step

Measure CPU parity against the PyTorch reference on DEV records, then build the CUDA backend.
