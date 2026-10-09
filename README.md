# tilde.cpp

Run **Tilde**, the Resident Labs decision model, on your own machine: any NVIDIA, AMD, Apple or Intel GPU, or just a CPU.
tilde.cpp is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (base tag b11483). It adds the Tilde model,
a `/v1/systemone` server and the kernels that make long requests fast.

Tilde does not write text. It reads a **state** (any text or JSON) and a set of **questions**, all at once, and returns
calibrated probabilities for every question's options:

- `choice`: pick one of up to 255 options;
- `noul`: yes or no;
- `score`: a 2-10 level scale.

One forward pass answers all the questions together, up to 65,536 tokens per request.

- Model: [`resident-labs/tilde-8b-a1b`](https://huggingface.co/resident-labs/tilde-8b-a1b): 7.9B parameters, ~1.3B active
  per token (Ling-3.0-tiny backbone, 24 layers: 18 linear-attention KDA layers and 6 MLA layers that read the whole
  request bidirectionally, 128-expert MoE), plus a joint decision head.
- GGUF files: [`resident-labs/tilde-8b-a1b-GGUF`](https://huggingface.co/resident-labs/tilde-8b-a1b-GGUF).
- Website: https://residentlabs.dev/

## Quick start

```sh
# build (pick your backend, see "Build"), then:
tilde-server                     # downloads the GGUF that fits your free memory, serves on http://127.0.0.1:8080
```

```sh
curl http://127.0.0.1:8080/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Support ticket: I was charged twice for my order last week and nobody has replied.",
  "questions": {
    "team":    {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "payments and refunds", "shipping": "deliveries", "tech": "login and app problems"}},
    "angry":   {"type": "noul",   "instructions": "Is the customer angry?"},
    "urgency": {"type": "score",  "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today", "right now"]}
  }
}'
```

The response looks like this:

```json
{"model": "tilde-8b-a1b",
 "answers": {
   "team":    {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "shipping": 0.02, "tech": 0.01}, "confidence": 0.955},
   "angry":   {"type": "noul", "noul": 0.81, "probabilities": {"true": 0.81, "false": 0.19}, "confidence": 0.62},
   "urgency": {"type": "score", "score": 1.7, "legend": {"0": "can wait", "1": "this week", "2": "today", "3": "right now"},
               "probabilities": {"0": 0.06, "1": 0.31, "2": 0.49, "3": 0.14}, "confidence": 0.32}},
 "usage": {"input_tokens": 231, "output_tokens": 0}, "latency_ms": 41.2}
```

(The values above show the shape only, not a real answer.)

Docker (NVIDIA): `docker build -f tilde/docker/cuda.Dockerfile -t tilde-server:cuda .` and then
`docker run --gpus all -p 8080:8080 -v tilde-cache:/root/.cache tilde-server:cuda`.

## For agents

- Endpoint: `POST /v1/systemone`, JSON body, `Content-Type: application/json`. Readiness: `GET /health` returns 200
  once the model is loaded.
- Body:
  - `state`: string, object or array (required). Objects and arrays are given to the model as compact JSON.
  - `questions`: object of 1-32 entries; `{id: {type, instructions, criteria}}`.
    - `type` is `choice` | `noul` | `score`.
    - `choice` criteria: object `{key: description or null}`, 1-255 entries.
    - `score` criteria: array of 2-10 level descriptions, lowest first.
    - `noul` criteria: optional `{true: ..., false: ...}`.
  - Optional: `request_id` (echoed), and `noul_temperature` (> 0; per question or per request) to sharpen (< 1) or
    soften (> 1) yes/no answers.
- Answers come back in question order. `noul` is P(true). `choice` and `score` carry `probabilities` and `confidence`.
  `score.score` is the expected level index.
- Limits: 65,536 tokens per request (state + all questions), 32,768 tokens for the state plus any single question,
  32 questions, 255 options. Over-limit requests are **rejected with HTTP 400** and a message saying which limit;
  they are never truncated.
- Errors: `{"error": "<message>"}` with 400 for bad requests.
- Latency grows with request length. Plan for tens of milliseconds at 1k tokens and seconds at 32k-64k (see
  "Benchmarks").

## Which file? (presets)

`tilde-server` picks the preset by free memory. To choose one yourself, use `-hf resident-labs/tilde-8b-a1b-GGUF:<preset>`
or `-m <file>`.

| preset | file | experts | rest | head | size | for |
|---|---|---|---|---|---:|---|
| Q8_0 | `tilde-8b-a1b-Q8_0.gguf` | Q8_0 | Q8_0 | f32 | 9.3 GB | default; GPUs with >= 16 GB, CPUs |
| Q4_KE | `tilde-8b-a1b-Q4_KE.gguf` | Q4_K | Q8_0 | f32 | 5.8 GB | 8-12 GB GPUs, Macs |
| I8CX | `tilde-8b-a1b-I8CX.gguf` | int8 + row scales | int8 + row scales (projections), Q8_0 | f32 | 8.9 GB | NVIDIA Ampere or newer with a CUTLASS build: the fastest (1.18 s at 32k on a 3090 Ti with `TILDE_FA_FAST=1`) |
| NVFP4E | `tilde-8b-a1b-NVFP4E.gguf` | NVFP4 | Q8_0 | f32 | 5.8 GB | experimental (on Blackwell's native FP4 it is over the accuracy budget and slower than I8CX) |
| MXFP4E | `tilde-8b-a1b-MXFP4E.gguf` | MXFP4 | Q8_0 | f32 | 5.4 GB | experimental |
| BF16 | `tilde-8b-a1b-BF16.gguf` | bf16 | bf16 | f32 | 16.2 GB | reference |

Every preset is gated against the PyTorch reference on DEV data (see "Parity"). The token embeddings stay in host
memory, so they cost no VRAM. `tilde-server` picks I8CX with the fast attention on NVIDIA Ampere or newer (CUTLASS
build) with enough memory, else Q8_0, or Q4_KE when memory is short (see "Supported hardware").

Attention on CUDA (Ampere or newer; environment of `tilde-server`), measured on 32k requests:
- `TILDE_FA_FAST=1`, the default with I8CX: a SageAttention2-style kernel (int8 Q*K^T with block scales, f16 P*V with
  f16 accumulation and an overflow guard that reruns the layer in f32); 14% faster than the int8 one, same DEV accuracy;
- `TILDE_FA_FAST=0`: int8 Q*K^T, f16 P*V (the default for the other presets);
- `TILDE_FA_PV8=1` (with `TILDE_FA_FAST=0`): int8 P*V as well (5% faster than int8 Q*K^T alone).

## Supported hardware

What `tilde-server` picks with no flags, and what was measured (one 32,768-token request through the server, warm;
STATUS.md has the details). "Not tested" means the code path exists but nobody has run it on that hardware.

| hardware | backend | preset picked | 32k request | memory (peak, 64k limit) |
|---|---|---|---|---|
| NVIDIA Ampere, 24 GB (RTX 3090 Ti) | CUDA + CUTLASS | I8CX + fast attention | **1,244 ms** (PyTorch bf16: 2,286 ms) | 13.5 GiB |
| NVIDIA Ampere or newer, 16 GB | CUDA + CUTLASS | I8CX + fast attention | 1,249 ms (3090 Ti capped to 15.5 GiB) | 13.5 GiB |
| NVIDIA, 12 GB | CUDA | Q4_KE | 1,997 ms (3090 Ti capped to 11.5 GiB) | 10.2 GiB |
| NVIDIA, 8 GB | CUDA | Q4_KE, requests up to 19,456 tokens | 32k rejected; 16k: 866 ms (3090 Ti capped to 7.5 GiB) | 6.8 GiB |
| NVIDIA Ada (RTX 40), Hopper | CUDA + CUTLASS | I8CX + fast attention | not tested | - |
| NVIDIA Blackwell, 32 GB (RTX 5090) | CUDA 12.8+ + CUTLASS (arch 120a) | I8CX + fast attention | **640 ms** (tilde-eval: 575 ms) | 14.1 GiB |
| NVIDIA Turing / Volta (RTX 20, V100) | CUDA | Q8_0, or Q4_KE when memory is short | not tested | - |
| NVIDIA, CUDA build without CUTLASS | CUDA | Q8_0 / Q4_KE | 1,994 ms (Q8_0 on the 3090 Ti) | 13.4 GiB |
| any GPU through Vulkan (measured: RTX 3090 Ti) | Vulkan | Q8_0 / Q4_KE | 3,512 ms (older build) | 13.6 GiB |
| AMD | ROCm | Q8_0 / Q4_KE | not tested | - |
| Apple silicon | Metal | Q8_0 / Q4_KE | not tested | - |
| CPU (measured: i7-12700K, 8 threads) | CPU | Q8_0 | not measured at 32k; 16k: 117 s | 12.1 GiB RSS |

`GET /health` reports the preset, why it was picked, the attention path and the request limit. Override with a preset
tag (`-hf resident-labs/tilde-8b-a1b-GGUF:Q8_0`), `-m <file>`, or `TILDE_FA_FAST=0` for the standard int8 attention.

## Build

```sh
git clone https://github.com/resident-labs/tilde.cpp && cd tilde.cpp
cmake -B build -DGGML_CUDA=ON          # NVIDIA (CUDA 12.4+; int8 attention needs Ampere or newer)
# add -DGGML_CUDA_CUTLASS_DIR=/path/to/cutlass (v3.9.2) for the I8CX preset's int8 GEMMs
# cmake -B build -DGGML_HIP=ON         # AMD (ROCm)
# cmake -B build -DGGML_METAL=ON       # Apple (default on macOS)
# cmake -B build -DGGML_VULKAN=ON      # any Vulkan GPU (Intel, AMD, NVIDIA)
# cmake -B build                       # CPU only
cmake --build build -j --target tilde-server
./build/bin/tilde-server
```

`tilde-server` accepts every `llama-server` option, for example `--port`, `--host`, `--api-key` and `-ngl`. It sets
these defaults:

- `-c/-b/-ub` = the request limit (one request is one ubatch: the MLA layers read all of it);
- `-np 1`;
- `-fa on`;
- the model alias `tilde-8b-a1b`;
- `TILDE_CHUNK_TOKENS`, the MoE chunk size picked from free memory.

The compute buffer for the largest request is reserved at startup, so a request that fits the limits never runs out
of memory later. When the memory is too small for 65,536 tokens, the request limit is lowered (and logged), and
longer requests are rejected.

## Convert your own checkpoint

```sh
pip install -r requirements.txt   # torch, safetensors, transformers, numpy
python tilde/convert_tilde_to_gguf.py --base Ling-3.0-tiny-base/ --checkpoint stage1_final/ --temperature 1.26 \
    --outtype bf16 --outfile tilde-8b-a1b-BF16.gguf
tilde/quantize_presets.sh build/bin tilde-8b-a1b-BF16.gguf out/ Q8_0 Q4_KE MXFP4E
python tilde/export_hf.py --base Ling-3.0-tiny-base/ --checkpoint stage1_final/ --temperature 1.26 --out tilde-8b-a1b/
```

The converter folds the fitted temperature into the head, verifies the checkpoint and base hashes, and streams (a few
GB of RAM).

## What tilde.cpp adds to llama.cpp

- `src/models/tilde.cpp`: the `tilde` architecture. It has:
  - non-causal MLA with K and V decompressed from the latent;
  - KDA from zero state;
  - MoE in token chunks;
  - the joint decision head in the graph (span pooling sized for 64k tokens, flash attention for the head's
    cross-attention).
- `tools/server/server-decision.cpp`: the `tilde` decision type. It encodes requests exactly like the training code,
  token for token, applies the limits and returns the response shape above.
- CUDA kernels:
  - int8 Q*K^T flash attention for unmasked long prefill (`ggml-cuda/fattn-i8.cu`), selected through
    `ggml_prec_set_src`, with int8 P*V and a SageAttention2-style variant (`ggml-cuda/fattn-sage.cu`,
    `ggml-cuda/vendor/sageattn`, Apache-2.0);
  - int8 GEMMs for I8 weights with row scales (`ggml-cuda/mmi8.cu`, CUTLASS): dense, and grouped for the experts with
    the SwiGLU and the router-weighted sum (plus shared expert and residual) in the epilogue passes;
  - fused grouped MoE routing (`ggml-cuda/moe-route.cu`), warp-per-row RMS norms that also write the int8 rows of the
    next GEMM, residual add + RMS norm;
  - a chunked gated-delta-net prefill on TF32 tensor cores (`ggml-cuda/gated_delta_net_chunked.cu`);
  - non-GQA 192/128 MMA flash attention;
  - a transposed-concat fast path.
- `tilde/`: the converter, quantization presets, `tilde-server`, and the parity and benchmark harness
  (`tilde/README.md`).

## Parity

See [STATUS.md](STATUS.md) for the per-preset table. In short, Q8_0, Q6_KE, Q4_KE and I8CX (with every attention
option) are within 0.5 accuracy points and +0.005 log loss of the bf16 PyTorch reference on 7,365 DEV fields; NVFP4E
and MXFP4E (measured as W4A8 on Ampere) are within 1 point and +0.01.

## Benchmarks

See [STATUS.md](STATUS.md). All numbers there are measured; none are extrapolated.

## Tilde Omni (tilde-4b-omni, in progress)

`tilde_omni` serves Tilde trained on TLive-Omni-4B (Qwen3.5-4B text decoder, Qwen3.5 ViT, AuT audio tower): the same
`/v1/systemone` API plus images, audio and a `point` question type. Status and numbers: STATUS.md, "Tilde Omni".

```sh
python tilde/convert_tilde_omni_to_gguf.py --base TLive-Omni-4B/ --checkpoint final_folded/ --temperature 1.40 \
    --unfolded-head final/joint_head.safetensors --outtype bf16 --outfile tilde-4b-omni-BF16.gguf
python tilde/convert_tilde_omni_mmproj.py --base TLive-Omni-4B/ --checkpoint final_folded/ --outfile mmproj-tilde-4b-omni-F16.gguf
build/bin/llama-quantize --token-embedding-type bf16 --output-tensor-type bf16 \
    --tensor-type '^(decision\.|dec\.|token_types|point\.)=f32' tilde-4b-omni-BF16.gguf tilde-4b-omni-Q8_0.gguf Q8_0 8
```

Temperature: `--temperature` is the total temperature to serve at (1.40, the full-DEV fit). The converter folds only
what the checkpoint does not already hold, and measures that from the heads rather than trusting report.json:
`--unfolded-head` (the unfolded `final/joint_head.safetensors`) gives T_ckpt = exp(joint_logit_scale(final) -
joint_logit_scale(folded)), checked against the residual scorer (or state it with `--checkpoint-temperature`). The HF
`omni_full_tlive4b_20261007/final_folded` of 2026-10-07 holds T = 1.28 (a quick text-DEV fit), so the converter folds
1.40 / 1.28 = 1.09375 (metadata `tilde.checkpoint_temperature` 1.28, `tilde.temperature_folded` 1.40); a re-folded
`final_folded` at 1.40 gets nothing more, so a re-fetch cannot double-fold. Nothing is applied at serving time.

Request additions (tilde omni):
- `images`: a list of data URLs / base64 strings, or `{"data": ..., "label": "screenshot", "time": t, "max_cells": n}`;
  `budget_cells` (default 2400) caps the request's image cells (tilda.vision.budget: images are scaled down, never up,
  and padded to 32 px cells);
- `audio`: a list of data URLs / base64 strings (wav, mp3, flac) or `{"data": ..., "label": "audio"}`, whole clips;
- a question `{"type": "point", "instructions": "...", "image": 0}` answers with `x`, `y` (original pixels), `none`,
  `p_none`, `p_token`, `p_blob`, `p_elsewhere`, `width`, `height`, `top` (8 most likely tokens), `p_blob_t` and
  `stored_xy`, as the reference browser server (tilda.agents.browser.serve) does; the response adds `images` (scale,
  stored size and tokens of each image).
- Media order: `media_order` `"time"` (images and clips sorted by their `time`, an image before a clip at equal
  time, as the training rows with `"media_order": "time"`), `"row"` (every image, then every clip). Default: by time
  when every image and clip has a `time`, else row. `media_layout`: `"blocks"` (default; a header line per block,
  `Image k: <label>` / `Audio k: <label>`) or `"stream"` (no header lines; Qwen-style video).
- Video: an image item with `"frames": [first, second]` is one temporal pair (2 fps); `"stored_scale": s` marks an
  image already at its stored size (s = stored / original pixels; no resize). Audio items take `time` and a slice
  `start` / `end` in seconds (the whole clip is encoded once per window and the slice's tokens kept, 13 per second); `"clip": j`
  instead of `data` takes the file of audio item j (send a clip once, slice it many times).
- webp images get HTTP 415.
- Limits: 48 MB of base64 per image or clip, 40 Mpx per image, 120 Mpx per request, 32 images, 256 audio items, 3,600 s of audio to process (clip length summed over its slices).

## License

MIT, like llama.cpp (see LICENSE). The model weights carry their own license on Hugging Face.
