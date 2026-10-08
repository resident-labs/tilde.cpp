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
| MXFP4E | `tilde-8b-a1b-MXFP4E.gguf` | MXFP4 | Q8_0 | f32 | 5.4 GB | Blackwell GPUs (native FP4) |
| BF16 | `tilde-8b-a1b-BF16.gguf` | bf16 | bf16 | f32 | 16.2 GB | reference |

Every preset is gated against the PyTorch reference on DEV data (see "Parity"). The token embeddings stay in host
memory, so they cost no VRAM.

## Build

```sh
git clone https://github.com/resident-labs/tilde.cpp && cd tilde.cpp
cmake -B build -DGGML_CUDA=ON          # NVIDIA (CUDA 12.4+; int8 attention needs Ampere or newer)
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
    `ggml_prec_set_src`;
  - a chunked gated-delta-net prefill (`ggml-cuda/gated_delta_net_chunked.cu`);
  - non-GQA 192/128 MMA flash attention;
  - a transposed-concat fast path.
- `tilde/`: the converter, quantization presets, `tilde-server`, and the parity and benchmark harness
  (`tilde/README.md`).

## Parity

See [STATUS.md](STATUS.md) for the per-preset table. In short, every shipped preset is within 0.5 accuracy points and
+0.005 log loss of the bf16 PyTorch reference on 7,365 DEV fields.

## Benchmarks

See [STATUS.md](STATUS.md). All numbers there are measured; none are extrapolated.

## License

MIT, like llama.cpp (see LICENSE). The model weights carry their own license on Hugging Face.
