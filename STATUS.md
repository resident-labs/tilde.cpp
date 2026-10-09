# tilde.cpp status

Last updated 2026-10-08 08:45 CDT (commit ab5d070a2). Every number below was measured on hermeslab (RTX 3090 Ti 24 GB,
i7-12700K, CUDA 13.1) unless marked as the RTX 5090 run. Nothing is extrapolated.

## Where we are

One 32,768-token request, warm, RTX 3090 Ti:

| | ms | vs PyTorch bf16 (server) |
|---|---:|---:|
| PyTorch bf16 (`tilda/serve.py`, server) | 2,286 | 1.0x |
| start of the speed work (Q8_0, server) | 2,436 | 0.94x |
| Q8_0 now (tilde-eval / server) | 1,958 / 1,994 | 1.15x |
| I8CX now (tilde-eval / server) | 1,372 / 1,429 | 1.60x |
| I8CX+PV8 (tilde-eval / server) | 1,311 / 1,368 | 1.67x |
| **I8CX+FAST (tilde-eval / server)** | **1,180 / 1,244** | **1.84x** |
| I8CX+FAST on an RTX 5090 (tilde-eval / server) | 575 / 640 | 3.6x |

The 1,000 ms target is reached on the RTX 5090 (575 ms) but not on the 3090 Ti. I8CX+FAST is the fastest configuration inside the accuracy budget
(1.0 pt, +0.01 log loss) and is the default (Ari, 2026-10-08): `tilde-server` picks it with no flags on NVIDIA Ampere or
newer with a CUTLASS build and the memory for 65,536-token requests (otherwise Q8_0, or Q4_KE when memory is short),
logs why, and reports it in `GET /health`; a preset tag, `-m` or `TILDE_FA_FAST=0` override it. The kernel profile and
what is left are at the end.

Presets:
- **Q8_0** (default GGUF, every backend), **Q6_KE**, **Q4_KE** (12 / 8 GB cards), **NVFP4E / MXFP4E** (4-bit experts;
  Blackwell has native FP4).
- **I8CX** (CUDA, Ampere or newer, needs a build with `GGML_CUDA_CUTLASS_DIR`): dense projections and routed experts as
  int8 with a scale per output row; CUTLASS int8 GEMMs (dense with the scale in the epilogue, grouped for the experts).
  `tilde/make_i8c.py` makes it from the BF16 and Q8_0 files.
- Attention options (I8CX or any preset on CUDA): default int8 Q*K^T + f16 P*V; `TILDE_FA_PV8=1` int8 P*V;
  `TILDE_FA_FAST=1` SageAttention2-style kernel (int8 Q*K^T with block scales, f16 P*V with f16 accumulation, overflow
  guard with an f32 rerun).

## Parity (GPU, 5,235 DEV records / 7,365 fields, against the bf16 PyTorch reference, build 507f5368d)

Strict gates: accuracy at most 0.5 pt below the reference, log loss at most +0.005, mean |dp| at most 0.02, argmax
agreement at least 98%. Accuracy budget for speed presets (lead, 2026-10-08): at most 1.0 pt and +0.01 log loss.

| preset | fields | accuracy (ref -> preset) | delta (pt) | log loss delta | mean abs dp | p99 abs dp | max abs dp | argmax agreement | strict gates | budget |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| Q8_0 | 7,365 | 0.8846 -> 0.8839 | -0.07 | +0.0005 | 0.0055 | 0.053 | 0.263 | 99.39% | pass | inside |
| Q6_KE | 7,365 | 0.8846 -> 0.8855 | +0.10 | -0.0003 | 0.0064 | 0.067 | 0.354 | 99.27% | pass | inside |
| Q4_KE | 7,365 | 0.8846 -> 0.8845 | -0.01 | +0.0022 | 0.0130 | 0.143 | 0.417 | 98.64% | pass | inside |
| NVFP4E (W4A8 on Ampere) | 7,365 | 0.8846 -> 0.8824 | -0.22 | +0.0061 | 0.0161 | 0.164 | 0.513 | 98.25% | fail (log loss) | **inside** |
| MXFP4E (W4A8 on Ampere) | 7,365 | 0.8846 -> 0.8815 | -0.31 | +0.0061 | 0.0208 | 0.205 | 0.563 | 97.96% | fail (log loss, mean dp, argmax) | inside |
| I8CX | 7,365 | 0.8846 -> 0.8849 | +0.03 | +0.0013 | 0.0082 | 0.086 | 0.545 | 99.14% | pass | inside |
| I8CX+PV8 | 7,365 | 0.8846 -> 0.8847 | +0.01 | +0.0001 | 0.0083 | 0.082 | 0.593 | 99.20% | pass | inside |
| I8CX+FAST | 7,365 | 0.8846 -> 0.8846 | +0.00 | -0.0005 | 0.0084 | 0.084 | 0.889 | 99.21% | pass | inside |

Earlier builds (same records): BF16 with f16 / int8 attention -0.04 / -0.03 pt, mean |dp| .0043 / .0045; Vulkan Q8_0
(318 records) +0.49 pt, mean |dp| .0054; CPU BF16 (318 records) argmax 99.5%. The request encoder is token- and
span-exact against `tilda.joint_serving.encode_request` on 3,015 requests.

The max |dp| rows come from one record, `d3_adversarial:adv3_11~stack` (1,175 tokens, 4 linked yes/no questions): it is
bistable, every preset moves it (mean |dp| on its group: I8CX .20, PV8 .35, FAST .41), and FAST lands on the labels
where the reference does not. By length, FAST matches the other presets (mean |dp| at 8k+ tokens: I8CX .0045, PV8
.0039, FAST .0044). Its overflow guard did not fire on any DEV record.

Long requests against PyTorch (bench records, 1 / 8 questions, mean |dp|):

| preset | 32k | 64k |
|---|---|---|
| I8CX+PV8 | .0039 / .0181 | .0108 / .0175 |
| I8CX+FAST | .0060 / .0225 | .0098 / .0209 |

## Latency per preset (tilde-eval, warm, best of 3, ms; one request = state + 1 or 8 questions)

| request | tokens | Q8_0 | Q6_KE | Q4_KE | NVFP4E | MXFP4E | I8CX | I8CX+PV8 | I8CX+FAST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1q-1024 | 1,024 | 57 | 62 | 58 | 67 | 59 | 36 | 36 | 36 |
| 1q-4096 | 4,096 | 194 | 200 | 190 | 215 | 199 | 122 | 121 | 119 |
| 1q-16384 | 16,384 | 845 | 881 | 843 | 930 | 867 | 556 | 541 | 507 |
| 1q-32768 | 32,768 | 1,958 | 2,030 | 1,955 | 2,129 | 2,004 | 1,372 | 1,311 | 1,180 |
| 8q-1024 | 1,024 | 58 | 62 | 58 | 67 | 59 | 36 | 36 | 36 |
| 8q-4096 | 4,096 | 194 | 200 | 189 | 215 | 199 | 122 | 121 | 119 |
| 8q-16384 | 16,384 | 842 | 879 | 841 | 928 | 865 | 555 | 539 | 506 |
| 8q-32768 | 32,768 | 1,953 | 2,027 | 1,951 | 2,124 | 2,000 | 1,369 | 1,307 | 1,176 |
| 8q-65536 | 64,106 | 4,862 | 5,009 | 4,859 | 5,198 | 4,957 | 3,689 | 3,450 | 2,962 |
| 1q-65536 | 65,536 | 5,028 | 5,175 | 5,021 | 5,365 | 5,122 | 3,827 | 3,565 | 3,064 |

## Benchmarks: real HTTP requests through `tilde-server` (warm median of 3, cold first request in parentheses), ms

Requests: a public long document cut to exact token counts (`tilde/bench/make_bench.py`), 1 or 8 questions. Tier rows run
on the 3090 Ti with a VRAM ballast leaving 15.5 / 11.5 / 7.5 GiB free (16 / 12 / 8 GB cards minus a desktop), so their
speed is "3090 Ti, VRAM-capped". 1q-65536 is not a valid request (32,768 per state + question).

Build 507f5368d's kernels (2026-10-08 06:40):

| run | 1q-1024 | 8q-1024 | 1q-4096 | 8q-4096 | 1q-16384 | 8q-16384 | 1q-32768 | 8q-32768 | 8q-65536 (64,106 tok) | peak GPU MiB | peak RSS MiB | ready s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I8CX 3090Ti | 41 (62) | 42 (68) | 131 (152) | 131 (141) | 585 (605) | 584 (598) | 1429 (1451) | 1426 (1441) | 3805 (3827) | 13855 | 4158 | 5.2 |
| I8CX+PV8 3090Ti | 40 (52) | 42 (74) | 130 (152) | 132 (141) | 568 (583) | 570 (575) | 1368 (1386) | 1364 (1382) | 3564 (3583) | 13981 | 4158 | 5.1 |
| **I8CX+FAST 3090Ti** | 40 (60) | 42 (65) | 130 (148) | 130 (150) | 540 (564) | 538 (550) | **1244 (1258)** | 1240 (1264) | **3084 (3110)** | 13845 | 4158 | 6.1 |
| Q8_0 3090Ti | 61 (85) | 62 (78) | 203 (219) | 205 (224) | 868 (876) | 864 (878) | 1994 (2015) | 1994 (2007) | 4938 (4958) | 13733 | 4099 | 7.1 |
| Q4_KE 3090Ti | 62 (78) | 64 (89) | 198 (217) | 201 (207) | 863 (884) | 865 (878) | 2001 (2017) | 1995 (2008) | 4931 (4961) | 10421 | 4165 | 8.2 |
| NVFP4E 3090Ti | 70 (95) | 73 (97) | 225 (240) | 223 (232) | 959 (967) | 952 (958) | 2172 (2188) | 2167 (2180) | 5268 (5292) | 10423 | 4105 | 8.1 |
| I8CX+FAST, 16 GB tier | 40 (60) | 43 (57) | 128 (147) | 129 (147) | 541 (557) | 540 (555) | 1249 (1266) | 1244 (1254) | 3097 (3118) | 13843 | 4156 | 12.1 |
| Q4_KE, 12 GB tier | 61 (76) | 62 (83) | 197 (218) | 198 (211) | 866 (878) | 865 (871) | 1997 (2014) | 1992 (2005) | 4925 (4952) | 10419 | 4130 | 6.1 |
| Q4_KE, 8 GB tier (limit 19,456) | 62 (75) | 62 (82) | 198 (225) | 201 (212) | 867 (888) | 864 (872) | rejected | rejected | rejected | 6919 | 1975 | 3.1 |

Earlier build (2026-10-08 01:00, before the speed work), unchanged references:

| run | 1q-1024 | 8q-1024 | 1q-4096 | 8q-4096 | 1q-16384 | 8q-16384 | 1q-32768 | 8q-32768 | 8q-65536 | peak GPU MiB | peak RSS MiB | ready s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch bf16 (tilda/serve.py) 3090Ti | 71 (186) | 71 (75) | 224 (273) | 225 (226) | 930 (5014) | 934 (934) | 2286 (6423) | 2296 (2302) | 6297 (6304) | 24107 | 3284 | 124.2 |
| BF16 3090Ti (old kernels) | 189 (486) | 190 (213) | 631 (793) | 606 (613) | 2424 (2650) | 2466 (2398) | 5177 (5551) | 5200 (5205) | 11255 (10924) | 20219 | 14979 | 23.3 |
| Q8_0 Vulkan 3090Ti | 119 (3390) | 114 (1530) | 378 (1921) | 368 (378) | 1566 (1950) | 1530 (1578) | 3512 (3912) | 3470 (3531) | 10348 (10470) | 13904 | 4084 | 20.0 |
| Q8_0 CPU 12700K (8 threads) | 4260 (4122) | 3750 (3798) | 19871 (19824) | 19162 (19831) | 117069 (118404) | - | - | - | - | 0 | 12369 | 10.5 |

Peak GPU is the whole device minus what was in use before the server started (includes the CUDA context).
Conformance (`tilde/bench/check_server.py`, I8CX, full limits): response shape OK, all rejection cases return 400 with
serve.py's messages, 20/21 expectations met (`schedule/free_11` p = 0.91, want false; Q8_0 missed it too).

## Memory (CUDA, measured)

Model buffer on the GPU: Q8_0 8,389 MiB, I8CX 7,977 MiB, Q4_KE / NVFP4E 5,077 MiB (token embeddings stay in host
memory). Compute buffer reserved at startup for a 64k request limit: 4,564 MiB (the int8 attention keeps the f32 K/V
sources alive instead of f16 copies, +128 MiB). `tilde-server` plans with weights + 640 MiB + 0.0715 MiB per token and
lowers the request limit when memory is short.

| card | preset | request limit | peak GPU (whole device) |
|---|---|---:|---:|
| 24 GB | I8CX (+FAST) | 65,536 | 13.5 GiB |
| 16 GB | I8CX+FAST | 65,536 | 13.5 GiB |
| 12 GB | Q4_KE | 65,536 | 10.2 GiB |
| 12 GB | Q8_0 (earlier build) | 30,720 | 10.7 GiB |
| 8 GB | Q4_KE | 19,456 | 6.8 GiB |

## Where the 32k time goes (I8CX+FAST, one 32,768-token request, CUPTI kernel times)

Kernel time 1,107 ms (wall 1,180 ms; the rest is launch gaps and host work):

| part | ms | notes |
|---|---:|---|
| attention, 6 MLA layers (`sage_attn_192_128`) | 344 | 57 ms per layer; QK^T int8 + PV f16 floor ~35 ms per layer |
| dense int8 GEMMs (CUTLASS, scale in the epilogue) | 153 | 426 calls |
| expert int8 grouped GEMMs (CUTLASS) | 144 | gate_up + down, 4 chunks x 23 layers |
| chunked gated delta net (18 KDA layers) | 126 | inter 67 (sequential over chunks) + intra 59 |
| KDA conv (concat + conv + SiLU fused) | 38 | |
| MoE epilogues: reduce (+ shared expert + residual) 36, SwiGLU -> int8 17, gather 19, ids 15 | 87 | |
| norms (fused add+norm 22, l2 21, others 25) | 68 | |
| attention gate sigmoid, KDA gate, casts, concat | 63 | |
| head GEMMs (F16 operands) | 24 | |
| quantization left (dense 13, MMQ 7, attention 7) | 27 | |
| routing (fused) | 2 | |

History at 32k (tilde-eval, warm): 2,092 (Q8_0, GDN v4) -> 1,911 (I8C) -> 1,642 (int8 experts, fused epilogues) ->
1,530 (v12, int8 P*V) -> 1,384 (cast elision, MoE reduce + residual, warp norms, fused routing) -> 1,311 (head F16, GDN
f16 temporaries, int8 rows from SwiGLU and norms) -> 1,180 (I8CX+FAST).

## RTX 5090 (Vast, Ryzen 9 9950X, CUDA 12.8, arch 120a; run 2026-10-08 08:34 CDT, commit ab5d070a2)

`tilde/bench/run_gpu_box.sh` on a fresh box; all kernel tests pass. Parity on the 400-record public-source DEV slice
(the full DEV set stays on hermeslab), latency with tilde-eval (warm, best of 3, ms):

| preset | 1k | 4k | 16k | 32k | 64k (1q) | slice accuracy | log loss | mean \|dp\| | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 26 | 84 | 408 | 939 | 2,372 | -0.27 pt | +0.0039 | .0066 | 14.0 GB |
| Q4_KE | 25 | 82 | 411 | 941 | 2,371 | -0.07 pt | -0.0001 | .0148 | 10.7 GB |
| I8CX | 18 | 52 | 267 | 654 | 1,781 | -0.07 pt | -0.0065 | .0100 | 14.5 GB |
| I8CX+PV8 | 18 | 51 | 258 | 621 | 1,652 | +0.20 pt | -0.0106 | .0128 | 14.6 GB |
| **I8CX+FAST (default)** | 18 | 50 | 246 | **575** | **1,473** | -0.07 pt | -0.0004 | .0099 | 14.5 GB |
| NVFP4E (native W4A4) | 22 | 79 | 386 | 892 | 2,271 | **-0.74 pt** | **+0.0139** | .0227 | 10.7 GB |

`tilde-server` with no flags (I8CX file): picks I8CX + fast attention; warm 32k 640 ms (1 question) / 623 ms (8),
64k 1,521 ms, 1k 23 ms; peak GPU 14.1 GB. Long requests vs PyTorch (FAST): mean |dp| .0056 / .0101 at 32k (1 / 8
questions), .0006 / .0204 at 64k. NVFP4E with Blackwell's native W4A4 is over the budget on this slice (log loss) and
slower than I8CX: not recommended.

The first run aborted at the 64k sweep: the Tilde graph was sized at `n_tokens * 40` nodes, a 50 GB host allocation for
the scheduler at 64k that only worked where memory is overcommitted. Fixed (b89d5a191): `n_tokens + 16,384` (the graph
has 12,383 nodes at 65,536 tokens).

## Open problems / what is left

- 1,000 ms at 32k on the 3090 Ti: the remaining big parts are attention (344 ms, ~1.6x its tensor floor), the int8
  GEMMs (297 ms, ~65% of the int8 peak at these shapes) and the delta net (126 ms, the sequential inter-chunk kernel
  uses 64 of 84 SMs). Ideas, none started: V cast elision for FAST (~8 ms), attention with fewer registers for 3 blocks
  per SM, a persistent delta-net inter kernel, fusing the MoE gather into the grouped GEMM's A loads (CUTLASS
  gather), quantizing the KDA output in its gate kernel.
- Metal and ROCm are untested (no hardware); Vulkan passes parity (older build).
- `schedule/free_11` in the conformance set is missed by I8CX and Q8_0 (p 0.91, want false); not yet checked against
  PyTorch.

# Tilde Omni (tilde-4b-omni)

Last updated 2026-10-08 23:59 CDT. Model: TLive-Omni-4B (TaoLiveAIGC/TLive-Omni-4B @ cfbac342, Qwen3.5-4B text decoder
+ Qwen3.5 ViT + AuT audio tower) trained from base with the Tilde joint head and the unified point head
(`omni_full_tlive4b_20261007/final_folded`). Measured on hermeslab (RTX 3090 Ti, i7-12700K) unless marked.

## Phase

1. Text path: **done** (arch `tilde_omni`, converter, CPU + CUDA parity, request-encoder parity; independent review
   clean after 10 findings fixed: lexical-table and none-vector checks in the converter, a compact media input path
   instead of 3072-wide inputs for every token, output-capacity rejection, smaller no-FA limit, gguf-py names).
2. Vision + pointing: **parity done** (mmproj converter, `tilde_omni_v` clip graph, point head in the LM graph,
   `/v1/systemone` images + `point` questions, tilde-eval `--mmproj`); every subset within the gates (below).
   Independent review: 9 findings (no parity defect; converter guards against projector norms and other ViT
   interpolation, request media limits checked from the image header before decoding, tilde-eval size checks,
   layout range checks, Python-exact rounding of point diagnostics), fixed; re-review clean.
3. Audio: **parity done** (`tilde_omni_a`: AuT + native bridge, Whisper-exact frames, valid-token windows).
4. Speed: **done for this round**: int8 attention for the 256-dim GQA heads (`fattn-sage256.cu`, `TILDE_OMNI_FA=fast`)
   + the I8C int8 GEMMs; 32k text 6.95 s (BF16) -> 3.03 s (I8C+fast), every parity gate passed (below).
5. Service: tilde-server picks the omni family (I8C + fast attention, mmproj beside it, alias tilde-4b-omni), memory
   constants measured; unit file `tilde/deploy/tilde-omni-server.service` (port 18797) written, **not installed**:
   it does not fit beside tilde-server at a 65,536-token limit (options below).

## Temperature

The HF `final_folded` of 2026-10-07 holds the quick-fit T = 1.28 (measured: `final`'s and `final_folded`'s
joint_logit_scale differ by exactly log 1.28; the fold also changed prior_logit_scale and residual_scorer.3). The
full-DEV fit is T = 1.40 (FINAL_EVALS.json), and the omni box's local `final_folded` was re-folded at 1.40 (to be
re-uploaded). The GGUF serves at T = 1.40: the converter measures the folded temperature from the heads
(`--unfolded-head final/joint_head.safetensors`; report.json is not trusted, a re-fold may leave it stale) and folds
only the rest, 1.40 / 1.28 = 1.09375 here (`tilde.temperature_folded` 1.40, `tilde.checkpoint_temperature` 1.28), so a
re-fetched 1.40 checkpoint is not folded twice. The point head is not temperature-scaled (as trained).

## Files

`tilde/convert_tilde_omni_to_gguf.py` (LM: text decoder, joint head, point head, markers + 7 media ids;
`tilde-4b-omni-BF16.gguf` 10.1 GB, `Q8_0` 6.8 GB via `llama-quantize --token-embedding-type bf16
--output-tensor-type bf16 --tensor-type '^(decision\.|dec\.|token_types|point\.)=f32'`),
`tilde/convert_tilde_omni_mmproj.py` (ViT + merger + bridge + sub-patch keys, AuT + audio bridge; F16 2.1 GB).

## Parity: text (5,235 DEV records / 7,365 fields; the Ling parity set's states re-encoded with the omni tokenizer)

Reference: `tilde/parity/reference_omni.py run` (tilda.qwen35 serving, profile ampere24_bf16 = exact bf16, autocast
bf16, logits / 1.09375). Records: `reference_omni.py encode` from `frozen/tilda41_20261006/dev_orig40_clean.jsonl`,
ids of `tilde_parity_20261008/gpu_sub.jsonl` (all 5,235 found, same field order).

| run | fields | accuracy (ref -> test) | delta (pt) | log loss delta | mean abs dp | p99 abs dp | max abs dp | argmax agreement | strict gates |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| CUDA BF16 | 7,365 | 0.8831 -> 0.8826 | -0.05 | +0.0002 | 0.0010 | 0.011 | 0.062 | 99.88% | pass |
| CUDA Q8_0 | 7,365 | 0.8831 -> 0.8823 | -0.08 | +0.0001 | 0.0022 | 0.026 | 0.147 | 99.73% | pass |

CPU (i7-12700K, 8 threads, 214 records <= 2,048 tokens / 277 fields): Q8_0 0.8700 -> 0.8628 (2 fields flip), mean
|dp| 0.0021, argmax 99.28%; BF16 0.8700 -> 0.8664 (1 field flips), mean |dp| 0.0011, argmax 99.64%.

Request encoder (`tilde/parity/encoder_parity.py` with the omni GGUF and tokenizer, 3,033 bodies from the DEV states +
edge cases): 3,016 / 3,016 accepted bodies token- and order-exact against `tilda.joint_serving.encode_request`, 17 both
rejected (15 with the same message).

Total model time over the 5,235 records (batch 1, one request per call, 2.63M tokens): PyTorch bf16 433.5 s,
CUDA BF16 481.1 s, CUDA Q8_0 371.3 s (not yet optimized; long requests are slower than PyTorch, e.g. 27k tokens
5.6 s BF16 vs 4.0 s PyTorch; Phase 4).

GPU window for these runs: tilde-server down 20:50:01-21:12:53 CDT (1,372 s).

## Parity: vision, pointing, ScreenSpot-v2, audio (the checkpoint-eval subsets of OMNI_FULL_RUN.md)

Subsets rebuilt exactly as the run's `EVAL_SUBSETS` (vision 1,248 rows, pointing DEV 722 rows, ScreenSpot-v2 600,
audio 849). Reference: `tilde/parity/reference_omni_media.py run` (tilda.qwen35 `build_t4` with the online
TLive encoders, serve profile ampere24_bf16, Clef logits / 1.09375); the same records through
`tilde-eval --mmproj` (CUDA, BF16 LM + F16 mmproj, flash attention). Point fields: hit = click inside the target box,
or `none` when the target is off screen (same rule both sides), same click = identical token and sub-cell location.

| subset | records | fields | accuracy (ref -> GGUF) | delta (pt) | log loss delta | mean abs dp | p99 abs dp | argmax agreement |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| vision | 781 | 1,248 | 0.8397 -> 0.8397 | 0.00 | +0.0005 | 0.0046 | 0.044 | 99.20% |
| pointing DEV (choice fields) | 267 | 396 | 0.9141 -> 0.9192 | +0.51 | -0.0030 | 0.0035 | 0.047 | 99.49% |
| audio | 351 | 849 | 0.7644 -> 0.7609 | -0.35 | +0.0003 | 0.0034 | 0.036 | 99.41% |

| point subset | point fields | hit (ref -> GGUF) | delta (pt) | same click | mean abs dp(none) | mean max abs dp(token) |
|---|---:|---|---:|---:|---:|---:|
| pointing DEV | 326 (272 on screen) | 0.8558 -> 0.8589 | +0.31 | 96.3% | 0.0022 | 0.016 |
| ScreenSpot-v2 | 600 | 0.8717 -> 0.8717 | 0.00 | 96.0% | 0.0023 | 0.019 |

Wall time per subset including model load (PyTorch / tilde-eval BF16): vision 187 / 132 s, pointing 113 / 81 s,
ScreenSpot 167 / 140 s, audio 170 / 51 s. GPU window: tilde-server down 21:55:02-22:12:42 CDT (1,060 s).

## Speed (phase 4; 3090 Ti, tilde-server stopped, tilde-eval batch 1, last of 3 repeats, ms)

| preset | 1k | 4k | 16k | 32k |
|---|---:|---:|---:|---:|
| BF16 | 147 | 576 | 2,777 | 6,952 |
| Q8_0 | 105 | 435 | 2,286 | 5,998 |
| Q8_0 + fast attention | 107 | 402 | 1,767 | 4,093 |
| I8C | 75 | 320 | 1,809 | 5,025 |
| I8C + int8 attention (f32 P*V accumulation) | 73 | 288 | 1,353 | 3,243 |
| I8C + fast attention (f16 accumulation) | 73 | 283 | 1,283 | 3,025 |

PyTorch reference for comparison (earlier, not this window): 32k ~2.8 s bf16 eager; ~1.3 s at 18.7k with the int8
stack. Text parity (5,235 records / 7,365 fields vs the PyTorch reference): I8C +0.03 pt / +0.0008 log loss (mean |dp|
0.0108); I8C+fast 0.00 pt / +0.0014 (0.0107); Q8_0+fast -0.08 pt / +0.0001 (0.0024: the attention adds little, the
I8C GEMMs most of the difference). Media, I8C+fast vs reference: vision -0.64 pt / +0.0039, pointing DEV choice +1.26 pt,
pointing hit +0.92 pt (same click 91.4%), ScreenSpot hit -0.50 pt (same click 90.5%), audio +0.35 pt / +0.0017. All
within the gates (<= 1.0 pt loss, <= +0.01 log loss).

HTTP (llama-server, I8C+fast, warm median): 1.1k tokens 92 ms, 3.8k 276 ms, 16.5k 1,341 ms; image (1 screenshot, point +
noul, 385 tokens) 117 ms (cold 190); audio clip + choice (187 tokens) 45 ms.

GPU window 3: tilde-server down 23:26:32-23:51:10 CDT (1,478 s) (an earlier attempt at 22:35 aborted after 13 s). Review (sonnet): clean after 2 fixes (padded keys masked to -inf, not a raw-unit -5e6; tilde-server matches family/preset on the file name); the mask fix only touches requests whose length is not a multiple of 64 and was not re-measured on the GPU.

## Service (phase 5)

Device memory (llama-server I8C + F16 mmproj, peak during requests): 11,751 MiB at a 32,768-token limit, 14,759 MiB at
65,536. tilde-server (8b) uses 12,850 MiB of the 24,114. Both at once: 12,850 + 14,759 > 24 GB, so no at 65,536.
Options (not decided): (a) omni with `-c 20480` (about 10.6 GB, estimate from the measured slope) beside tilde-server;
(b) swap: one server at a time; (c) a smaller tilde-server limit; (d) a second GPU.

## Open problems

- Media order: images then audio; time-ordered interleaving to add after phase 5 and measure on omni_av2_both.
- webp is refused (decode it or answer 415).

