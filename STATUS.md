# tilde.cpp status

Last updated 2026-10-08 01:05 CDT. Every number below was measured on hermeslab (RTX 3090 Ti 24 GB, i7-12700K,
CUDA 13.1). Nothing is extrapolated.

## What upstream llama.cpp already covers (base tag b11483) and what tilde.cpp adds

Upstream already has:
- the Ling-3.0 backbone (`bailingmoe3`: KDA through `ggml_gated_delta_net`, MLA, MoE);
- a Clef joint decision head (`clef` arch, the same JointSchemaHead that Tilde's "fast" head computes);
- the `/v1/systemone` endpoint in `llama-server` (`tools/server/server-decision.cpp`) with decision-order spans.

tilde.cpp adds (kept small for an upstream PR, speed work separate and generic):
- the `tilde` arch (`src/models/tilde.cpp`): non-causal MLA with K/V decompressed from the latent (192/128), zero-state
  KDA (no memory module), MoE in token chunks, the joint head in the graph sized for 64k tokens;
- the `tilde` decision type in `server-decision.cpp` (the Tilda serialization, noul format, limits, serve.py response
  and error shapes);
- the converter (`tilde/convert_tilde_to_gguf.py`, T folded), quant presets, `tilde-server`, parity and bench tools;
- generic CUDA speed work: int8 flash attention (`fattn-i8.cu`), chunked gated delta net (`gated_delta_net_chunked.cu`),
  192/128 MMA flash attention without GQA, a transposed-concat path, 65,535 grid-limit fixes.

## Phase

Phases 1-5 are done (converter, graph + CPU parity, CUDA/Vulkan, presets with gates, server). Phase 6 (benchmarks) is
measured below. **Current priority (lead, 2026-10-08): one 32k request under 1,000 ms warm on the 3090 Ti.** We are at
2,436 ms (Q8_0, server, warm); PyTorch bf16 is at 2,286 ms on the same requests.

## Parity (GPU, 5,235 DEV records / 7,365 fields, against the bf16 PyTorch reference)

Gates: accuracy at most 0.5 pt below the reference, log loss at most +0.005, mean |dp| at most 0.02, argmax agreement
at least 98%. (Lead, 2026-10-08: a speed preset may cost up to 1.0 pt accuracy and +0.01 log loss; anything over that
goes to Ari.)

| preset | fields | accuracy (ref -> preset) | delta (pt) | log loss delta | mean abs dp | p99 abs dp | max abs dp | argmax agreement | gates |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| BF16, f16 attention | 7,365 | 0.8846 -> 0.8842 | -0.04 | +0.0004 | 0.0043 | 0.043 | 0.289 | 99.59% | pass |
| BF16, int8 attention | 7,365 | 0.8846 -> 0.8843 | -0.03 | +0.0004 | 0.0045 | 0.044 | 0.279 | 99.58% | pass |
| Q8_0, f16 attention | 7,365 | 0.8846 -> 0.8851 | +0.05 | +0.0004 | 0.0055 | 0.056 | 0.333 | 99.39% | pass |
| **Q8_0 (default), int8 attention** | 7,365 | 0.8846 -> 0.8849 | +0.03 | -0.0001 | 0.0056 | 0.056 | 0.292 | 99.42% | pass |
| Q6_KE, int8 attention | 7,365 | 0.8846 -> 0.8843 | -0.03 | +0.0006 | 0.0062 | 0.066 | 0.263 | 99.42% | pass |
| Q4_KE, int8 attention | 7,365 | 0.8846 -> 0.8836 | -0.10 | +0.0010 | 0.0130 | 0.142 | 0.564 | 98.60% | pass |
| MXFP4E, int8 attention | 7,365 | 0.8846 -> 0.8821 | -0.24 | +0.0058 | 0.0212 | 0.205 | 0.663 | 97.98% | FAIL (log loss, mean dp, argmax) |
| NVFP4E, int8 attention | 7,365 | 0.8846 -> 0.8819 | -0.27 | +0.0052 | 0.0165 | 0.171 | 0.520 | 98.25% | FAIL (log loss by 0.0002) |
| Vulkan Q8_0 (318 records) | 405 | 0.8741 -> 0.8790 | +0.49 | +0.0001 | 0.0054 | 0.039 | 0.245 | 99.26% | pass |

CPU BF16 (i7-12700K, 318 records): .8741 -> .8765, log loss +0.0002, argmax 99.5%.
MXFP4E and NVFP4E are measured with W4A8 MMQ on Ampere; Blackwell's native path is W4A4 and needs the 5090 run.
Request encoder vs `tilda.joint_serving.encode_request`: token- and span-exact on 3,015 requests.

Long requests (bench records, Q8_0, int8 vs f16 attention, max |dp| per request): 0.0005-0.06; mean 0.02-0.026 at
32k-64k with 8 questions. The comparison against PyTorch on these records is running.

## Benchmarks: real HTTP requests through the server (warm median of 3, cold first request in parentheses), ms

Requests: a public long document cut to exact token counts (`tilde/bench/make_bench.py`), 1 or 8 questions. "Tier" rows
run on the 3090 Ti with a VRAM ballast leaving 15.5 / 11.5 / 7.5 GiB free (16 / 12 / 8 GB cards minus a desktop), so
their speed is "3090 Ti, VRAM-capped". 1q-65536 is not a valid request (32,768 per state + question).

| run | 1q-1024 | 8q-1024 | 1q-4096 | 8q-4096 | 1q-16384 | 8q-16384 | 1q-32768 | 8q-32768 | 8q-65536 (64,106 tok) | peak GPU MiB | peak RSS MiB | ready s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 3090Ti | 77 (97) | 80 (97) | 257 (261) | 255 (273) | 1089 (1102) | 1089 (1100) | 2436 (2447) | 2439 (2459) | 5801 (5818) | 13627 | 4211 | 5.1 |
| Q4_KE 3090Ti | 78 (93) | 80 (99) | 249 (257) | 253 (257) | 1093 (1103) | 1083 (1090) | 2435 (2441) | 2423 (2440) | 5774 (5789) | 10315 | 4218 | 5.1 |
| BF16 3090Ti | 189 (486) | 190 (213) | 631 (793) | 606 (613) | 2424 (2650) | 2466 (2398) | 5177 (5551) | 5200 (5205) | 11255 (10924) | 20219 | 14979 | 23.3 |
| NVFP4E 3090Ti | 86 (103) | 87 (105) | 270 (279) | 271 (278) | 1150 (1164) | 1150 (1156) | 2556 (2569) | 2565 (2566) | 6037 (6042) | 10315 | 4223 | 7.2 |
| Q8_0, 16 GB tier | 77 (279) | 78 (106) | 257 (351) | 258 (260) | 1091 (1171) | 1090 (1092) | 2440 (2498) | 2434 (2438) | 5782 (5809) | 13625 | 3944 | 13.2 |
| Q4_KE, 12 GB tier | 78 (93) | 80 (101) | 250 (258) | 254 (255) | 1080 (1096) | 1078 (1093) | 2420 (2442) | 2420 (2421) | 5770 (5786) | 10313 | 4132 | 8.1 |
| Q8_0, 12 GB tier (limit 30,720) | 77 (91) | 79 (95) | 254 (263) | 257 (270) | 1087 (1109) | 1084 (1094) | rejected | rejected | rejected | 10967 | 2502 | 6.1 |
| Q4_KE, 8 GB tier (limit 19,456) | 77 (96) | 79 (99) | 252 (265) | 249 (260) | 1079 (1090) | 1076 (1088) | rejected | rejected | rejected | 6897 | 1983 | 3.1 |
| PyTorch bf16 (tilda/serve.py) 3090Ti | 71 (186) | 71 (75) | 224 (273) | 225 (226) | 930 (5014) | 934 (934) | 2286 (6423) | 2296 (2302) | 6297 (6304) | 24107 | 3284 | 124.2 |
| Q8_0 Vulkan 3090Ti | 119 (3390) | 114 (1530) | 378 (1921) | 368 (378) | 1566 (1950) | 1530 (1578) | 3512 (3912) | 3470 (3531) | 10348 (10470) | 13904 | 4084 | 20.0 |
| Q8_0 CPU 12700K (8 threads) | 4260 (4122) | 3750 (3798) | 19871 (19824) | 19162 (19831) | 117069 (118404) | - | - | - | - | 0 | 12369 | 10.5 |

Peak GPU is the whole device minus what was in use before the server started (includes the CUDA context). PyTorch's peak
is its caching allocator's. The CPU row ran 1 warm repeat.

Conformance (`tilde/bench/check_server.py`, Q8_0, full limits): response shape OK, all 10 rejection cases return 400 with
serve.py's messages, 20/21 expectations of `release/check_requests.json` met (`schedule/free_11` p = 0.91, want false;
not yet checked against PyTorch).

## Memory (CUDA, measured)

Model buffer on the GPU: Q8_0 8,389 MiB, Q4_KE 5,077 MiB (token embeddings stay in host memory). Compute buffer
reserved at startup: 477 / 1,123 / 2,227 / 4,436 MiB for a 4k / 16k / 32k / 64k request limit (independent of the MoE
chunk size). Peak device use = weights + compute + ~440-700 MiB (CUDA context, int8-attention and delta-net temporaries).
`tilde-server` plans with weights + 640 MiB + 0.0715 MiB per token and lowers the request limit when the memory is short.

Consumer tiers (from the table): 16 GB -> Q8_0 at full limits (13.3 GiB peak); 12 GB -> Q4_KE at full limits
(10.1 GiB), or Q8_0 up to 30,720 tokens; 8 GB -> Q4_KE up to 19,456 tokens (6.7 GiB).

## Where the 32k time goes (Q8_0, one 32,768-token request, CUPTI kernel times, build of 2026-10-08 00:20)

GPU kernel time 2,318 ms per request (wall 2,401 ms):

| part | ms | share | notes |
|---|---:|---:|---|
| MMQ int8 matmuls (dense + MoE experts, `mul_mat_q<Q8_0>`) | 707 | 30% | int8 MMQ is used for every projection; ~1.8 GFLOP/token -> ~83 TOPS, 29% of the 3090 Ti's dense int8 peak |
| int8 flash attention, 6 MLA layers (`fattn_i8_main`) | 525 | 23% | the int8 path is taken; 6.6e13 FLOP -> ~126 TOPS |
| chunked gated delta net, 18 KDA layers | 274 | 12% | PyTorch/fla: 108 ms |
| activation quantization for MMQ (`quantize_mmq_q8_1`) | 115 | 5% | one pass per matmul input |
| MoE routing: ids helper 58, get_rows 65, argsort 27, weighted sum 46 | 196 | 8% | get_rows of 1-float rows launches one block per row |
| norms (RMS, l2) | 115 | 5% | per-head norms run 256 threads on 128-element rows |
| adds (residuals, biases, MoE+shared) | 63 | 3% | |
| SwiGLU 48, SiLU 32, sigmoid 27, mul 15 | 122 | 5% | SiLU now fused into the conv |
| KDA conv: ssm_conv 43, transposed concat 34 | 77 | 3% | |
| f16 casts of K/V for attention | 25 | 1% | |
| head projection over all tokens (f32 GEMM) | 35 | 2% | |
| other | ~65 | 3% | |

PyTorch at 33,472 tokens for comparison (torch.profiler): flash attention bf16 1,024 ms, MoE grouped GEMM 419 ms,
dense bf16 GEMM ~400 ms, fla KDA kernels 108 ms, everything else ~370 ms.

## In progress (speed)

- GDN v4 (built, tests pass, numbers pending): the matrix products of both chunk kernels on TF32 tensor cores, per-pair
  gate factors inside 16-token sub-chunks and a split factor across them (safe for any gate size), a one-warp forward
  substitution, token-major temporaries.
- Merged gate/up experts (`ffn_gate_up_exps`, one expert matmul instead of two; converter writes it, `tilde/merge_gate_up.py`
  converts existing files byte for byte): GGUFs made, loader done, numbers pending.
- get_rows fast path for tiny rows, 128-thread RMS norm for 128-element rows: done, numbers pending.
- Next, by expected gain: KDA projection fusion (one matmul for q/k/v/gate/beta), fewer elementwise passes, int8
  attention kernel tuning (ldmatrix for K, fewer registers), MMQ tile tuning for ~2k tokens per expert.

## Open problems

- The 1,000 ms target at 32k: with the current kernels the floor of the three big parts alone is about
  attention ~300 ms (QK^T at the int8 peak + PV at the f16-accumulate peak) + matmuls ~210 ms at the int8 peak (~350 ms
  at a realistic 60%) + delta net ~100 ms + the rest; it needs every part near peak.
- Metal and ROCm are untested (no hardware); the Vulkan path passes parity.
- MXFP4E/NVFP4E fail the gates with W4A8; Blackwell W4A4 is untested.

## Next step

Measure GDN v4 + merged experts + the small-kernel fixes (1k-64k timings, DEV parity), commit and push, then work down
the profile table from the top.
