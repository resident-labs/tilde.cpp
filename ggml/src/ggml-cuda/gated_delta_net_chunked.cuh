#pragma once

#include "common.cuh"

// Chunked (prefill) form of GGML_OP_GATED_DELTA_NET for long batches: the per-token recurrence is replaced by
// chunks of 32 tokens (the WY form of the delta rule, as in flash-linear-attention's chunk_kda / chunk_gated_delta_rule).
// Returns false if the shape is not supported (the caller then runs the per-token kernel).
bool ggml_cuda_gated_delta_net_chunked(
        ggml_backend_cuda_context & ctx,
        const float * q, const float * k, const float * v, const float * g, const float * beta, const float * s0,
        float * dst, float * state_out,
        bool kda, int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, float scale, cudaStream_t stream);
