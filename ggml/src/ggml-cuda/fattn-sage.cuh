#pragma once

// SageAttention2-style int8 Q*K^T / f16 P*V (f16 accumulation) for 192/128 heads without mask (fattn-sage.cu).
// Plain-type interface: the kernels are compiled apart from the rest of the CUDA backend.
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

size_t  ggml_sage_smem();
int64_t ggml_sage_q_scales(int64_t n_q);   // per head
int64_t ggml_sage_k_scales(int64_t n_kv);  // per head

// rows (token stride nb1, head stride nb2, bytes) -> int8 [H][n][192] and block scales. ksum == nullptr: Q (blocks of
// 32 rows); otherwise K minus ksum / n (blocks of 64 rows). round_f16: an f32 source that stands for an f16 copy.
void ggml_sage_quantize(const void * src, bool src_f32, bool round_f16, int64_t nb1, int64_t nb2, int64_t n, int64_t H,
        const float * ksum, int8_t * dst, float * scales, cudaStream_t stream);

// dst[token * o_tok + head * o_head + 0..127] (f32) = softmax(sm_scale Q K^T) V; v: f16 rows (strides in elements).
// flag: device int, set when the f16 accumulation overflowed (the layer is then recomputed with f32 accumulation)
void ggml_sage_attn(const int8_t * q8, const int8_t * k8, const void * v, int64_t v_seq, int64_t v_head, const float * qs,
        const float * ks, float * dst, int64_t o_tok, int64_t o_head, int64_t n_q, int64_t n_kv, int64_t H, float sm_scale,
        int * flag, cudaStream_t stream);
