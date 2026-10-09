#pragma once

// SageAttention2-style int8 Q*K^T / f16 P*V for 256-dim heads with grouped K/V heads and no mask (fattn-sage256.cu;
// the Qwen3.5 full-attention layers of Tilde Omni). Plain-type interface: compiled apart from the rest of the backend.
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

size_t  ggml_sage256_smem();
int64_t ggml_sage256_q_scales(int64_t n_q);   // per Q head
int64_t ggml_sage256_k_scales(int64_t n_kv);  // per K head

// softmax(sm_scale Q K^T) V for every Q head h (K/V head h / (H / H_kv)):
//   q: f32 rows, k: f32 (an f16 copy the graph skipped: k_f32) or f16 rows, v: f16 rows; strides in bytes
//   dst[token * o_tok + head * o_head + 0..255] (f32, strides in elements)
// Scratch (device): q8 [H][n_q][256], k8 [H_kv][n_kv][256], qs [H][q_scales], ks [H_kv][k_scales],
// ksum [H_kv][256], kpart [ceil(n_kv / 256)][H_kv][256], flag (int)
// acc32: f32 accumulation of the output across key tiles (f16 within a tile); otherwise f16 over the whole row with an
// overflow guard (the layer reruns with f32 accumulation when any output is not finite)
void ggml_sage256_attn(const void * q, int64_t q_nb1, int64_t q_nb2, const void * k, bool k_f32, int64_t k_nb1,
        int64_t k_nb2, const void * v, int64_t v_nb1, int64_t v_nb2, float * dst, int64_t o_tok, int64_t o_head,
        int64_t n_q, int64_t n_kv, int64_t H, int64_t H_kv, float sm_scale, bool acc32,
        int8_t * q8, int8_t * k8, float * qs, float * ks, float * ksum, float * kpart, int * flag, cudaStream_t stream);

int64_t ggml_sage256_kpart_rows(); // keys per partial K sum
