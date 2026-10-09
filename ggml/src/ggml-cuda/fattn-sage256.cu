// SageAttention2-style attention for 256-dim heads with grouped K/V heads and no mask (Tilde Omni / Qwen3.5 full
// attention: 16 Q heads, 4 K/V heads). int8 Q*K^T with one scale per 16 queries and per 64 (mean-smoothed) keys, f16
// P*V on f16 tensor cores. Adapted from SageAttention's sm80 kernel (vendor/sageattn, Apache-2.0) the same way as
// fattn-sage.cu (192/128 MLA heads): 4 warps of 16 query rows per block, the whole 256-dim Q fragment of a warp in
// registers (staged once through the V buffer), K (64 x 256 int8) and V (64 x 256 f16) tiles in shared memory, 48 KB
// per block (two blocks per SM on sm_86), the output written as f32 straight from the registers.
//
// Accumulation of P*V: f16 over the whole row (mode 0, the fast one; every output is checked and a flag is raised on
// any non-finite value, and the same kernel with f32 accumulation then reruns the layer: it exits at once otherwise),
// or f32 (mode 2).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>
#include <cuda/pipeline>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <cstdlib>
#include <cstdio>

#include "fattn-sage256.cuh"

namespace ggml_sage256 {
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#include "vendor/sageattn/math.cuh"
#include "vendor/sageattn/attn_utils.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define GGML_SAGE256_AVAILABLE
#endif

constexpr uint32_t WARP_Q = 16;
constexpr uint32_t num_warps_q = 4;
constexpr uint32_t CTA_Q  = WARP_Q * num_warps_q; // 64
constexpr uint32_t CTA_K  = 64;
constexpr uint32_t WARP_K = 64;
constexpr uint32_t D      = 256;
constexpr uint32_t num_warps_k = CTA_K / WARP_K; // 1
constexpr uint32_t num_warps   = num_warps_q * num_warps_k;
constexpr uint32_t num_tiles_q = WARP_Q / 16;    // 1
constexpr uint32_t num_tiles_k = WARP_K / 16;    // 4
constexpr uint32_t num_tiles_v = D / 16;         // 16
constexpr uint32_t QK_STEPS    = D / 32;         // int8 k-steps of 32
constexpr uint32_t SMEM_K = 0;
constexpr uint32_t SMEM_V = SMEM_K + CTA_K * D;  // int8 K
constexpr uint32_t SMEM_Q = SMEM_V;              // Q staging, before the first V tile
constexpr uint32_t SMEM   = SMEM_V + CTA_K * D * 2;
static_assert(CTA_Q * D <= CTA_K * D * 2, "the Q staging fits in the V buffer");

// lanes per row and rows per warp of the 16-byte global -> shared copies
constexpr uint32_t LQ = 8, RQ_ = 4, LV = 8, RV = 4;
constexpr uint32_t KPART_ROWS = 256;

// mode 0: f16 accumulation over the whole row (unguarded: raises flag on overflow); mode 2: f32 accumulation
// (guarded: runs only when flag is set, as the rerun of mode 0; unguarded: the acc32 path)
template <int MODE, bool guarded>
__global__ void __launch_bounds__(num_warps * WARP_SIZE) sage_attn_256(
        const int8_t * __restrict__ Q, const int8_t * __restrict__ K, const half * __restrict__ V, float * __restrict__ O,
        const float * __restrict__ Q_scale, const float * __restrict__ K_scale, const uint32_t qo_len, const uint32_t kv_len,
        const uint32_t gqa, const uint32_t stride_seq_v, const uint32_t stride_h_v,
        const int64_t o_tok, const int64_t o_head, float sm_scale, int * flag) {
#ifdef GGML_SAGE256_AVAILABLE
    using ACC = typename std::conditional<MODE == 0, half, float>::type;
    if constexpr (guarded) {
        if (*(volatile int *) flag == 0) {
            return;
        }
    }
    extern __shared__ __align__(128) int8_t smem[];

    const uint32_t lane_id = get_lane_id();
    const uint32_t warp_id = get_warp_id();
    const uint32_t bx      = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t kv_head = head_id / gqa;
    const uint32_t wq      = get_warp_idx_q<num_warps_q, num_warps_k>();
    const uint32_t wk      = get_warp_idx_k<num_warps_q, num_warps_k>();

    sm_scale *= math::log2e;

    int32_t RS[num_tiles_q][num_tiles_k][8];
    ACC     RO[num_tiles_q][num_tiles_v][8];
    float   m[num_tiles_q][2];
    float   d[num_tiles_q][2];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            if constexpr (std::is_same<ACC, float>::value) {
#pragma unroll
                for (uint32_t k = 0; k < 8; k++) {
                    RO[fq][fv][k] = 0.0f;
                }
            } else {
#pragma unroll
                for (uint32_t k = 0; k < 4; k++) {
                    ((int32_t *) RO[fq][fv])[k] = 0;
                }
            }
        }
#pragma unroll
        for (uint32_t k = 0; k < 2; k++) {
            m[fq][k] = -5000000.0f;
            d[fq][k] = 1.0f;
        }
    }

    const uint32_t q_scale_idx = head_id * (gridDim.x * num_warps_q) + bx * num_warps_q + wq;
    const uint32_t n_kblocks   = (kv_len + CTA_K - 1) / CTA_K;
    const uint32_t k_scale_idx = kv_head * n_kblocks;

    smem_t<SwizzleMode::k128B, D / 16> smem_Q(smem + SMEM_Q);
    smem_t<SwizzleMode::k128B, D / 16> smem_K(smem + SMEM_K);
    smem_t<SwizzleMode::k128B, D / 8>  smem_V(smem + SMEM_V);

    constexpr uint32_t Q_row = D / (LQ * 16), Q_col = CTA_Q / (num_warps * RQ_);
    constexpr uint32_t K_col = CTA_K / (num_warps * RQ_);
    constexpr uint32_t V_row = D / (LV * 8),  V_col = CTA_K / (num_warps * RV);

    int8_t * qp = (int8_t *) Q + (int64_t) head_id * qo_len * D + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / LQ) * D + (lane_id % LQ) * 16;
    int8_t * kp = (int8_t *) K + (int64_t) kv_head * kv_len * D + (CTA_K / num_warps * warp_id + lane_id / LQ) * D + (lane_id % LQ) * 16;
    half   * vp = (half *) V + (int64_t) kv_head * stride_h_v + (int64_t) (CTA_K / num_warps * warp_id + lane_id / LV) * stride_seq_v + (lane_id % LV) * 8;

    uint32_t q_ld = smem_Q.get_permuted_offset(warp_id * RQ_ * Q_col + lane_id / LQ, lane_id % LQ);
    uint32_t k_ld = smem_K.get_permuted_offset(warp_id * RQ_ * K_col + lane_id / LQ, lane_id % LQ);
    uint32_t v_ld = smem_V.get_permuted_offset(warp_id * RV * V_col + lane_id / LV, lane_id % LV);

    uint32_t q_mma = smem_Q.get_permuted_offset(wq * WARP_Q + lane_id % 16, lane_id / 16);
    const uint32_t k_mma = smem_K.get_permuted_offset(wk * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
    uint32_t v_mma = smem_V.get_permuted_offset(wk * WARP_K + lane_id % 16, lane_id / 16);

    const uint32_t row_d   = D; // the loaders take the row stride by reference
    const uint32_t q_row   = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / LQ;
    uint32_t k_row = CTA_K / num_warps * warp_id + lane_id / LQ;
    uint32_t v_row = CTA_K / num_warps * warp_id + lane_id / LV;
    uint32_t k_idx = wk * WARP_K + 2 * (lane_id % 4); // key index of the lane's score fragments (out-of-bound mask)

    load_global_to_share<LQ, RQ_, Q_row, Q_col, SwizzleMode::k128B, D / 16, CTA_Q>(&qp, q_ld, row_d, smem_Q, q_row, qo_len);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();
    // the warp's Q fragments to registers; the staging space is then free for V
    uint32_t RQ[QK_STEPS][num_tiles_q][4];
#pragma unroll
    for (uint32_t it = 0; it < QK_STEPS; it++) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            smem_Q.ldmatrix_m8n8x4(q_mma, RQ[it][fq]);
            q_mma = smem_Q.template advance_offset_by_row<16>(q_mma);
        }
        q_mma = smem_Q.template advance_offset_by_column<2>(q_mma - (num_tiles_q * 16 * (D / 16)), it);
    }
    __syncthreads();

    load_global_to_share<LQ, RQ_, Q_row, K_col, SwizzleMode::k128B, D / 16, CTA_K>(&kp, k_ld, row_d, smem_K, k_row, kv_len);
    cp_async::commit_group();
    load_global_to_share<LV, RV, V_row, V_col, SwizzleMode::k128B, D / 8, CTA_K>(&vp, v_ld, stride_seq_v, smem_V, v_row, kv_len);
    cp_async::commit_group();
    k_row += CTA_K; v_row += CTA_K;

    const float q_scale = Q_scale[q_scale_idx];
    float scale = sm_scale * q_scale * K_scale[k_scale_idx];

    for (uint32_t it = 0; it < n_kblocks; it++) {
        const bool last = it + 1 == n_kblocks;
        cp_async::wait_group<1>(); // K
        __syncthreads();

        {
            uint32_t off = k_mma;
#pragma unroll
            for (uint32_t s = 0; s < QK_STEPS; s++) {
#pragma unroll
                for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                    uint32_t RK[4];
                    smem_K.ldmatrix_m8n8x4(off, RK);
                    off = smem_K.template advance_offset_by_row<16>(off);
#pragma unroll
                    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                        if (s == 0) {
                            mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[s][fq], RK);
                        } else {
                            mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS[fq][fk], RQ[s][fq], RK);
                        }
                    }
                }
                off = smem_K.template advance_offset_by_column<2>(off - (num_tiles_k * 16 * (D / 16)), s);
            }
        }

        float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t k = 0; k < 8; k++) {
                    RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
                }
            }
        }
        if (last) {
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(k_idx, RS_f32, kv_len);
        }
        k_idx += CTA_K;

        update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, MODE == 0, false, false>(RS_f32, RO, m, d, scale);

        uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
        RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);
        accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);

        __syncthreads();
        if (!last) {
            load_global_to_share<LQ, RQ_, Q_row, K_col, SwizzleMode::k128B, D / 16, CTA_K>(&kp, k_ld, row_d, smem_K, k_row, kv_len);
            cp_async::commit_group();
            scale = sm_scale * q_scale * K_scale[k_scale_idx + it + 1];
            cp_async::wait_group<1>(); // V
        } else {
            cp_async::wait_group<0>();
        }
        __syncthreads();

        compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, SwizzleMode::k128B, D / 8, 4>(
            smem_V, RS_f16, RO, d, v_mma);

        __syncthreads();
        if (!last) {
            load_global_to_share<LV, RV, V_row, V_col, SwizzleMode::k128B, D / 8, CTA_K>(&vp, v_ld, stride_seq_v, smem_V, v_row, kv_len);
            cp_async::commit_group();
            k_row += CTA_K; v_row += CTA_K;
        }
    }

    normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kTensorCore>(RO, m, d);

    // O (f32) straight from the fragments: RO[fq][fv] holds rows g, g + 8 and columns 2 tq, 2 tq + 1, + 8
    bool bad = false;
    const uint32_t g = lane_id / 4, tq = lane_id % 4;
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        const uint32_t r0 = bx * CTA_Q + wq * WARP_Q + fq * 16 + g;
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            float v[8];
#pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                if constexpr (std::is_same<ACC, float>::value) {
                    v[k] = RO[fq][fv][k];
                } else {
                    v[k] = __half2float(RO[fq][fv][k]);
                }
            }
            const uint32_t c = fv * 16 + 2 * tq;
#pragma unroll
            for (uint32_t h = 0; h < 2; h++) { // row r0 + 8 h
                const uint32_t r = r0 + 8 * h;
                if (r < qo_len) {
                    float * o = O + (int64_t) r * o_tok + (int64_t) head_id * o_head;
                    const float2 a = make_float2(v[2 * h], v[2 * h + 1]);
                    const float2 b = make_float2(v[4 + 2 * h], v[4 + 2 * h + 1]);
                    *(float2 *) (o + c)     = a;
                    *(float2 *) (o + c + 8) = b;
                    bad |= !isfinite(a.x) || !isfinite(a.y) || !isfinite(b.x) || !isfinite(b.y);
                }
            }
        }
    }
    if constexpr (MODE == 0) {
        if (__any_sync(0xffffffff, bad) && lane_id == 0) {
            atomicExch(flag, 1);
        }
    }
#else
    (void) Q; (void) K; (void) V; (void) O; (void) Q_scale; (void) K_scale; (void) qo_len; (void) kv_len; (void) gqa;
    (void) stride_seq_v; (void) stride_h_v; (void) o_tok; (void) o_head; (void) sm_scale; (void) flag;
#endif // GGML_SAGE256_AVAILABLE
}

// sum of K over the keys per K head and dim: partial sums of KPART_ROWS keys, then a fixed-order reduction
template <typename T>
__global__ void sage256_k_sum_partial(const char * __restrict__ K, float * __restrict__ part, const int64_t n_kv,
        const int64_t nb1, const int64_t nb2) {
    const int h = blockIdx.y;
    const int dd = threadIdx.x;
    const int64_t t0 = (int64_t) blockIdx.x * KPART_ROWS;
    const int64_t t1 = min(t0 + (int64_t) KPART_ROWS, n_kv);
    float s = 0.0f;
    for (int64_t t = t0; t < t1; t++) {
        const T * p = (const T *) (K + h * nb2 + t * nb1);
        if constexpr (std::is_same<T, half>::value) {
            s += __half2float(p[dd]);
        } else {
            s += p[dd];
        }
    }
    part[((int64_t) blockIdx.x * gridDim.y + h) * D + dd] = s;
}

__global__ void sage256_k_sum_reduce(const float * __restrict__ part, float * __restrict__ ksum, const int n_part,
        const int H_kv) {
    const int h = blockIdx.x;
    const int dd = threadIdx.x;
    float s = 0.0f;
    for (int i = 0; i < n_part; i++) {
        s += part[((int64_t) i * H_kv + h) * D + dd];
    }
    ksum[h * D + dd] = s;
}

// rows of R tokens per block share one int8 scale (max |x| / 127 over the block): Q blocks of 16, K blocks of 64 (K
// minus its mean over all keys first). 8 warps, R / 8 rows per warp, 8 values per lane.
template <typename T, int R, bool round_f16>
__global__ void sage256_quantize_rows(const char * __restrict__ src, const int64_t nb1, const int64_t nb2, const int64_t n,
        const float * __restrict__ ksum, const float inv_n, int8_t * __restrict__ dst, float * __restrict__ scales,
        const int64_t n_blocks) {
    constexpr int W = 8;
    constexpr int PER = R / W;
    constexpr int PL = D / WARP_SIZE;
    __shared__ float s_max[W];
    const int h = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t row0 = (int64_t) blockIdx.x * R + warp * PER;
    float x[PER][PL];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int64_t row = row0 + i;
#pragma unroll
        for (int j = 0; j < PL; j++) {
            const int dd = lane + j * WARP_SIZE;
            float v = 0.0f;
            if (row < n) {
                const T * p = (const T *) (src + h * nb2 + row * nb1);
                if constexpr (std::is_same<T, half>::value) {
                    v = __half2float(p[dd]);
                } else {
                    v = p[dd];
                    if constexpr (round_f16) {
                        v = __half2float(__float2half(v));
                    }
                }
                if (ksum != nullptr) {
                    v -= ksum[h * D + dd] * inv_n;
                }
            }
            x[i][j] = v;
            amax = fmaxf(amax, fabsf(v));
        }
    }
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    if (lane == 0) {
        s_max[warp] = amax;
    }
    __syncthreads();
    amax = s_max[0];
#pragma unroll
    for (int w = 1; w < W; w++) {
        amax = fmaxf(amax, s_max[w]);
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int64_t row = row0 + i;
        if (row < n) {
            int8_t * o = dst + ((int64_t) h * n + row) * D;
#pragma unroll
            for (int j = 0; j < PL; j++) {
                o[lane + j * WARP_SIZE] = (int8_t) __float2int_rn(fminf(127.0f, fmaxf(-127.0f, x[i][j] * inv)));
            }
        }
    }
    if (threadIdx.x == 0) {
        scales[h * n_blocks + blockIdx.x] = scale;
    }
}

} // namespace ggml_sage256

size_t ggml_sage256_smem() {
    return ggml_sage256::SMEM;
}

int64_t ggml_sage256_q_scales(int64_t n_q) {
    return (n_q + ggml_sage256::CTA_Q - 1) / ggml_sage256::CTA_Q * ggml_sage256::num_warps_q;
}

int64_t ggml_sage256_k_scales(int64_t n_kv) {
    return (n_kv + ggml_sage256::CTA_K - 1) / ggml_sage256::CTA_K;
}

int64_t ggml_sage256_kpart_rows() {
    return ggml_sage256::KPART_ROWS;
}

void ggml_sage256_attn(const void * q, int64_t q_nb1, int64_t q_nb2, const void * k, bool k_f32, int64_t k_nb1,
        int64_t k_nb2, const void * v, int64_t v_nb1, int64_t v_nb2, float * dst, int64_t o_tok, int64_t o_head,
        int64_t n_q, int64_t n_kv, int64_t H, int64_t H_kv, float sm_scale, bool acc32,
        int8_t * q8, int8_t * k8, float * qs, float * ks, float * ksum, float * kpart, int * flag, cudaStream_t stream) {
    using namespace ggml_sage256;
    // the predicated loads leave rows past the end unfilled: with at least one full key tile every unfilled V row
    // holds finite values of the previous tile (times exactly 0), never the Q staging bytes
    if (n_kv < (int64_t) CTA_K || H % H_kv != 0) {
        fprintf(stderr, "%s: needs at least %u keys and whole K/V head groups\n", __func__, CTA_K);
        abort();
    }
    static bool smem_set[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 64 && !smem_set[dev]) {
        cudaFuncSetAttribute(sage_attn_256<0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SMEM);
        cudaFuncSetAttribute(sage_attn_256<2, true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SMEM);
        cudaFuncSetAttribute(sage_attn_256<2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SMEM);
        smem_set[dev] = true;
    }

    // K: mean over the keys, then int8 blocks of 64 keys
    const int n_part = (int) ((n_kv + KPART_ROWS - 1) / KPART_ROWS);
    const int64_t nbk = ggml_sage256_k_scales(n_kv);
    const float inv_n = 1.0f / (float) n_kv;
    if (k_f32) {
        sage256_k_sum_partial<float><<<dim3(n_part, H_kv), D, 0, stream>>>((const char *) k, kpart, n_kv, k_nb1, k_nb2);
        sage256_k_sum_reduce<<<H_kv, D, 0, stream>>>(kpart, ksum, n_part, (int) H_kv);
        sage256_quantize_rows<float, 64, true><<<dim3(nbk, H_kv), 256, 0, stream>>>((const char *) k, k_nb1, k_nb2, n_kv, ksum, inv_n, k8, ks, nbk);
    } else {
        sage256_k_sum_partial<half><<<dim3(n_part, H_kv), D, 0, stream>>>((const char *) k, kpart, n_kv, k_nb1, k_nb2);
        sage256_k_sum_reduce<<<H_kv, D, 0, stream>>>(kpart, ksum, n_part, (int) H_kv);
        sage256_quantize_rows<half, 64, false><<<dim3(nbk, H_kv), 256, 0, stream>>>((const char *) k, k_nb1, k_nb2, n_kv, ksum, inv_n, k8, ks, nbk);
    }
    // Q: blocks of 16 rows (one scale per warp of the attention kernel)
    const int64_t nbq = ggml_sage256_q_scales(n_q);
    sage256_quantize_rows<float, WARP_Q, false><<<dim3(nbq, H), 256, 0, stream>>>((const char *) q, q_nb1, q_nb2, n_q, nullptr, 0.0f, q8, qs, nbq);

    const dim3 grid((unsigned) ((n_q + CTA_Q - 1) / CTA_Q), (unsigned) H);
    const dim3 block(WARP_SIZE, num_warps);
    const uint32_t gqa = (uint32_t) (H / H_kv);
    const uint32_t v_seq = (uint32_t) (v_nb1 / 2), v_head = (uint32_t) (v_nb2 / 2);
    if (acc32) {
        sage_attn_256<2, false><<<grid, block, SMEM, stream>>>(q8, k8, (const half *) v, dst, qs, ks, (uint32_t) n_q,
            (uint32_t) n_kv, gqa, v_seq, v_head, o_tok, o_head, sm_scale, flag);
        return;
    }
    cudaMemsetAsync(flag, 0, sizeof(int), stream);
    sage_attn_256<0, false><<<grid, block, SMEM, stream>>>(q8, k8, (const half *) v, dst, qs, ks, (uint32_t) n_q,
        (uint32_t) n_kv, gqa, v_seq, v_head, o_tok, o_head, sm_scale, flag);
    // overflow guard: the f32-accumulation rerun exits at once unless the flag is set (GGML_SAGE_NO_GUARD=1: tests)
    static const bool no_guard = getenv("GGML_SAGE_NO_GUARD") != nullptr;
    if (no_guard) {
        return;
    }
    sage_attn_256<2, true><<<grid, block, SMEM, stream>>>(q8, k8, (const half *) v, dst, qs, ks, (uint32_t) n_q,
        (uint32_t) n_kv, gqa, v_seq, v_head, o_tok, o_head, sm_scale, flag);
}
