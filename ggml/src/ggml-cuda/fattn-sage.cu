// SageAttention2-style attention for the decompressed Tilde MLA heads (Q/K 192, V 128, no mask): int8 Q*K^T with
// one scale per 32 queries and per 64 (mean-smoothed) keys, f16 P*V with f16 accumulation over the whole row,
// 32 query rows per warp. Adapted from SageAttention's sm80 kernel (vendor/sageattn, Apache-2.0): the 192-dim Q and K
// tiles are split into a 128-byte part (128B swizzle) and a 64-byte part (64B swizzle), V and O are 128 wide, and the
// output is written as f32 straight from the registers.
//
// f16 accumulation can overflow for very long, flat attention rows: every output is checked and a flag is raised on
// any non-finite value; the same kernel with f32 accumulation then reruns the layer (it exits at once otherwise).

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

#include "fattn-sage.cuh"

namespace ggml_sage {
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#include "vendor/sageattn/math.cuh"
#include "vendor/sageattn/attn_utils.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define GGML_SAGE_AVAILABLE
#endif

constexpr uint32_t CTA_Q  = 128;
constexpr uint32_t CTA_K  = 64;
#ifndef GGML_SAGE_WARP_Q
#define GGML_SAGE_WARP_Q 32
#endif
constexpr uint32_t WARP_Q = GGML_SAGE_WARP_Q; // query rows per warp (32: 4 warps per block, 16: 8 warps)
constexpr uint32_t WARP_K = 64;
constexpr uint32_t DQK    = 192;          // Q/K head size: parts A (128) and B (64)
constexpr uint32_t DA     = 128;
constexpr uint32_t DB     = 64;
constexpr uint32_t DV     = 128;
constexpr uint32_t num_warps_q = CTA_Q / WARP_Q; // 4
constexpr uint32_t num_warps_k = CTA_K / WARP_K; // 1
constexpr uint32_t num_warps   = num_warps_q * num_warps_k;
constexpr uint32_t num_tiles_q = WARP_Q / 16;    // 2
constexpr uint32_t num_tiles_k = WARP_K / 16;    // 4
constexpr uint32_t num_tiles_v = DV / 16;        // 8
// the 64-dim part of Q lives in registers after the first load (staged through the V buffer): 44 KB of shared memory,
// two blocks per SM on sm_86
constexpr uint32_t SMEM_QA = 0;
constexpr uint32_t SMEM_KA = SMEM_QA + CTA_Q * DA;
constexpr uint32_t SMEM_KB = SMEM_KA + CTA_K * DA;
constexpr uint32_t SMEM_V  = SMEM_KB + CTA_K * DB;
constexpr uint32_t SMEM_QB = SMEM_V;
constexpr uint32_t SMEM    = SMEM_V + CTA_K * DV * 2;
static_assert(CTA_Q * DB <= CTA_K * DV * 2, "the Q staging fits in the V buffer");

// S += Q_b K_b^T with the Q fragments in registers (inner steps of 32 of the 64-dim part)
template <SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void sage_qk_regs(const smem_t<swizzle_mode, stride> & smem_K, int32_t RS[][num_tiles_k][8],
        const uint32_t (&RQ)[DB / 32][num_tiles_q][4], uint32_t offset_K) {
#pragma unroll
    for (uint32_t it = 0; it < DB / 32; it++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
            uint32_t RK[4];
            smem_K.ldmatrix_m8n8x4(offset_K, RK);
            offset_K = smem_K.template advance_offset_by_row<16>(offset_K);
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS[fq][fk], const_cast<uint32_t *>(RQ[it][fq]), RK);
            }
        }
        offset_K = smem_K.template advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), it);
    }
}

// lanes per row, rows per warp and iterations of the 16-byte global -> shared copies
constexpr uint32_t LA = 8, RA = 4, LB = 4, RB = 8, LV = 8, RV = 4;

template <typename ACC, bool guarded>
__global__ void __launch_bounds__(num_warps * WARP_SIZE) sage_attn_192_128(
        const int8_t * __restrict__ Q, const int8_t * __restrict__ K, const half * __restrict__ V, float * __restrict__ O,
        const float * __restrict__ Q_scale, const float * __restrict__ K_scale, const uint32_t qo_len, const uint32_t kv_len,
        const uint32_t stride_h_q, const uint32_t stride_h_k, const uint32_t stride_seq_v, const uint32_t stride_h_v,
        const int64_t o_tok, const int64_t o_head, float sm_scale, int * flag) {
#ifdef GGML_SAGE_AVAILABLE
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
    const uint32_t k_scale_idx = head_id * n_kblocks;

    smem_t<SwizzleMode::k128B, DA / 16> smem_Qa(smem + SMEM_QA);
    smem_t<SwizzleMode::k64B,  DB / 16> smem_Qb(smem + SMEM_QB);
    smem_t<SwizzleMode::k128B, DA / 16> smem_Ka(smem + SMEM_KA);
    smem_t<SwizzleMode::k64B,  DB / 16> smem_Kb(smem + SMEM_KB);
    smem_t<SwizzleMode::k128B, DV / 8>  smem_V(smem + SMEM_V);

    constexpr uint32_t QA_col = CTA_Q / (num_warps * RA), QB_col = CTA_Q / (num_warps * RB);
    constexpr uint32_t KA_col = CTA_K / (num_warps * RA), KB_col = CTA_K / (num_warps * RB);
    constexpr uint32_t V_row  = DV / (LV * 8),            V_col  = CTA_K / (num_warps * RV);

    const int8_t * qa_ptr = Q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / LA) * DQK + (lane_id % LA) * 16;
    const int8_t * qb_ptr = Q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / LB) * DQK + DA + (lane_id % LB) * 16;
    const int8_t * ka_ptr = K + head_id * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / LA) * DQK + (lane_id % LA) * 16;
    const int8_t * kb_ptr = K + head_id * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / LB) * DQK + DA + (lane_id % LB) * 16;
    const half   * v_ptr  = V + head_id * stride_h_v + (CTA_K / num_warps * warp_id + lane_id / LV) * stride_seq_v + (lane_id % LV) * 8;
    // the loaders take non-const pointers
    int8_t * qa = (int8_t *) qa_ptr; int8_t * qb = (int8_t *) qb_ptr;
    int8_t * ka = (int8_t *) ka_ptr; int8_t * kb = (int8_t *) kb_ptr;
    half   * vp = (half *) v_ptr;

    uint32_t qa_ld = smem_Qa.get_permuted_offset(warp_id * RA * QA_col + lane_id / LA, lane_id % LA);
    uint32_t qb_ld = smem_Qb.get_permuted_offset(warp_id * RB * QB_col + lane_id / LB, lane_id % LB);
    uint32_t ka_ld = smem_Ka.get_permuted_offset(warp_id * RA * KA_col + lane_id / LA, lane_id % LA);
    uint32_t kb_ld = smem_Kb.get_permuted_offset(warp_id * RB * KB_col + lane_id / LB, lane_id % LB);
    uint32_t v_ld  = smem_V.get_permuted_offset(warp_id * RV * V_col + lane_id / LV, lane_id % LV);

    uint32_t qa_mma = smem_Qa.get_permuted_offset(wq * WARP_Q + lane_id % 16, lane_id / 16);
    uint32_t qb_mma = smem_Qb.get_permuted_offset(wq * WARP_Q + lane_id % 16, lane_id / 16);
    uint32_t ka_mma = smem_Ka.get_permuted_offset(wk * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
    uint32_t kb_mma = smem_Kb.get_permuted_offset(wk * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
    uint32_t v_mma  = smem_V.get_permuted_offset(wk * WARP_K + lane_id % 16, lane_id / 16);

    const uint32_t row_qk  = DQK; // the loaders take the row stride by reference
    const uint32_t q_row_a = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / LA;
    const uint32_t q_row_b = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / LB;
    uint32_t k_row_a = CTA_K / num_warps * warp_id + lane_id / LA;
    uint32_t k_row_b = CTA_K / num_warps * warp_id + lane_id / LB;
    uint32_t v_row   = CTA_K / num_warps * warp_id + lane_id / LV;
    uint32_t k_idx   = wk * WARP_K + 2 * (lane_id % 4); // key index of the lane's score fragments (out-of-bound mask)

    load_global_to_share<LA, RA, DA / (LA * 16), QA_col, SwizzleMode::k128B, DA / 16, CTA_Q>(&qa, qa_ld, row_qk, smem_Qa, q_row_a, qo_len);
    load_global_to_share<LB, RB, DB / (LB * 16), QB_col, SwizzleMode::k64B,  DB / 16, CTA_Q>(&qb, qb_ld, row_qk, smem_Qb, q_row_b, qo_len);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();
    // Q_b fragments to registers (as compute_int_qk reads them), then its staging space is free for V
    uint32_t RQb[DB / 32][num_tiles_q][4];
#pragma unroll
    for (uint32_t it = 0; it < DB / 32; it++) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            smem_Qb.ldmatrix_m8n8x4(qb_mma, RQb[it][fq]);
            qb_mma = smem_Qb.template advance_offset_by_row<16>(qb_mma);
        }
        qb_mma = smem_Qb.template advance_offset_by_column<2>(qb_mma - (num_tiles_q * 16 * (DB / 16)), it);
    }
    __syncthreads();

    load_global_to_share<LA, RA, DA / (LA * 16), KA_col, SwizzleMode::k128B, DA / 16, CTA_K>(&ka, ka_ld, row_qk, smem_Ka, k_row_a, kv_len);
    load_global_to_share<LB, RB, DB / (LB * 16), KB_col, SwizzleMode::k64B,  DB / 16, CTA_K>(&kb, kb_ld, row_qk, smem_Kb, k_row_b, kv_len);
    cp_async::commit_group();
    load_global_to_share<LV, RV, V_row, V_col, SwizzleMode::k128B, DV / 8, CTA_K>(&vp, v_ld, stride_seq_v, smem_V, v_row, kv_len);
    cp_async::commit_group();
    k_row_a += CTA_K; k_row_b += CTA_K; v_row += CTA_K;

    const float q_scale = Q_scale[q_scale_idx];
    float scale = sm_scale * q_scale * K_scale[k_scale_idx];

    for (uint32_t it = 0; it < n_kblocks; it++) {
        const bool last = it + 1 == n_kblocks;
        cp_async::wait_group<1>(); // K
        __syncthreads();

        compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, DA / 32, SwizzleMode::k128B, DA / 16, DataType::kInt8, false>(
            smem_Qa, smem_Ka, RS, qa_mma, ka_mma);
        sage_qk_regs(smem_Kb, RS, RQb, kb_mma);

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

        update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, std::is_same<ACC, half>::value, false, false>(RS_f32, RO, m, d, scale);

        uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
        RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);
        accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);

        __syncthreads();
        if (!last) {
            load_global_to_share<LA, RA, DA / (LA * 16), KA_col, SwizzleMode::k128B, DA / 16, CTA_K>(&ka, ka_ld, row_qk, smem_Ka, k_row_a, kv_len);
            load_global_to_share<LB, RB, DB / (LB * 16), KB_col, SwizzleMode::k64B,  DB / 16, CTA_K>(&kb, kb_ld, row_qk, smem_Kb, k_row_b, kv_len);
            cp_async::commit_group();
            scale = sm_scale * q_scale * K_scale[k_scale_idx + it + 1];
            cp_async::wait_group<1>(); // V
        } else {
            cp_async::wait_group<0>();
        }
        __syncthreads();

        compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, SwizzleMode::k128B, DV / 8, 4>(
            smem_V, RS_f16, RO, d, v_mma);

        __syncthreads();
        if (!last) {
            load_global_to_share<LV, RV, V_row, V_col, SwizzleMode::k128B, DV / 8, CTA_K>(&vp, v_ld, stride_seq_v, smem_V, v_row, kv_len);
            cp_async::commit_group();
            k_row_a += CTA_K; k_row_b += CTA_K; v_row += CTA_K;
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
    if constexpr (!guarded) {
        if (__any_sync(0xffffffff, bad) && lane_id == 0) {
            atomicExch(flag, 1);
        }
    }
#else
    (void) Q; (void) K; (void) V; (void) O; (void) Q_scale; (void) K_scale; (void) qo_len; (void) kv_len; (void) stride_h_q;
    (void) stride_h_k; (void) stride_seq_v; (void) stride_h_v; (void) o_tok; (void) o_head; (void) sm_scale; (void) flag;
#endif // GGML_SAGE_AVAILABLE
}

// rows of R tokens per block share one int8 scale (max |x| / 127 over the block): Q blocks of 32, K blocks of 64 (K
// minus its mean over all keys first). One warp per row, 6 values per lane.
template <typename T, int R, bool round_f16>
__global__ void sage_quantize_rows(const char * __restrict__ src, const int64_t nb1, const int64_t nb2, const int64_t n,
        const float * __restrict__ ksum, const float inv_n, int8_t * __restrict__ dst, float * __restrict__ scales,
        const int64_t n_blocks) {
    constexpr int W = 8;            // warps per block
    constexpr int PER = R / W;      // rows per warp
    constexpr int PL = DQK / WARP_SIZE;
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
                    v -= ksum[h * DQK + dd] * inv_n;
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
            int8_t * o = dst + ((int64_t) h * n + row) * DQK;
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

} // namespace ggml_sage

size_t ggml_sage_smem() {
    return ggml_sage::SMEM;
}

int64_t ggml_sage_q_scales(int64_t n_q) {
    return (n_q + ggml_sage::CTA_Q - 1) / ggml_sage::CTA_Q * ggml_sage::num_warps_q;
}

int64_t ggml_sage_k_scales(int64_t n_kv) {
    return (n_kv + ggml_sage::CTA_K - 1) / ggml_sage::CTA_K;
}

void ggml_sage_quantize(const void * src, bool src_f32, bool round_f16, int64_t nb1, int64_t nb2, int64_t n, int64_t H,
        const float * ksum, int8_t * dst, float * scales, cudaStream_t stream) {
    using namespace ggml_sage;
    if (ksum == nullptr) {
        // Q: blocks of WARP_Q rows (one scale per warp of the attention kernel)
        const int64_t nb = ggml_sage_q_scales(n);
        const dim3 grid((unsigned) nb, (unsigned) H);
        if (src_f32) {
            sage_quantize_rows<float, WARP_Q, false><<<grid, 256, 0, stream>>>((const char *) src, nb1, nb2, n, nullptr, 0.0f, dst, scales, nb);
        } else {
            sage_quantize_rows<half, WARP_Q, false><<<grid, 256, 0, stream>>>((const char *) src, nb1, nb2, n, nullptr, 0.0f, dst, scales, nb);
        }
    } else {
        const int64_t nb = ggml_sage_k_scales(n);
        const dim3 grid((unsigned) nb, (unsigned) H);
        const float inv_n = 1.0f / (float) n;
        if (src_f32 && round_f16) {
            sage_quantize_rows<float, 64, true><<<grid, 256, 0, stream>>>((const char *) src, nb1, nb2, n, ksum, inv_n, dst, scales, nb);
        } else if (src_f32) {
            sage_quantize_rows<float, 64, false><<<grid, 256, 0, stream>>>((const char *) src, nb1, nb2, n, ksum, inv_n, dst, scales, nb);
        } else {
            sage_quantize_rows<half, 64, false><<<grid, 256, 0, stream>>>((const char *) src, nb1, nb2, n, ksum, inv_n, dst, scales, nb);
        }
    }
}

void ggml_sage_attn(const int8_t * q8, const int8_t * k8, const void * v, int64_t v_seq, int64_t v_head, const float * qs,
        const float * ks, float * dst, int64_t o_tok, int64_t o_head, int64_t n_q, int64_t n_kv, int64_t H, float sm_scale,
        int * flag, cudaStream_t stream) {
    using namespace ggml_sage;
    // the predicated loads leave rows past the end unfilled: with at least one full key tile every unfilled V row
    // holds finite values of the previous tile (times exactly 0), never the Q staging bytes
    if (n_kv < (int64_t) CTA_K) {
        fprintf(stderr, "%s: needs at least %u keys\n", __func__, CTA_K);
        abort();
    }
    static bool smem_set[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 64 && !smem_set[dev]) {
        cudaFuncSetAttribute(sage_attn_192_128<half, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SMEM);
        cudaFuncSetAttribute(sage_attn_192_128<float, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SMEM);
        smem_set[dev] = true;
    }
    const dim3 grid((unsigned) ((n_q + CTA_Q - 1) / CTA_Q), (unsigned) H);
    const dim3 block(WARP_SIZE, num_warps);
    cudaMemsetAsync(flag, 0, sizeof(int), stream);
    sage_attn_192_128<half, false><<<grid, block, SMEM, stream>>>(q8, k8, (const half *) v, dst, qs, ks, (uint32_t) n_q,
        (uint32_t) n_kv, (uint32_t) (n_q * DQK), (uint32_t) (n_kv * DQK), (uint32_t) v_seq, (uint32_t) v_head, o_tok, o_head,
        sm_scale, flag);
    // overflow guard: the f32-accumulation rerun exits at once unless the flag is set (GGML_SAGE_NO_GUARD=1: tests)
    static const bool no_guard = getenv("GGML_SAGE_NO_GUARD") != nullptr;
    if (no_guard) {
        return;
    }
    sage_attn_192_128<float, true><<<grid, block, SMEM, stream>>>(q8, k8, (const half *) v, dst, qs, ks, (uint32_t) n_q,
        (uint32_t) n_kv, (uint32_t) (n_q * DQK), (uint32_t) (n_kv * DQK), (uint32_t) v_seq, (uint32_t) v_head, o_tok, o_head,
        sm_scale, flag);
}
