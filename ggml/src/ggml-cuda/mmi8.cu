#include "mmi8.cuh"

#ifdef GGML_CUDA_USE_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#endif // GGML_CUDA_USE_CUTLASS

// activations: one warp per token row, x [K] f32 -> q [K] int8 with scale max|x| / 127. Rows of up to 32*4*NV floats
// are held in registers (one read of x); longer rows read x twice.
template <int NV>
static __global__ void mmi8_quantize_rows(const char * __restrict__ x, int8_t * __restrict__ q, float * __restrict__ s,
        const int64_t K, const int64_t n_rows, const int64_t nb1, const int64_t nb2, const int64_t ne1) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (row >= n_rows) {
        return;
    }
    const float4 * xr = (const float4 *) (x + (row % ne1) * nb1 + (row / ne1) * nb2);
    const int K4 = (int) (K / 4);
    char4 * qr = (char4 *) (q + row * K);
    float amax = 0.0f;
    if (NV > 0 && K4 <= NV * WARP_SIZE) {
        float4 v[NV > 0 ? NV : 1];
#pragma unroll
        for (int j = 0; j < (NV > 0 ? NV : 1); j++) {
            const int i = lane + j * WARP_SIZE;
            v[j] = i < K4 ? xr[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[j].x), fabsf(v[j].y)), fmaxf(fabsf(v[j].z), fabsf(v[j].w))));
        }
        amax = warp_reduce_max(amax);
        const float inv = amax > 0.0f ? 127.0f / amax : 1.0f;
#pragma unroll
        for (int j = 0; j < (NV > 0 ? NV : 1); j++) {
            const int i = lane + j * WARP_SIZE;
            if (i < K4) {
                qr[i] = make_char4((signed char) __float2int_rn(v[j].x * inv), (signed char) __float2int_rn(v[j].y * inv),
                                   (signed char) __float2int_rn(v[j].z * inv), (signed char) __float2int_rn(v[j].w * inv));
            }
        }
    } else {
        for (int i = lane; i < K4; i += WARP_SIZE) {
            const float4 v = xr[i];
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
        amax = warp_reduce_max(amax);
        const float inv = amax > 0.0f ? 127.0f / amax : 1.0f;
        for (int i = lane; i < K4; i += WARP_SIZE) {
            const float4 v = xr[i];
            qr[i] = make_char4((signed char) __float2int_rn(v.x * inv), (signed char) __float2int_rn(v.y * inv),
                               (signed char) __float2int_rn(v.z * inv), (signed char) __float2int_rn(v.w * inv));
        }
    }
    if (lane == 0) {
        s[row] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
}

// the last node index (from node_idx + 1 on) before which no node writes into the memory of t
static int mmi8_valid_until(const ggml_cgraph * cgraph, int node_idx, const ggml_tensor * t) {
    const char * lo = (const char *) t->data;
    const char * hi = lo + ggml_nbytes(t);
    const int last = std::min(cgraph->n_nodes - 1, node_idx + 512);
    for (int j = node_idx + 1; j <= last; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_NONE || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE ||
                n->op == GGML_OP_TRANSPOSE || n->data == nullptr) {
            continue;
        }
        const char * nlo = (const char *) n->data;
        const char * nhi = nlo + ggml_nbytes(n);
        if (nlo < hi && lo < nhi) {
            return j - 1;
        }
    }
    return last;
}

// per-token int8 copy of the rows of src1 ([K, ne1, ne2], rows (i1, i2)), kept in the context and reused by the next
// I8 product that reads the same data before a later node of the graph writes to it
static void mmi8_token_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_cgraph * cgraph,
        const int node_idx, int8_t * & q, float * & s) {
    const int64_t K = src1->ne[0];
    const int64_t T = src1->ne[1] * src1->ne[2];
    cudaStream_t stream = ctx.stream();
    auto & cache = ctx.i8_act;
    const bool hit = cgraph != nullptr && cache.graph == cgraph && cache.data == src1->data && cache.K == K &&
        cache.n_rows == T && cache.nb1 == (int64_t) src1->nb[1] && cache.nb2 == (int64_t) src1->nb[2] &&
        cache.ne1 == src1->ne[1] && node_idx <= cache.valid_until;
    if (!hit) {
        if ((size_t) (T * K) > cache.q_size || (size_t) T > cache.s_size) {
            // grow (never during a CUDA graph capture: the first evaluation of a new graph shape is not captured);
            // the old buffers stay allocated because a captured graph may still refer to them
            if (cache.q != nullptr) {
                cache.retired.push_back(cache.q);
                cache.retired.push_back(cache.s);
            }
            cache.q_size = std::max((size_t) (T * K), cache.q_size + cache.q_size / 2);
            cache.s_size = std::max((size_t) T, cache.s_size + cache.s_size / 2);
            CUDA_CHECK(cudaMalloc((void **) &cache.q, cache.q_size));
            CUDA_CHECK(cudaMalloc((void **) &cache.s, cache.s_size * sizeof(float)));
        }
        const int rows_per_block = 8;
        const dim3 grid((T + rows_per_block - 1) / rows_per_block);
        const int threads = rows_per_block * WARP_SIZE;
        if (K <= 4 * 16 * WARP_SIZE) {
            mmi8_quantize_rows<16><<<grid, threads, 0, stream>>>(
                (const char *) src1->data, cache.q, cache.s, K, T, src1->nb[1], src1->nb[2], src1->ne[1]);
        } else {
            mmi8_quantize_rows<0><<<grid, threads, 0, stream>>>(
                (const char *) src1->data, cache.q, cache.s, K, T, src1->nb[1], src1->nb[2], src1->ne[1]);
        }
        CUDA_CHECK(cudaGetLastError());
        cache.graph       = cgraph;
        cache.data        = cgraph != nullptr ? src1->data : nullptr;
        cache.K           = K;
        cache.n_rows      = T;
        cache.nb1         = src1->nb[1];
        cache.nb2         = src1->nb[2];
        cache.ne1         = src1->ne[1];
        cache.valid_until = cgraph != nullptr ? mmi8_valid_until(cgraph, node_idx, src1) : -1;
    }
    q = cache.q;
    s = cache.s;
}

#ifdef GGML_CUDA_USE_CUTLASS
namespace {
using namespace cute;

// A = activations [T, K] int8 row-major, B = weights [K, M] int8 column-major (each weight row is K contiguous codes),
// D = [T, M] f32 row-major (ggml dst [M, T]); D = acc * sa[t] * sb[m]
struct mmi8_cfg {
    using TB  = cutlass::gemm::GemmShape<128, 256, 64>;
    using WS  = cutlass::gemm::GemmShape<64, 64, 64>;
    using IS  = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int stages = 3;
    using Map   = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, WS, float, 4, 1>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using SA    = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, Stride<_1, _0, int32_t>>;
    using SB    = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, Stride<_0, _1, int32_t>>;
    using Mul   = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float,
                                                                 cutlass::FloatRoundStyle::round_to_nearest>;
    using E0    = cutlass::epilogue::threadblock::Sm80EVT<Mul, Accum, SB>;
    using E1    = cutlass::epilogue::threadblock::Sm80EVT<Mul, E0, SA>;
    using D     = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest,
                                                                   Stride<int64_t, _1, int64_t>>;
    using EVT   = cutlass::epilogue::threadblock::Sm80EVT<D, E1>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
        float, cutlass::layout::RowMajor, 4, int32_t, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, IS, EVT,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, stages,
        cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
} // namespace
#endif // GGML_CUDA_USE_CUTLASS

bool ggml_cuda_mul_mat_i8_supported(int device, const ggml_tensor * dst) {
#ifdef GGML_CUDA_USE_CUTLASS
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int cc = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE &&
        dst->op == GGML_OP_MUL_MAT && src0->type == GGML_TYPE_I8 && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->nb[0] == sizeof(float) && src1->ne[3] == 1 && (src1->ne[2] == 1 || src1->nb[2] == src1->nb[1] * src1->ne[1]) &&
        ggml_is_contiguous(dst) && src0->ne[0] % 16 == 0 && src0->ne[1] % 4 == 0 &&
        src1->nb[1] % 16 == 0 && ggml_cuda_info().devices[device].smpbo >= sizeof(typename mmi8_cfg::Kernel::SharedStorage);
#else
    GGML_UNUSED_VARS(device, dst);
    return false;
#endif // GGML_CUDA_USE_CUTLASS
}

bool ggml_cuda_mul_mat_i8_scale_fusable(const ggml_tensor * mm, const ggml_tensor * mul) {
    const ggml_tensor * s = mul->src[0] == mm ? mul->src[1] : mul->src[0];
    return mm->src[0]->type == GGML_TYPE_I8 && s->type == GGML_TYPE_F32 && ggml_is_contiguous(s) &&
        ggml_nelements(s) == mm->ne[0] && s->ne[0] == mm->ne[0] && ggml_are_same_shape(mm, mul) &&
        mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul);
}

void ggml_cuda_mul_mat_i8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * mm, const float * w_scale, ggml_tensor * out, const ggml_cgraph * cgraph, int node_idx) {
#ifdef GGML_CUDA_USE_CUTLASS
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t T = src1->ne[1] * src1->ne[2];
    cudaStream_t stream = ctx.stream();

    // per-token int8 activations, reused from the previous I8 product on the same data when still valid
    int8_t * xq = nullptr;
    float  * xs = nullptr;
    mmi8_token_q8(ctx, src1, cgraph, node_idx, xq, xs);

    using C = mmi8_cfg;
    float * d = (float *) out->data;
    typename C::EVT::Arguments cb{
        { { {}, {w_scale, 1.0f, {_0{}, _1{}, int32_t(M)}}, {} }, {xs, 1.0f, {_1{}, _0{}, int32_t(T)}}, {} },
        {d, {int64_t(out->nb[1] / sizeof(float)), _1{}, int64_t(T) * M}} };
    typename C::Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {(int) T, (int) M, (int) K}, 1, cb,
        xq, (const int8_t *) src0->data, nullptr, nullptr, T * K, M * K, 0, 0, K, src0->nb[1], 0, 0);

    typename C::Gemm gemm;
    const size_t ws_size = C::Gemm::get_workspace_size(args);
    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), ws_size > 0 ? ws_size : 1);
    GGML_ASSERT(gemm.can_implement(args) == cutlass::Status::kSuccess);
    GGML_ASSERT(gemm.initialize(args, ws.get(), stream) == cutlass::Status::kSuccess);
    GGML_ASSERT(gemm.run(stream) == cutlass::Status::kSuccess);
    CUDA_CHECK(cudaGetLastError());
    GGML_UNUSED(mm);
#else
    GGML_UNUSED_VARS(ctx, src0, src1, mm, w_scale, out, cgraph, node_idx);
    GGML_ABORT("the int8 GEMM needs a build with GGML_CUDA_CUTLASS_DIR");
#endif // GGML_CUDA_USE_CUTLASS
}

// ---------------------------------------------------------------------------------------------------------------------
// MUL_MAT_ID with I8 experts

#include "mmid.cuh"

#ifdef GGML_CUDA_USE_CUTLASS
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace {
// the int32 products leave the GEMM as f16, scaled into range (mmid_i8_d_scale), which halves the bytes the
// epilogue pass reads
struct mmid_i8_cfg {
    using Epi = cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, int32_t, float>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
        cutlass::half_t, cutlass::layout::RowMajor, int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
        Epi, cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;
};
} // namespace
#endif // GGML_CUDA_USE_CUTLASS

// gathered rows: compact row r holds token slot ids_dst[r] = token * n_used + slot, i.e. src1 row
// (token, slot % ne11)
template <int NV>
static __global__ void mmid_i8_quantize_rows(const char * __restrict__ x, const int32_t * __restrict__ ids_dst,
        int8_t * __restrict__ q, float * __restrict__ s, const int64_t K, const int64_t n_rows, const int64_t nb11,
        const int64_t nb12, const int n_used, const int ne11) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (row >= n_rows) {
        return;
    }
    const int slot_id = ids_dst[row];
    const float4 * xr = (const float4 *) (x + (int64_t) (slot_id / n_used) * nb12 + (int64_t) ((slot_id % n_used) % ne11) * nb11);
    const int K4 = (int) (K / 4);
    char4 * qr = (char4 *) (q + row * K);
    float4 v[NV];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < NV; j++) {
        const int i = lane + j * WARP_SIZE;
        v[j] = i < K4 ? xr[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[j].x), fabsf(v[j].y)), fmaxf(fabsf(v[j].z), fabsf(v[j].w))));
    }
    amax = warp_reduce_max(amax);
    const float inv = amax > 0.0f ? 127.0f / amax : 1.0f;
#pragma unroll
    for (int j = 0; j < NV; j++) {
        const int i = lane + j * WARP_SIZE;
        if (i < K4) {
            qr[i] = make_char4((signed char) __float2int_rn(v[j].x * inv), (signed char) __float2int_rn(v[j].y * inv),
                               (signed char) __float2int_rn(v[j].z * inv), (signed char) __float2int_rn(v[j].w * inv));
        }
    }
    if (lane == 0) {
        s[row] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
}

// compact row r <- the int8 row of its token (token = ids_fwd[r] / n_used), with its scale
static __global__ void mmid_i8_gather_rows(const int8_t * __restrict__ q, const float * __restrict__ s,
        const int32_t * __restrict__ ids_fwd, int8_t * __restrict__ A, float * __restrict__ sa, const int64_t K,
        const int64_t n_rows, const int n_used) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (row >= n_rows) {
        return;
    }
    const int64_t token = ids_fwd[row] / n_used;
    const int4 * src = (const int4 *) (q + token * K);
    int4 * dst = (int4 *) (A + row * K);
    for (int64_t i = lane; i < K / 16; i += WARP_SIZE) {
        dst[i] = src[i];
    }
    if (lane == 0) {
        sa[row] = s[token];
    }
}

#ifdef GGML_CUDA_USE_CUTLASS
// the per-expert problems of the grouped GEMM from the expert bounds (device side, no host sync)
static __global__ void mmid_i8_setup(const int32_t * __restrict__ bounds, const int E, const int N, const int K,
        int8_t * A, const int8_t * W, const int64_t w_nb2, cutlass::half_t * D,
        cutlass::gemm::GemmCoord * problems, int8_t ** pa, int8_t ** pb, cutlass::half_t ** pd,
        int64_t * lda, int64_t * ldb, int64_t * ldd, const int64_t w_nb1) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }
    const int b0 = bounds[e];
    const int b1 = bounds[e + 1];
    problems[e] = cutlass::gemm::GemmCoord(b1 - b0, N, K);
    pa[e] = A + (int64_t) b0 * K;
    pb[e] = (int8_t *) W + e * w_nb2;
    pd[e] = D + (int64_t) b0 * N;
    lda[e] = K;
    ldb[e] = w_nb1;
    ldd[e] = N;
}
#endif // GGML_CUDA_USE_CUTLASS

// the GEMM writes acc * 2^-e as f16, with e such that K * 127^2 * 2^-e < 60000 (no overflow); the readers multiply back
static float mmid_i8_d_scale(const int64_t K) {
    return 1.0f / exp2f(ceilf(log2f((float) K * 16129.0f / 60000.0f)));
}

static __device__ __forceinline__ float4 mmid_i8_load4(const half * p, const float k) {
    const float2 a = __half22float2(*(const half2 *) p);
    const float2 b = __half22float2(*(const half2 *) (p + 2));
    return make_float4(a.x * k, a.y * k, b.x * k, b.y * k);
}

// out[ids_dst[r]][n] = D[r][n] * sa[r] * sb[e(r)][n]; one block per (row, chunk of 4*blockDim columns): row in x
static __global__ void mmid_i8_scatter(const half * __restrict__ D, const float dk, const float * __restrict__ sa,
        const float * __restrict__ w_scale, const int32_t * __restrict__ bounds, const int32_t * __restrict__ ids_dst,
        float * __restrict__ out, const int E, const int N, const int64_t out_nb1) {
    const int64_t r = blockIdx.x;
    // expert of row r: the last e with bounds[e] <= r
    int lo = 0, hi = E - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (bounds[mid] <= r) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int e = lo;
    const float s = sa[r];
    const int n = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (n >= N) {
        return;
    }
    const float4 d = mmid_i8_load4(D + r * N + n, dk);
    float4 o;
    if (w_scale != nullptr) {
        const float4 w = *(const float4 *) (w_scale + (int64_t) e * N + n);
        o = make_float4(d.x * s * w.x, d.y * s * w.y, d.z * s * w.z, d.w * s * w.w);
    } else {
        o = make_float4(d.x * s, d.y * s, d.z * s, d.w * s);
    }
    *(float4 *) ((char *) out + (int64_t) ids_dst[r] * out_nb1 + n * sizeof(float)) = o;
}

static __device__ __forceinline__ int mmid_i8_expert_of_row(const int32_t * __restrict__ bounds, const int E, const int64_t r) {
    int lo = 0, hi = E - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (bounds[mid] <= r) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// gate_up -> SwiGLU: out[slot][n] = silu(g) * u with g = D[r][n], u = D[r][n + H] (dequantized), H = N / 2
static __global__ void mmid_i8_swiglu(const half * __restrict__ D, const float dk, const float * __restrict__ sa,
        const float * __restrict__ w_scale, const int32_t * __restrict__ bounds, const int32_t * __restrict__ ids_dst,
        float * __restrict__ out, const int E, const int N, const int64_t out_nb1) {
    const int64_t r = blockIdx.x;
    const int H = N / 2;
    const int n = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (n >= H) {
        return;
    }
    const int e = mmid_i8_expert_of_row(bounds, E, r);
    const float s = sa[r];
    const float4 dg = mmid_i8_load4(D + r * N + n, dk);
    const float4 du = mmid_i8_load4(D + r * N + H + n, dk);
    float4 wg = make_float4(1.0f, 1.0f, 1.0f, 1.0f), wu = wg;
    if (w_scale != nullptr) {
        wg = *(const float4 *) (w_scale + (int64_t) e * N + n);
        wu = *(const float4 *) (w_scale + (int64_t) e * N + H + n);
    }
    auto sw = [](const float g, const float u) { return g / (1.0f + expf(-g)) * u; };
    const float4 o = make_float4(sw(dg.x * s * wg.x, du.x * s * wu.x), sw(dg.y * s * wg.y, du.y * s * wu.y),
                                 sw(dg.z * s * wg.z, du.z * s * wu.z), sw(dg.w * s * wg.w, du.w * s * wu.w));
    *(float4 *) ((char *) out + (int64_t) ids_dst[r] * out_nb1 + n * sizeof(float)) = o;
}

// down -> router-weighted sum over the token's experts: out[t][n] = sum_s w[s, t] * D[inv[t * n_used + s]][n] * ...
// (only pool buffers are read: the output may share memory with the ids and router weights of the graph, which are
// dead once the fused nodes are done)
static __global__ void mmid_i8_reduce(const half * __restrict__ D, const float dk, const float * __restrict__ sa,
        const float * __restrict__ w_scale, const int32_t * __restrict__ inv, const int32_t * __restrict__ bounds,
        const int E, const float * __restrict__ weights, float * __restrict__ out, const int N,
        const int n_used, const int64_t out_nb1) {
    const int64_t t = blockIdx.x;
    const int n = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (n >= N) {
        return;
    }
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int sl = 0; sl < n_used; sl++) {
        const int64_t r = inv[t * n_used + sl];
        const int e = mmid_i8_expert_of_row(bounds, E, r);
        const float f = sa[r] * weights[t * n_used + sl];
        const float4 d = mmid_i8_load4(D + r * N + n, dk);
        float4 w = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        if (w_scale != nullptr) {
            w = *(const float4 *) (w_scale + (int64_t) e * N + n);
        }
        acc.x += d.x * f * w.x; acc.y += d.y * f * w.y; acc.z += d.z * f * w.z; acc.w += d.w * f * w.w;
    }
    *(float4 *) ((char *) out + t * out_nb1 + n * sizeof(float)) = acc;
}

bool ggml_cuda_mul_mat_id_i8_supported(int device, const ggml_tensor * dst) {
#ifdef GGML_CUDA_USE_CUTLASS
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const int cc = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE &&
        dst->op == GGML_OP_MUL_MAT_ID && src0->type == GGML_TYPE_I8 && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
        ggml_is_contiguous(src0) && src0->ne[3] == 1 && src0->ne[0] % 16 == 0 && src0->ne[1] % 4 == 0 &&
        src0->ne[0] <= 4 * 16 * WARP_SIZE && src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 &&
        src1->ne[3] == 1 && src1->nb[2] % src1->nb[1] == 0 && ggml_is_contiguous(dst) && dst->ne[1] == ids->ne[0] &&
        (dst->src[3] == nullptr || (dst->src[3]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst->src[3]))) &&
        ggml_cuda_info().devices[device].smpbo >= (size_t) src1->ne[2] * 4;
#else
    GGML_UNUSED_VARS(device, dst);
    return false;
#endif // GGML_CUDA_USE_CUTLASS
}

void ggml_cuda_mul_mat_id_i8(ggml_backend_cuda_context & ctx, const ggml_tensor * mmid, const float * w_scale, ggml_tensor * out,
        mmid_i8_epilogue epilogue, const ggml_tensor * weights, const ggml_cgraph * cgraph, int node_idx) {
#ifdef GGML_CUDA_USE_CUTLASS
    const ggml_tensor * src0 = mmid->src[0];
    const ggml_tensor * src1 = mmid->src[1];
    const ggml_tensor * ids  = mmid->src[2];
    cudaStream_t stream = ctx.stream();

    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int E = (int) src0->ne[2];
    const int n_used = (int) ids->ne[0];
    const int n_tokens = (int) src1->ne[2];
    const int64_t R = (int64_t) n_used * n_tokens;

    // the expert sort (forward map: compact row -> token slot; inverse map; expert bounds), reused by the next I8
    // MUL_MAT_ID on the same ids (gate_up then down) unless a later node overwrites them
    auto & ic = ctx.i8_ids;
    const bool ids_hit = cgraph != nullptr && ic.graph == cgraph && ic.ids == ids->data && ic.n_used == n_used &&
        ic.n_tokens == n_tokens && ic.n_expert == E && ic.nb1 == (int64_t) ids->nb[1] && node_idx <= ic.valid_until;
    if (!ids_hit) {
        if ((size_t) R > ic.size || (size_t) (E + 1) > ic.bounds_size) {
            if (ic.fwd != nullptr) {
                ic.retired.push_back(ic.fwd);
                ic.retired.push_back(ic.inv);
                ic.retired.push_back(ic.bounds);
            }
            ic.size        = std::max((size_t) R, ic.size + ic.size / 2);
            ic.bounds_size = std::max((size_t) (E + 1), ic.bounds_size);
            CUDA_CHECK(cudaMalloc((void **) &ic.fwd, ic.size * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc((void **) &ic.inv, ic.size * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc((void **) &ic.bounds, ic.bounds_size * sizeof(int32_t)));
        }
        const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
        const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ic.inv, ic.fwd, ic.bounds,
            E, n_tokens, n_used, (int) src1->ne[1], si1, sis1, /*write_inverse =*/ true, stream);
        CUDA_CHECK(cudaGetLastError());
        ic.graph       = cgraph;
        ic.ids         = cgraph != nullptr ? ids->data : nullptr;
        ic.n_used      = n_used;
        ic.n_tokens    = n_tokens;
        ic.n_expert    = E;
        ic.nb1         = ids->nb[1];
        ic.valid_until = cgraph != nullptr ? mmi8_valid_until(cgraph, node_idx, ids) : -1;
    }
    int32_t * ids_inv = ic.inv;
    int32_t * ids_fwd = ic.fwd;
    int32_t * bounds  = ic.bounds;

    // the router weights, copied before anything is written (the fused output may reuse their memory)
    ggml_cuda_pool_alloc<float> w_copy(ctx.pool());
    if (epilogue == MMID_I8_REDUCE) {
        GGML_ASSERT(weights != nullptr && ggml_is_contiguous(weights) && ggml_nelements(weights) == R);
        w_copy.alloc(R);
        CUDA_CHECK(cudaMemcpyAsync(w_copy.get(), weights->data, R * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    ggml_cuda_pool_alloc<int8_t> A(ctx.pool(), R * K);
    ggml_cuda_pool_alloc<float>  sa(ctx.pool(), R);
    if (src1->ne[1] == 1) {
        // the same input row for every expert of a token (gate_up): quantize each token once (shared with the dense
        // products on the same data, e.g. the shared expert), then gather the int8 rows
        int8_t * tq_q = nullptr;
        float  * tq_s = nullptr;
        mmi8_token_q8(ctx, src1, cgraph, node_idx, tq_q, tq_s);
        mmid_i8_gather_rows<<<(R + 7) / 8, 8 * WARP_SIZE, 0, stream>>>(tq_q, tq_s, ids_fwd, A.get(), sa.get(), K, R, n_used);
    } else {
        const int rows_per_block = 8;
        mmid_i8_quantize_rows<16><<<(R + rows_per_block - 1) / rows_per_block, rows_per_block * WARP_SIZE, 0, stream>>>(
            (const char *) src1->data, ids_fwd, A.get(), sa.get(), K, R, src1->nb[1], src1->nb[2], n_used, (int) src1->ne[1]);
    }
    CUDA_CHECK(cudaGetLastError());

    using C = mmid_i8_cfg;
    const float d_scale = mmid_i8_d_scale(K);
    ggml_cuda_pool_alloc<cutlass::half_t> D(ctx.pool(), R * N);
    ggml_cuda_pool_alloc<cutlass::gemm::GemmCoord> problems(ctx.pool(), E);
    ggml_cuda_pool_alloc<int8_t *>  pa(ctx.pool(), E);
    ggml_cuda_pool_alloc<int8_t *>  pb(ctx.pool(), E);
    ggml_cuda_pool_alloc<cutlass::half_t *> pd(ctx.pool(), E);
    ggml_cuda_pool_alloc<int64_t>   lda(ctx.pool(), E);
    ggml_cuda_pool_alloc<int64_t>   ldb(ctx.pool(), E);
    ggml_cuda_pool_alloc<int64_t>   ldd(ctx.pool(), E);
    mmid_i8_setup<<<(E + 127) / 128, 128, 0, stream>>>(bounds, E, N, K, A.get(), (const int8_t *) src0->data,
        src0->nb[2], D.get(), problems.get(), pa.get(), pb.get(), pd.get(), lda.get(), ldb.get(), ldd.get(), src0->nb[1]);
    CUDA_CHECK(cudaGetLastError());

    static int threadblock_count[GGML_CUDA_MAX_DEVICES] = {};
    if (threadblock_count[ctx.device] == 0) {
        threadblock_count[ctx.device] = C::Gemm::maximum_active_blocks() * ggml_cuda_info().devices[ctx.device].nsm;
    }
    typename C::Gemm::Arguments args(problems.get(), E, threadblock_count[ctx.device], {d_scale, 0.0f},
        pa.get(), pb.get(), pd.get(), pd.get(), lda.get(), ldb.get(), ldd.get(), ldd.get(), nullptr);
    typename C::Gemm gemm;
    const size_t ws_size = gemm.get_workspace_size(args);
    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), ws_size > 0 ? ws_size : 1);
    GGML_ASSERT(gemm.initialize(args, ws.get(), stream) == cutlass::Status::kSuccess);
    GGML_ASSERT(gemm.run(stream) == cutlass::Status::kSuccess);
    CUDA_CHECK(cudaGetLastError());

    const int threads = 128;
    switch (epilogue) {
        case MMID_I8_SCATTER: {
            const dim3 grid((unsigned) R, (N / 4 + threads - 1) / threads);
            mmid_i8_scatter<<<grid, threads, 0, stream>>>((const half *) D.get(), 1.0f / d_scale, sa.get(), w_scale, bounds, ids_fwd,
                (float *) out->data, E, N, out->nb[1]);
        } break;
        case MMID_I8_SWIGLU: {
            const dim3 grid((unsigned) R, (N / 8 + threads - 1) / threads);
            mmid_i8_swiglu<<<grid, threads, 0, stream>>>((const half *) D.get(), 1.0f / d_scale, sa.get(), w_scale, bounds, ids_fwd,
                (float *) out->data, E, N, out->nb[1]);
        } break;
        case MMID_I8_REDUCE: {
            const dim3 grid((unsigned) n_tokens, (N / 4 + threads - 1) / threads);
            mmid_i8_reduce<<<grid, threads, 0, stream>>>((const half *) D.get(), 1.0f / d_scale, sa.get(), w_scale, ids_inv, bounds, E,
                w_copy.get(), (float *) out->data, N, n_used, out->nb[1]);
        } break;
    }
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, mmid, w_scale, out, epilogue, weights, cgraph, node_idx);
    GGML_ABORT("the int8 GEMM needs a build with GGML_CUDA_CUTLASS_DIR");
#endif // GGML_CUDA_USE_CUTLASS
}
