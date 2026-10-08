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

    // per-token int8 activations, reused from the previous I8 MUL_MAT on the same tensor when still valid
    auto & cache = ctx.i8_act;
    const bool hit = cgraph != nullptr && cache.graph == cgraph && cache.src == src1 && node_idx <= cache.valid_until;
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
        cache.src         = cgraph != nullptr ? src1 : nullptr;
        cache.valid_until = cgraph != nullptr ? mmi8_valid_until(cgraph, node_idx, src1) : -1;
    }
    int8_t * xq = cache.q;
    float  * xs = cache.s;

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
