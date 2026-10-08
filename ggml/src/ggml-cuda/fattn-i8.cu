#include "fattn-i8.cuh"
#include "convert.cuh"
#include "cp-async.cuh"

// Flash attention with int8 Q*K^T (SageAttention-style) for long prefill without a mask.
//
// Used when the graph allows int8 Q and K (ggml_prec_set_src(fa, GGML_PREC_Q8, 0/1)), for the head sizes of
// decompressed MLA (DKQ 192, DV 128), one K/V head per Q head, no mask, sinks, ALiBi or softcap, on Ampere or newer.
//   - K is smoothed (the mean key of each head is subtracted, which shifts every score of a row by the same amount and
//     leaves the softmax unchanged), then quantized to int8 with one scale per key;
//   - Q is multiplied by the softmax scale and log2(e) and quantized to int8 with one scale per query;
//   - S = Q K^T runs on int8 tensor cores (mma m16n8k32, int32 accumulation), then is rescaled per row and column;
//   - the online softmax is in the exp2 domain; P (f16) V (f16) runs on f16 tensor cores. Within a tile of 64 keys
//     it accumulates in f16 (GeForce cards run that at twice the f32-accumulate rate); the running output across
//     tiles stays f32. GGML_CUDA_FATTN_I8_ACC16=0 accumulates every tile in f32 instead.
// On a GeForce Ampere card the int8 path has 4x the tensor throughput of f16 with f32 accumulation.
// Allowing int8 Q and K (GGML_PREC_Q8 on src 0 and 1) is what opts an op into these approximations.

static constexpr int I8_DQK   = 192;
static constexpr int I8_DV    = 128;
#ifndef GGML_CUDA_FATTN_I8_WARPS
#define GGML_CUDA_FATTN_I8_WARPS 8
#endif
static constexpr int I8_WARPS = GGML_CUDA_FATTN_I8_WARPS;
static constexpr int I8_BQ    = 16 * I8_WARPS;  // queries per block (16 per warp)
static constexpr int I8_MIN_BLOCKS = 8 / I8_WARPS; // blocks per SM the register budget is sized for
static constexpr int I8_BK    = 64;   // keys per tile
static constexpr int I8_KSTR  = I8_DQK + 16;       // bytes per key row in shared memory (bank-conflict padding)
static constexpr int I8_VSTR  = (I8_DV + 8) * 2;   // bytes per value row in shared memory
static constexpr int I8_STAGE = I8_BK * I8_KSTR + I8_BK * I8_VSTR + I8_BK * (int) sizeof(float);
#ifndef GGML_CUDA_FATTN_I8_STAGES
#define GGML_CUDA_FATTN_I8_STAGES 2
#endif
static constexpr int I8_NSTAGES = GGML_CUDA_FATTN_I8_STAGES; // cp.async pipeline depth (K/V tiles in flight)
static constexpr int I8_SMEM  = I8_NSTAGES * I8_STAGE;

bool ggml_cuda_flash_attn_ext_i8_applies(int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (dst->src[3] != nullptr || dst->src[4] != nullptr) {
        return false; // mask or sinks
    }
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    const int32_t prec_q = ggml_get_op_params_i32(dst, 5);
    const int32_t prec_k = ggml_get_op_params_i32(dst, 6);
    const int cc = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE && cc >= GGML_CUDA_CC_AMPERE &&
        prec_q >= GGML_PREC_Q8 && prec_k >= GGML_PREC_Q8 && max_bias == 0.0f && softcap == 0.0f &&
        Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 &&
        Q->ne[0] == I8_DQK && K->ne[0] == I8_DQK && V->ne[0] == I8_DV &&
        Q->ne[2] == K->ne[2] && K->ne[2] == V->ne[2] && Q->ne[3] == 1 && K->ne[3] == 1 &&
        K->ne[1] == V->ne[1] && Q->ne[1] >= I8_BQ && K->ne[1] >= I8_BK &&
        Q->nb[0] == sizeof(float) && K->nb[0] == 2 && V->nb[0] == 2 && V->nb[1] % 16 == 0 && V->nb[2] % 16 == 0 &&
        ggml_cuda_info().devices[device].smpbo >= (size_t) I8_SMEM;
#endif
}

// sum of K over the keys, per head and dim: partial sums of 256 keys per block, then a fixed-order reduction (no
// atomics, so the result is the same on every run)
static constexpr int I8_KSUM_ROWS = 256;

static __global__ void fattn_i8_k_sum_partial(const char * __restrict__ K, float * __restrict__ part,
        const int64_t n_kv, const int64_t nbk1, const int64_t nbk2) {
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int64_t t0 = (int64_t) blockIdx.x * I8_KSUM_ROWS;
    const int64_t t1 = min(t0 + I8_KSUM_ROWS, n_kv);
    float s = 0.0f;
    for (int64_t t = t0; t < t1; t++) {
        s += __half2float(*(const half *) (K + h * nbk2 + t * nbk1 + d * 2));
    }
    part[((int64_t) blockIdx.x * gridDim.y + h) * I8_DQK + d] = s;
}

static __global__ void fattn_i8_k_sum_reduce(const float * __restrict__ part, float * __restrict__ ksum,
        const int n_part, const int H) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    float s = 0.0f;
    for (int i = 0; i < n_part; i++) {
        s += part[((int64_t) i * H + h) * I8_DQK + d];
    }
    ksum[h * I8_DQK + d] = s;
}

// one row (key or query) per warp: x = (src - mean) * mul, int8 with scale max|x| / 127
template <typename T>
static __global__ void fattn_i8_quantize(const char * __restrict__ src, const float * __restrict__ mean,
        int8_t * __restrict__ dst, float * __restrict__ scales, const int64_t n_rows, const int64_t nb1,
        const int64_t nb2, const float mul, const float inv_n_mean) {
    constexpr int per_lane = I8_DQK / WARP_SIZE; // 6
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int h = blockIdx.y;
    if (row >= n_rows) {
        return;
    }
    const char * s = src + h * nb2 + row * nb1;
    float x[per_lane];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < per_lane; i++) {
        const int d = lane * 2 + (i / 2) * 64 + (i % 2); // pairs: 2 consecutive dims per lane per 64-dim block
        float v = ggml_cuda_cast<float>(((const T *) s)[d]);
        if (mean != nullptr) {
            v -= mean[h * I8_DQK + d] * inv_n_mean;
        }
        v *= mul;
        x[i] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = warp_reduce_max(amax);
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    int8_t * o = dst + ((int64_t) h * n_rows + row) * I8_DQK;
#pragma unroll
    for (int i = 0; i < per_lane; i++) {
        const int d = lane * 2 + (i / 2) * 64 + (i % 2);
        o[d] = (int8_t) __float2int_rn(fminf(127.0f, fmaxf(-127.0f, x[i] * inv)));
    }
    if (lane == 0) {
        scales[(int64_t) h * n_rows + row] = scale;
    }
}

static __device__ __forceinline__ void i8_mma_s8(int (&c)[4], const int (&a)[4], const int b0, const int b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void i8_mma_f16(float (&c)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void i8_mma_f16_acc16(uint32_t (&c)[2], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void i8_ldmatrix_x4_trans(uint32_t (&r)[4], const void * smem_ptr) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned int addr = ggml_cuda_cvta_generic_to_shared((void *) smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
#else
    GGML_UNUSED_VARS(r, smem_ptr);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ float i8_i2f(const int x) {
    return __int_as_float(x + 0x4B400000) - 12582912.0f; // exact for |x| < 2^22
}

static __device__ __forceinline__ uint32_t i8_pack_half2(const float lo, const float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *) &h;
}

// acc16: P V of each tile accumulates in f16 (2x the f32-accumulate rate on GeForce cards), then joins the f32
// output
template <bool acc16>
static __global__ void __launch_bounds__(I8_WARPS * WARP_SIZE, I8_MIN_BLOCKS) fattn_i8_main(
        const int8_t * __restrict__ Q8, const float * __restrict__ sq_all,
        const int8_t * __restrict__ K8, const float * __restrict__ sk_all,
        const char * __restrict__ V, float * __restrict__ dst,
        const int64_t n_q, const int64_t n_kv, const int64_t nbv1, const int64_t nbv2,
        const int64_t nbd1, const int64_t nbd2) {
#ifdef AMPERE_MMA_AVAILABLE
    extern __shared__ __align__(16) char smem[];

    const int h    = blockIdx.y;
    const int q0   = blockIdx.x * I8_BQ;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int g    = lane / 4;
    const int tq   = lane % 4;

    const int8_t * Kh  = K8 + (int64_t) h * n_kv * I8_DQK;
    const float  * skh = sk_all + (int64_t) h * n_kv;
    const char   * Vh  = V + h * nbv2;

    // Q fragments for 16 rows x 192 dims: 6 k-steps of 32
    const int64_t r0 = q0 + warp * 16 + g;
    const int64_t r1 = r0 + 8;
    int qa[I8_DQK / 32][4];
    {
        const int8_t * Qh = Q8 + (int64_t) h * n_q * I8_DQK;
#pragma unroll
        for (int ks = 0; ks < I8_DQK / 32; ks++) {
            const int off = ks * 32 + tq * 4;
            qa[ks][0] = r0 < n_q ? *(const int *) (Qh + r0 * I8_DQK + off)      : 0;
            qa[ks][1] = r1 < n_q ? *(const int *) (Qh + r1 * I8_DQK + off)      : 0;
            qa[ks][2] = r0 < n_q ? *(const int *) (Qh + r0 * I8_DQK + off + 16) : 0;
            qa[ks][3] = r1 < n_q ? *(const int *) (Qh + r1 * I8_DQK + off + 16) : 0;
        }
    }
    const float sq0 = r0 < n_q ? sq_all[(int64_t) h * n_q + r0] : 0.0f;
    const float sq1 = r1 < n_q ? sq_all[(int64_t) h * n_q + r1] : 0.0f;

    float O[I8_DV / 8][4] = {};
    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;

    const int n_tiles = (int) ((n_kv + I8_BK - 1) / I8_BK);

    auto load_tile = [&](const int kt, const int stage) {
        char * sK  = smem + stage * I8_STAGE;
        char * sV  = sK + I8_BK * I8_KSTR;
        float * sS = (float *) (sV + I8_BK * I8_VSTR);
        const int64_t k0 = (int64_t) kt * I8_BK;
        // K: 64 rows x 12 chunks of 16 bytes
        for (int c = tid; c < I8_BK * (I8_DQK / 16); c += I8_WARPS * WARP_SIZE) {
            const int row = c / (I8_DQK / 16);
            const int col = c % (I8_DQK / 16);
            char * d = sK + row * I8_KSTR + col * 16;
            if (k0 + row < n_kv) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(d), Kh + (k0 + row) * I8_DQK + col * 16);
            } else {
                *(int4 *) d = make_int4(0, 0, 0, 0);
            }
        }
        // V: 64 rows x 16 chunks of 16 bytes
        for (int c = tid; c < I8_BK * (I8_DV * 2 / 16); c += I8_WARPS * WARP_SIZE) {
            const int row = c / (I8_DV * 2 / 16);
            const int col = c % (I8_DV * 2 / 16);
            char * d = sV + row * I8_VSTR + col * 16;
            if (k0 + row < n_kv) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(d), Vh + (k0 + row) * nbv1 + col * 16);
            } else {
                *(int4 *) d = make_int4(0, 0, 0, 0);
            }
        }
        if (tid < I8_BK) {
            sS[tid] = k0 + tid < n_kv ? skh[k0 + tid] : 0.0f;
        }
    };

    // prologue: tiles 0 .. NSTAGES-2 in flight, one commit group each
#pragma unroll
    for (int st = 0; st < I8_NSTAGES - 1; st++) {
        if (st < n_tiles) {
            load_tile(st, st);
        }
        asm volatile("cp.async.commit_group;");
    }

    for (int kt = 0; kt < n_tiles; kt++) {
        const int stage = kt % I8_NSTAGES;
        // the buffer of tile kt + NSTAGES - 1 was last read in iteration kt - 1, which ended with a barrier
        if (kt + I8_NSTAGES - 1 < n_tiles) {
            load_tile(kt + I8_NSTAGES - 1, (kt + I8_NSTAGES - 1) % I8_NSTAGES);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" : : "n"(I8_NSTAGES - 1));
        __syncthreads();

        const char  * sK = smem + stage * I8_STAGE;
        const char  * sV = sK + I8_BK * I8_KSTR;
        const float * sS = (const float *) (sV + I8_BK * I8_VSTR);
        const int64_t k0 = (int64_t) kt * I8_BK;

        // S = Q K^T: 8 n-tiles of 8 keys
        int c[I8_BK / 8][4] = {};
#pragma unroll
        for (int ks = 0; ks < I8_DQK / 32; ks++) {
#pragma unroll
            for (int nt = 0; nt < I8_BK / 8; nt++) {
                const char * kr = sK + (nt * 8 + g) * I8_KSTR + ks * 32 + tq * 4;
                i8_mma_s8(c[nt], qa[ks], *(const int *) kr, *(const int *) (kr + 16));
            }
        }

        // rescale, mask keys past the end, online softmax (exp2 domain). The scores are kept in units of the query
        // scale (s' = c * sk, s = s' * sq with sq > 0), which moves sq into the exponent's FMA. int32 -> float by an
        // integer add and a float subtraction (exact: |c| <= 192 * 127 * 127 < 2^22) instead of the quarter-rate I2F.
        float s[I8_BK / 8][4];
        float mt0 = -INFINITY, mt1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < I8_BK / 8; nt++) {
            const int j = nt * 8 + 2 * tq;
            const float ska = sS[j];
            const float skb = sS[j + 1];
            const bool va = k0 + j < n_kv;
            const bool vb = k0 + j + 1 < n_kv;
            s[nt][0] = va ? i8_i2f(c[nt][0]) * ska : -INFINITY;
            s[nt][1] = vb ? i8_i2f(c[nt][1]) * skb : -INFINITY;
            s[nt][2] = va ? i8_i2f(c[nt][2]) * ska : -INFINITY;
            s[nt][3] = vb ? i8_i2f(c[nt][3]) * skb : -INFINITY;
            mt0 = fmaxf(mt0, fmaxf(s[nt][0], s[nt][1]));
            mt1 = fmaxf(mt1, fmaxf(s[nt][2], s[nt][3]));
        }
        mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFF, mt0, 1));
        mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFF, mt0, 2));
        mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFF, mt1, 1));
        mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFF, mt1, 2));
        const float mn0 = fmaxf(m0, mt0 * sq0);
        const float mn1 = fmaxf(m1, mt1 * sq1);
        const float a0 = exp2f(m0 - mn0);
        const float a1 = exp2f(m1 - mn1);
        m0 = mn0;
        m1 = mn1;
        l0 *= a0;
        l1 *= a1;
        // the running maxima rarely change after the first tiles: skip the rescale when no row of the warp moved
        if (__any_sync(0xFFFFFFFF, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int nt = 0; nt < I8_DV / 8; nt++) {
                O[nt][0] *= a0; O[nt][1] *= a0;
                O[nt][2] *= a1; O[nt][3] *= a1;
            }
        }
        uint32_t p[I8_BK / 8][2];
#pragma unroll
        for (int nt = 0; nt < I8_BK / 8; nt++) {
            const float p0 = exp2f(fmaf(s[nt][0], sq0, -mn0));
            const float p1 = exp2f(fmaf(s[nt][1], sq0, -mn0));
            const float p2 = exp2f(fmaf(s[nt][2], sq1, -mn1));
            const float p3 = exp2f(fmaf(s[nt][3], sq1, -mn1));
            l0 += p0 + p1;
            l1 += p2 + p3;
            p[nt][0] = i8_pack_half2(p0, p1);
            p[nt][1] = i8_pack_half2(p2, p3);
        }

        // O += P V: 4 k-steps of 16 keys, 16 n-tiles of 8 dims (two per ldmatrix.x4)
        if constexpr (acc16) {
            uint32_t O16[I8_DV / 8][2] = {};
#pragma unroll
            for (int kk = 0; kk < I8_BK / 16; kk++) {
                const uint32_t pa[4] = { p[2 * kk][0], p[2 * kk][1], p[2 * kk + 1][0], p[2 * kk + 1][1] };
                const int key = kk * 16 + ((lane / 8) % 2) * 8 + lane % 8;
#pragma unroll
                for (int np = 0; np < I8_DV / 16; np++) {
                    const int dim = np * 16 + (lane / 16) * 8;
                    uint32_t b[4];
                    i8_ldmatrix_x4_trans(b, sV + key * I8_VSTR + dim * 2);
                    i8_mma_f16_acc16(O16[2 * np],     pa, b[0], b[1]);
                    i8_mma_f16_acc16(O16[2 * np + 1], pa, b[2], b[3]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < I8_DV / 8; nt++) {
                const float2 lo = __half22float2(*(const half2 *) &O16[nt][0]);
                const float2 hi = __half22float2(*(const half2 *) &O16[nt][1]);
                O[nt][0] += lo.x; O[nt][1] += lo.y;
                O[nt][2] += hi.x; O[nt][3] += hi.y;
            }
        } else {
#pragma unroll
            for (int kk = 0; kk < I8_BK / 16; kk++) {
                const uint32_t pa[4] = { p[2 * kk][0], p[2 * kk][1], p[2 * kk + 1][0], p[2 * kk + 1][1] };
                const int key = kk * 16 + ((lane / 8) % 2) * 8 + lane % 8;
#pragma unroll
                for (int np = 0; np < I8_DV / 16; np++) {
                    const int dim = np * 16 + (lane / 16) * 8;
                    uint32_t b[4];
                    i8_ldmatrix_x4_trans(b, sV + key * I8_VSTR + dim * 2);
                    i8_mma_f16(O[2 * np],     pa, b[0], b[1]);
                    i8_mma_f16(O[2 * np + 1], pa, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;");

    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 1);
    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 2);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 1);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 2);
    const float il0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float il1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;

#pragma unroll
    for (int nt = 0; nt < I8_DV / 8; nt++) {
        const int d = nt * 8 + 2 * tq;
        if (r0 < n_q) {
            float2 * o = (float2 *) ((char *) dst + r0 * nbd2 + h * nbd1 + d * sizeof(float));
            *o = make_float2(O[nt][0] * il0, O[nt][1] * il0);
        }
        if (r1 < n_q) {
            float2 * o = (float2 *) ((char *) dst + r1 * nbd2 + h * nbd1 + d * sizeof(float));
            *o = make_float2(O[nt][2] * il1, O[nt][3] * il1);
        }
    }
#else
    GGML_UNUSED_VARS(Q8, sq_all, K8, sk_all, V, dst, n_q, n_kv, nbv1, nbv2, nbd1, nbd2);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------------------------------
// int8 P*V (opt-in: GGML_PREC_Q8 on src 2, V). V is quantized per (head, dim) channel over all keys and stored
// transposed, [dim][key], with the keys of every 16-key block permuted so that the P values a thread already holds
// from the Q*K^T fragments form the A operand of mma m16n8k32 u8*s8 directly. P is quantized per row and key tile
// relative to the tile's row maximum (exp2(tile max - running max) <= 1), so tiles of small weights keep their
// relative precision. The V channel scales are applied once, to the final output.

static constexpr int I8V_VSTR  = I8_BK + 16;                      // bytes per dim row of the V8 tile
static constexpr int I8V_STAGE = I8_BK * I8_KSTR + I8_DV * I8V_VSTR + I8_BK * (int) sizeof(float);
static constexpr int I8V_SMEM  = 2 * I8V_STAGE;

// logical position L of a key inside its 16-key block -> actual key: L = tq*4 + b  <-  2*tq + (b & 1) + 8*(b >> 1)
static __device__ __forceinline__ int i8v_actual_key(const int L) {
    const int tq = L / 4, b = L % 4;
    return 2 * tq + (b & 1) + 8 * (b >> 1);
}

// partial channel maxima of |V| over blocks of 256 keys
static __global__ void fattn_i8v_amax_partial(const char * __restrict__ V, float * __restrict__ part,
        const int64_t n_kv, const int64_t nbv1, const int64_t nbv2) {
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int64_t t0 = (int64_t) blockIdx.x * I8_KSUM_ROWS;
    const int64_t t1 = min(t0 + I8_KSUM_ROWS, n_kv);
    float m = 0.0f;
    for (int64_t t = t0; t < t1; t++) {
        m = fmaxf(m, fabsf(__half2float(*(const half *) (V + h * nbv2 + t * nbv1 + d * 2))));
    }
    part[((int64_t) blockIdx.x * gridDim.y + h) * I8_DV + d] = m;
}

static __global__ void fattn_i8v_amax_reduce(const float * __restrict__ part, float * __restrict__ sv, const int n_part, const int H) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    float m = 0.0f;
    for (int i = 0; i < n_part; i++) {
        m = fmaxf(m, part[((int64_t) i * H + h) * I8_DV + d]);
    }
    sv[h * I8_DV + d] = m > 0.0f ? m / 127.0f : 1.0f;
}

// V8[h][d][key'] (key' = permuted key, n_kv padded to a multiple of 64 with zeros): one block per (64 keys, head)
static __global__ void fattn_i8v_quantize(const char * __restrict__ V, const float * __restrict__ sv,
        int8_t * __restrict__ V8, const int64_t n_kv, const int64_t n_kv_pad, const int64_t nbv1, const int64_t nbv2) {
    __shared__ int8_t tile[I8_BK][I8_DV + 4];
    const int h = blockIdx.y;
    const int64_t k0 = (int64_t) blockIdx.x * I8_BK;
    for (int i = threadIdx.x; i < I8_BK * I8_DV; i += blockDim.x) {
        const int k = i / I8_DV;
        const int d = i % I8_DV;
        float v = 0.0f;
        if (k0 + k < n_kv) {
            v = __half2float(*(const half *) (V + h * nbv2 + (k0 + k) * nbv1 + d * 2));
        }
        tile[k][d] = (int8_t) __float2int_rn(fminf(127.0f, fmaxf(-127.0f, v / sv[h * I8_DV + d])));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < I8_BK * I8_DV; i += blockDim.x) {
        const int d = i / I8_BK;
        const int L = i % I8_BK;                       // logical key position in the tile
        const int k = (L / 16) * 16 + i8v_actual_key(L % 16);
        V8[((int64_t) h * I8_DV + d) * n_kv_pad + k0 + L] = tile[k][d];
    }
}

static __device__ __forceinline__ void i8_mma_u8s8(int (&c)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ uint32_t i8v_pack_u8(const float a, const float b, const float c, const float d) {
    return (uint32_t) __float2uint_rn(a) | ((uint32_t) __float2uint_rn(b) << 8) |
           ((uint32_t) __float2uint_rn(c) << 16) | ((uint32_t) __float2uint_rn(d) << 24);
}

static __global__ void __launch_bounds__(I8_WARPS * WARP_SIZE, I8_MIN_BLOCKS) fattn_i8v_main(
        const int8_t * __restrict__ Q8, const float * __restrict__ sq_all,
        const int8_t * __restrict__ K8, const float * __restrict__ sk_all,
        const int8_t * __restrict__ V8, const float * __restrict__ sv_all, float * __restrict__ dst,
        const int64_t n_q, const int64_t n_kv, const int64_t n_kv_pad, const int64_t nbd1, const int64_t nbd2) {
#ifdef AMPERE_MMA_AVAILABLE
    extern __shared__ __align__(16) char smem[];

    const int h    = blockIdx.y;
    const int q0   = blockIdx.x * I8_BQ;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int g    = lane / 4;
    const int tq   = lane % 4;

    const int8_t * Kh  = K8 + (int64_t) h * n_kv * I8_DQK;
    const float  * skh = sk_all + (int64_t) h * n_kv;
    const int8_t * Vh  = V8 + (int64_t) h * I8_DV * n_kv_pad;

    const int64_t r0 = q0 + warp * 16 + g;
    const int64_t r1 = r0 + 8;
    int qa[I8_DQK / 32][4];
    {
        const int8_t * Qh = Q8 + (int64_t) h * n_q * I8_DQK;
#pragma unroll
        for (int ks = 0; ks < I8_DQK / 32; ks++) {
            const int off = ks * 32 + tq * 4;
            qa[ks][0] = r0 < n_q ? *(const int *) (Qh + r0 * I8_DQK + off)      : 0;
            qa[ks][1] = r1 < n_q ? *(const int *) (Qh + r1 * I8_DQK + off)      : 0;
            qa[ks][2] = r0 < n_q ? *(const int *) (Qh + r0 * I8_DQK + off + 16) : 0;
            qa[ks][3] = r1 < n_q ? *(const int *) (Qh + r1 * I8_DQK + off + 16) : 0;
        }
    }
    const float sq0 = r0 < n_q ? sq_all[(int64_t) h * n_q + r0] : 0.0f;
    const float sq1 = r1 < n_q ? sq_all[(int64_t) h * n_q + r1] : 0.0f;

    float O[I8_DV / 8][4] = {};
    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;

    const int n_tiles = (int) ((n_kv + I8_BK - 1) / I8_BK);

    auto load_tile = [&](const int kt, const int stage) {
        char * sK  = smem + stage * I8V_STAGE;
        char * sV  = sK + I8_BK * I8_KSTR;
        float * sS = (float *) (sV + I8_DV * I8V_VSTR);
        const int64_t k0 = (int64_t) kt * I8_BK;
        for (int c = tid; c < I8_BK * (I8_DQK / 16); c += I8_WARPS * WARP_SIZE) {
            const int row = c / (I8_DQK / 16);
            const int col = c % (I8_DQK / 16);
            char * d = sK + row * I8_KSTR + col * 16;
            if (k0 + row < n_kv) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(d), Kh + (k0 + row) * I8_DQK + col * 16);
            } else {
                *(int4 *) d = make_int4(0, 0, 0, 0);
            }
        }
        // V8: 128 dim rows x 4 chunks of 16 keys (padded keys are zero in global memory)
        for (int c = tid; c < I8_DV * (I8_BK / 16); c += I8_WARPS * WARP_SIZE) {
            const int row = c / (I8_BK / 16);
            const int col = c % (I8_BK / 16);
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(sV + row * I8V_VSTR + col * 16), Vh + row * n_kv_pad + k0 + col * 16);
        }
        if (tid < I8_BK) {
            sS[tid] = k0 + tid < n_kv ? skh[k0 + tid] : 0.0f;
        }
    };

    load_tile(0, 0);
    asm volatile("cp.async.commit_group;");

    for (int kt = 0; kt < n_tiles; kt++) {
        const int stage = kt % 2;
        if (kt + 1 < n_tiles) {
            load_tile(kt + 1, stage ^ 1);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        const char  * sK = smem + stage * I8V_STAGE;
        const char  * sV = sK + I8_BK * I8_KSTR;
        const float * sS = (const float *) (sV + I8_DV * I8V_VSTR);
        const int64_t k0 = (int64_t) kt * I8_BK;

        int c[I8_BK / 8][4] = {};
#pragma unroll
        for (int ks = 0; ks < I8_DQK / 32; ks++) {
#pragma unroll
            for (int nt = 0; nt < I8_BK / 8; nt++) {
                const char * kr = sK + (nt * 8 + g) * I8_KSTR + ks * 32 + tq * 4;
                i8_mma_s8(c[nt], qa[ks], *(const int *) kr, *(const int *) (kr + 16));
            }
        }

        float s[I8_BK / 8][4];
        float mt0 = -INFINITY, mt1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < I8_BK / 8; nt++) {
            const int j = nt * 8 + 2 * tq;
            const float ska = sS[j];
            const float skb = sS[j + 1];
            const bool va = k0 + j < n_kv;
            const bool vb = k0 + j + 1 < n_kv;
            s[nt][0] = va ? i8_i2f(c[nt][0]) * ska : -INFINITY;
            s[nt][1] = vb ? i8_i2f(c[nt][1]) * skb : -INFINITY;
            s[nt][2] = va ? i8_i2f(c[nt][2]) * ska : -INFINITY;
            s[nt][3] = vb ? i8_i2f(c[nt][3]) * skb : -INFINITY;
            mt0 = fmaxf(mt0, fmaxf(s[nt][0], s[nt][1]));
            mt1 = fmaxf(mt1, fmaxf(s[nt][2], s[nt][3]));
        }
        mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFF, mt0, 1));
        mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFF, mt0, 2));
        mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFF, mt1, 1));
        mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFF, mt1, 2));
        const float mn0 = fmaxf(m0, mt0 * sq0);
        const float mn1 = fmaxf(m1, mt1 * sq1);
        const float a0 = exp2f(m0 - mn0);
        const float a1 = exp2f(m1 - mn1);
        m0 = mn0;
        m1 = mn1;
        l0 *= a0;
        l1 *= a1;
        if (__any_sync(0xFFFFFFFF, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int nt = 0; nt < I8_DV / 8; nt++) {
                O[nt][0] *= a0; O[nt][1] *= a0;
                O[nt][2] *= a1; O[nt][3] *= a1;
            }
        }
        // P relative to the tile's row maximum: u8 = round(p * 255 / pmax), pmax = exp2(tile max - running max)
        const float pm0 = exp2f(fmaf(mt0, sq0, -mn0));
        const float pm1 = exp2f(fmaf(mt1, sq1, -mn1));
        const float u0  = pm0 > 0.0f ? 255.0f / pm0 : 0.0f;
        const float u1  = pm1 > 0.0f ? 255.0f / pm1 : 0.0f;
        float pf[I8_BK / 8][4];
#pragma unroll
        for (int nt = 0; nt < I8_BK / 8; nt++) {
            pf[nt][0] = exp2f(fmaf(s[nt][0], sq0, -mn0));
            pf[nt][1] = exp2f(fmaf(s[nt][1], sq0, -mn0));
            pf[nt][2] = exp2f(fmaf(s[nt][2], sq1, -mn1));
            pf[nt][3] = exp2f(fmaf(s[nt][3], sq1, -mn1));
            l0 += pf[nt][0] + pf[nt][1];
            l1 += pf[nt][2] + pf[nt][3];
        }
        // A fragments for the two k-steps of 32 keys: bytes = keys (2tq, 2tq+1) of n-tile 2j and of n-tile 2j+1
        uint32_t pa[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int n0 = 4 * kk;
            pa[kk][0] = i8v_pack_u8(pf[n0][0] * u0, pf[n0][1] * u0, pf[n0 + 1][0] * u0, pf[n0 + 1][1] * u0);
            pa[kk][1] = i8v_pack_u8(pf[n0][2] * u1, pf[n0][3] * u1, pf[n0 + 1][2] * u1, pf[n0 + 1][3] * u1);
            pa[kk][2] = i8v_pack_u8(pf[n0 + 2][0] * u0, pf[n0 + 2][1] * u0, pf[n0 + 3][0] * u0, pf[n0 + 3][1] * u0);
            pa[kk][3] = i8v_pack_u8(pf[n0 + 2][2] * u1, pf[n0 + 2][3] * u1, pf[n0 + 3][2] * u1, pf[n0 + 3][3] * u1);
        }
        const float f0 = pm0 / 255.0f;
        const float f1 = pm1 / 255.0f;
        // O += P V in two halves of 64 dims (8 n-tiles each), int32 per tile
#pragma unroll
        for (int half = 0; half < 2; half++) {
            int acc[8][4] = {};
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    const char * vr = sV + (half * 64 + nt * 8 + g) * I8V_VSTR + kk * 32 + tq * 4;
                    i8_mma_u8s8(acc[nt], pa[kk], *(const uint32_t *) vr, *(const uint32_t *) (vr + 16));
                }
            }
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                O[half * 8 + nt][0] += i8_i2f(acc[nt][0]) * f0;
                O[half * 8 + nt][1] += i8_i2f(acc[nt][1]) * f0;
                O[half * 8 + nt][2] += i8_i2f(acc[nt][2]) * f1;
                O[half * 8 + nt][3] += i8_i2f(acc[nt][3]) * f1;
            }
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;");

    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 1);
    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 2);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 1);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 2);
    const float il0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float il1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const float * svh = sv_all + h * I8_DV;

#pragma unroll
    for (int nt = 0; nt < I8_DV / 8; nt++) {
        const int d = nt * 8 + 2 * tq;
        const float v0 = svh[d], v1 = svh[d + 1];
        if (r0 < n_q) {
            float2 * o = (float2 *) ((char *) dst + r0 * nbd2 + h * nbd1 + d * sizeof(float));
            *o = make_float2(O[nt][0] * il0 * v0, O[nt][1] * il0 * v1);
        }
        if (r1 < n_q) {
            float2 * o = (float2 *) ((char *) dst + r1 * nbd2 + h * nbd1 + d * sizeof(float));
            *o = make_float2(O[nt][2] * il1 * v0, O[nt][3] * il1 * v1);
        }
    }
#else
    GGML_UNUSED_VARS(Q8, sq_all, K8, sk_all, V8, sv_all, dst, n_q, n_kv, n_kv_pad, nbd1, nbd2);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

void ggml_cuda_flash_attn_ext_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    cudaStream_t stream = ctx.stream();

    const int64_t n_q  = Q->ne[1];
    const int64_t n_kv = K->ne[1];
    const int64_t H    = Q->ne[2];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_part = (int) ((n_kv + I8_KSUM_ROWS - 1) / I8_KSUM_ROWS);
    ggml_cuda_pool_alloc<float>  kpart(ctx.pool(), (int64_t) n_part * H * I8_DQK);
    ggml_cuda_pool_alloc<float>  ksum(ctx.pool(), H * I8_DQK);
    ggml_cuda_pool_alloc<int8_t> K8(ctx.pool(), H * n_kv * I8_DQK);
    ggml_cuda_pool_alloc<float>  sk(ctx.pool(), H * n_kv);
    ggml_cuda_pool_alloc<int8_t> Q8(ctx.pool(), H * n_q * I8_DQK);
    ggml_cuda_pool_alloc<float>  sq(ctx.pool(), H * n_q);

    fattn_i8_k_sum_partial<<<dim3(n_part, H), I8_DQK, 0, stream>>>(
        (const char *) K->data, kpart.get(), n_kv, K->nb[1], K->nb[2]);
    fattn_i8_k_sum_reduce<<<H, I8_DQK, 0, stream>>>(kpart.get(), ksum.get(), n_part, (int) H);
    constexpr int rows_per_block = 8;
    fattn_i8_quantize<half><<<dim3((n_kv + rows_per_block - 1) / rows_per_block, H), rows_per_block * WARP_SIZE, 0, stream>>>(
        (const char *) K->data, ksum.get(), K8.get(), sk.get(), n_kv, K->nb[1], K->nb[2], 1.0f, 1.0f / (float) n_kv);
    // Q * scale * log2(e): the softmax runs on exp2
    fattn_i8_quantize<float><<<dim3((n_q + rows_per_block - 1) / rows_per_block, H), rows_per_block * WARP_SIZE, 0, stream>>>(
        (const char *) Q->data, nullptr, Q8.get(), sq.get(), n_q, Q->nb[1], Q->nb[2], scale * 1.4426950408889634f, 0.0f);

    // int8 P*V when the op allows int8 V (GGML_PREC_Q8 on src 2)
    if (ggml_get_op_params_i32(dst, 7) >= GGML_PREC_Q8) {
        const int64_t n_kv_pad = (n_kv + I8_BK - 1) / I8_BK * I8_BK;
        ggml_cuda_pool_alloc<float>  vpart(ctx.pool(), (int64_t) n_part * H * I8_DV);
        ggml_cuda_pool_alloc<float>  sv(ctx.pool(), H * I8_DV);
        ggml_cuda_pool_alloc<int8_t> V8(ctx.pool(), H * I8_DV * n_kv_pad);
        fattn_i8v_amax_partial<<<dim3(n_part, H), I8_DV, 0, stream>>>((const char *) V->data, vpart.get(), n_kv, V->nb[1], V->nb[2]);
        fattn_i8v_amax_reduce<<<H, I8_DV, 0, stream>>>(vpart.get(), sv.get(), n_part, (int) H);
        fattn_i8v_quantize<<<dim3(n_kv_pad / I8_BK, H), 256, 0, stream>>>((const char *) V->data, sv.get(), V8.get(), n_kv, n_kv_pad,
            V->nb[1], V->nb[2]);
        static bool smem_v_set[GGML_CUDA_MAX_DEVICES] = {};
        const int dev = ggml_cuda_get_device();
        if (!smem_v_set[dev]) {
            CUDA_CHECK(cudaFuncSetAttribute(fattn_i8v_main, cudaFuncAttributeMaxDynamicSharedMemorySize, I8V_SMEM));
            smem_v_set[dev] = true;
        }
        fattn_i8v_main<<<dim3((n_q + I8_BQ - 1) / I8_BQ, H), I8_WARPS * WARP_SIZE, I8V_SMEM, stream>>>(
            Q8.get(), sq.get(), K8.get(), sk.get(), V8.get(), sv.get(), (float *) dst->data,
            n_q, n_kv, n_kv_pad, dst->nb[1], dst->nb[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // f16 accumulation of P V within each key tile (see the top of the file)
    const char * acc16_env = getenv("GGML_CUDA_FATTN_I8_ACC16");
    const bool acc16 = acc16_env == nullptr || atoi(acc16_env) != 0;

    static bool smem_set[GGML_CUDA_MAX_DEVICES] = {};
    const int device = ggml_cuda_get_device();
    if (!smem_set[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(fattn_i8_main<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, I8_SMEM));
        CUDA_CHECK(cudaFuncSetAttribute(fattn_i8_main<true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, I8_SMEM));
        smem_set[device] = true;
    }
    const dim3 grid((n_q + I8_BQ - 1) / I8_BQ, H);
    if (acc16) {
        fattn_i8_main<true><<<grid, I8_WARPS * WARP_SIZE, I8_SMEM, stream>>>(
            Q8.get(), sq.get(), K8.get(), sk.get(), (const char *) V->data, (float *) dst->data,
            n_q, n_kv, V->nb[1], V->nb[2], dst->nb[1], dst->nb[2]);
    } else {
        fattn_i8_main<false><<<grid, I8_WARPS * WARP_SIZE, I8_SMEM, stream>>>(
            Q8.get(), sq.get(), K8.get(), sk.get(), (const char *) V->data, (float *) dst->data,
            n_q, n_kv, V->nb[1], V->nb[2], dst->nb[1], dst->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());
}

