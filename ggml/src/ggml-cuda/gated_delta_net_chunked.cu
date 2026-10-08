#include "gated_delta_net_chunked.cuh"
#include "cp-async.cuh"

#include <cstdlib>

// Chunked prefill for GGML_OP_GATED_DELTA_NET (KDA per-channel gates, or one scalar gate per head).
//
// The per-token recurrence (gated_delta_net.cu)
//     S_t = diag(exp(g_t)) S_{t-1};  S_t += beta_t k_t (v_t - S_t^T k_t)^T;  o_t = scale * S_t^T q_t
// is evaluated in chunks of BT tokens with the WY representation (flash-linear-attention, chunk_kda):
//     G      = cumsum of g inside the chunk (per channel)
//     N[c,i] = -beta_c sum_d k_c k_i exp(G_c - G_i)          (c > i)
//     M      = (I - N)^-1 diag(beta)
//     w      = M (k * exp(G)),   u = M v
//     A[c,j] = scale sum_d q_c k_j exp(G_c - G_j)            (j <= c)
//   per chunk, in order:
//     v' = u - w S;   o = scale (q * exp(G)) S + A v';   S = exp(G_last) * S + (k * exp(G_last - G))^T v'
//
// Kernel 1 (gdn_chunk_intra) computes w, u, A and the gated q/k of every chunk in parallel. Kernel 2
// (gdn_chunk_inter) runs the chunks in order, one block per (head, group of 32 value columns), and prefetches the next
// chunk's operands with cp.async while it works on the current one. exp(G_c - G_i) is evaluated per term (never
// exp(-G) alone), so any gate magnitude is safe. Everything is f32.
//
// The sequence is processed in segments of SEG_CHUNKS chunks so the temporary buffer stays small.

static constexpr int GDN_D          = 128; // head size (S_v == S_k)
static constexpr int GDN_BT         = 32;  // tokens per chunk
static constexpr int GDN_NV         = 32;  // value columns per block of the sequential kernel
static constexpr int GDN_SEG_CHUNKS = 128; // chunks per segment (4096 tokens)
static constexpr int GDN_THREADS    = 256;
static constexpr int GDN_TP         = GDN_BT + 1; // padded token stride of the [D][BT] shared arrays

// per (seq, head, chunk) in the temporary buffer; every [D][BT] array is transposed (token fastest)
struct gdn_layout {
    static constexpr int WT     = 0;                          // w^T               [D][BT]
    static constexpr int QGT    = WT  + GDN_D * GDN_BT;       // (scale q e^G)^T   [D][BT]
    static constexpr int KLT    = QGT + GDN_D * GDN_BT;       // (k e^(Gl-G))^T    [D][BT]
    static constexpr int UT     = KLT + GDN_D * GDN_BT;       // u^T               [D][BT]
    static constexpr int A      = UT  + GDN_D * GDN_BT;       // A                 [BT][BT]
    static constexpr int GL     = A   + GDN_BT * GDN_BT;      // exp(G_last)       [D]
    static constexpr int stride = GL  + GDN_D;
};
static_assert(gdn_layout::stride % 4 == 0, "16-byte aligned chunks");

struct gdn_src {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    bool kda;
    int H;
    int H_k;
    int64_t n_tokens;
    int64_t rq3;
    int64_t sq1, sq2, sq3;
    int64_t sv1, sv2, sv3;
    int64_t sb1, sb2, sb3;
    float scale;
};

static constexpr size_t GDN_INTRA_SMEM = (4 * GDN_D * GDN_TP + GDN_BT * GDN_TP + GDN_BT) * sizeof(float);

__global__ void __launch_bounds__(GDN_THREADS) gdn_chunk_intra(const gdn_src src, float * __restrict__ tmp,
        const int64_t t0, const int n_chunks_seg) {
    constexpr int D  = GDN_D;
    constexpr int BT = GDN_BT;
    constexpr int TP = GDN_TP;

    extern __shared__ float smem[];
    float * qT    = smem;                // [D][TP], later k e^G
    float * kT    = qT + D * TP;         // [D][TP]
    float * gT    = kT + D * TP;         // [D][TP]: g, then its cumsum G
    float * vT    = gT + D * TP;         // [D][TP]
    float * sM    = vT + D * TP;         // [BT][TP]: N -> T -> M
    float * sBeta = sM + BT * TP;        // [BT]

    const int c   = blockIdx.x;
    const int h   = blockIdx.y;
    const int seq = blockIdx.z;
    const int tid = threadIdx.x;

    const int64_t tc0 = t0 + (int64_t) c * BT;
    const int nvalid = (int) min((int64_t) BT, src.n_tokens - tc0);
    const int hk = h % src.H_k;
    const int64_t sk = seq / src.rq3;

    float * out = tmp + ((int64_t) (seq * src.H + h) * n_chunks_seg + c) * gdn_layout::stride;

    // ---- load q, k, v, g of the chunk (coalesced over d), transposed into [D][TP]
    for (int idx = tid; idx < BT * D; idx += GDN_THREADS) {
        const int t = idx / D;
        const int d = idx % D;
        float qv = 0.0f, kv = 0.0f, vv = 0.0f, gv = 0.0f;
        if (t < nvalid) {
            const int64_t tok = tc0 + t;
            const int64_t qo  = sk * src.sq3 + tok * src.sq2 + hk * src.sq1 + d;
            qv = src.q[qo];
            kv = src.k[qo];
            vv = src.v[seq * src.sv3 + tok * src.sv2 + h * src.sv1 + d];
            const int64_t gb = seq * src.sb3 + tok * src.sb2 + h * src.sb1;
            gv = src.kda ? src.g[gb * D + d] : src.g[gb];
        }
        qT[d * TP + t] = qv;
        kT[d * TP + t] = kv;
        vT[d * TP + t] = vv;
        gT[d * TP + t] = gv;
    }
    if (tid < BT) {
        sBeta[tid] = tid < nvalid ? src.beta[seq * src.sb3 + (tc0 + tid) * src.sb2 + h * src.sb1] : 0.0f;
    }
    __syncthreads();

    // ---- gate cumsum per dim
    if (tid < D) {
        float G = 0.0f;
        for (int t = 0; t < BT; t++) {
            G += gT[tid * TP + t];
            gT[tid * TP + t] = G;
        }
        out[gdn_layout::GL + tid] = expf(G);
    }
    __syncthreads();

    // ---- A[r][s] (s <= r) and N[r][s] (s < r): 2x2 tiles, 16x16 threads
    {
        const int ti = tid / 16;
        const int tj = tid % 16;
        const int r0 = ti * 2;
        const int s0 = tj * 2;
        const bool active = tj <= ti;
        float accA[2][2] = {};
        float accN[2][2] = {};
        if (active) {
#pragma unroll 4
            for (int d = 0; d < D; d++) {
                const float * q  = qT + d * TP;
                const float * kk = kT + d * TP;
                const float * G  = gT + d * TP;
                const float qa[2] = {q[r0], q[r0 + 1]};
                const float ka[2] = {kk[r0], kk[r0 + 1]};
                const float kb[2] = {kk[s0], kk[s0 + 1]};
                const float ga[2] = {G[r0], G[r0 + 1]};
                const float gb[2] = {G[s0], G[s0 + 1]};
#pragma unroll
                for (int a = 0; a < 2; a++) {
#pragma unroll
                    for (int b = 0; b < 2; b++) {
                        if (r0 + a >= s0 + b) {
                            const float e = expf(ga[a] - gb[b]);
                            accA[a][b] += qa[a] * kb[b] * e;
                            accN[a][b] += ka[a] * kb[b] * e;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int a = 0; a < 2; a++) {
#pragma unroll
            for (int b = 0; b < 2; b++) {
                const int r = r0 + a;
                const int s = s0 + b;
                out[gdn_layout::A + r * BT + s] = (active && s <= r) ? src.scale * accA[a][b] : 0.0f;
                sM[r * TP + s] = (active && s < r) ? -sBeta[r] * accN[a][b] : 0.0f;
            }
        }
    }
    __syncthreads();

    // ---- T = (I - N)^-1 - I by forward substitution, in place, row by row:
    //      T[i][j] = N[i][j] + sum_{j<m<i} N[i][m] T[m][j]   (rows 0 and 1 are already final)
    for (int i = 2; i < BT; i++) {
        const int j = tid;
        float val = 0.0f;
        if (j < i - 1) {
            val = sM[i * TP + j];
            for (int m = j + 1; m < i; m++) {
                val += sM[i * TP + m] * sM[m * TP + j];
            }
        }
        __syncthreads();
        if (j < i - 1) {
            sM[i * TP + j] = val;
        }
        __syncthreads();
    }
    // M = (T + I) diag(beta), lower triangular (each thread owns a column)
    if (tid < BT) {
        const int j = tid;
        for (int i = 0; i < BT; i++) {
            float m = 0.0f;
            if (i > j) {
                m = sM[i * TP + j] * sBeta[j];
            } else if (i == j) {
                m = sBeta[j];
            }
            sM[i * TP + j] = m;
        }
    }

    // ---- QG^T, KL^T; then k e^G in place of q. thread -> (token cc = lane, 16 dims per warp)
    const int cc = tid % 32;
    const int dw = (tid / 32) * (D / 8);
    for (int dd = 0; dd < D / 8; dd++) {
        const int d = dw + dd;
        const float G  = gT[d * TP + cc];
        const float Gl = gT[d * TP + BT - 1];
        const float kv = kT[d * TP + cc];
        out[gdn_layout::QGT + d * BT + cc] = src.scale * qT[d * TP + cc] * expf(G);
        out[gdn_layout::KLT + d * BT + cc] = kv * expf(Gl - G);
        qT[d * TP + cc] = kv * expf(G);
    }
    __syncthreads();

    // ---- w = M (k e^G), u = M v, transposed
    for (int dd = 0; dd < D / 8; dd++) {
        const int d = dw + dd;
        float w = 0.0f, u = 0.0f;
        for (int j = 0; j <= cc; j++) {
            const float m = sM[cc * TP + j];
            w += m * qT[d * TP + j];
            u += m * vT[d * TP + j];
        }
        out[gdn_layout::WT + d * BT + cc] = w;
        out[gdn_layout::UT + d * BT + cc] = u;
    }
}

// 16-byte global -> shared copy: asynchronous where cp.async exists, else a plain copy
static __device__ __forceinline__ void gdn_copy16(float * dst, const float * src) {
#ifdef CP_ASYNC_AVAILABLE
    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dst), src);
#else
    *(float4 *) dst = *(const float4 *) src;
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void gdn_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template <int n>
static __device__ __forceinline__ void gdn_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}

static constexpr int GDN_SPAD = GDN_NV + 1;
static constexpr size_t GDN_INTER_SMEM =
    (3 * GDN_D * GDN_BT        // W^T, QG^T, KL^T
     + GDN_NV * GDN_BT         // u^T of this block's columns
     + GDN_BT * GDN_BT         // A
     + GDN_D                   // exp(G_last)
     + GDN_BT * GDN_NV         // v'
     + GDN_D * GDN_SPAD) * sizeof(float); // state S[D][NV]

__global__ void __launch_bounds__(GDN_THREADS) gdn_chunk_inter(
        const float * __restrict__ tmp, const float * __restrict__ s_in, float * __restrict__ s_out,
        float * __restrict__ dst, const int H, const int64_t n_tokens, const int64_t t0, const int n_chunks_seg,
        const int n_chunks_run, const int dbg) {
    constexpr int D  = GDN_D;
    constexpr int BT = GDN_BT;
    constexpr int NV = GDN_NV;
    constexpr int SP = GDN_SPAD;

    static_assert(NV == WARP_SIZE && BT == 32, "one value column per lane, one token per lane");
    static_assert(GDN_THREADS / WARP_SIZE == 8 && D == 128, "the thread mapping assumes these sizes");

    extern __shared__ float smem[];
    float * sW  = smem;              // [D][BT]
    float * sQ  = sW + D * BT;       // [D][BT]
    float * sK  = sQ + D * BT;       // [D][BT]
    float * sU  = sK + D * BT;       // [NV][BT]
    float * sA  = sU + NV * BT;      // [BT][BT]
    float * sGL = sA + BT * BT;      // [D]
    float * sVN = sGL + D;           // [BT][NV]
    float * sS  = sVN + BT * NV;     // [D][SP]

    const int cg   = blockIdx.x;
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int n0   = cg * NV;

    const float * head = tmp + (int64_t) (seq * H + h) * n_chunks_seg * gdn_layout::stride;
    auto chunk = [&](int c) { return head + (int64_t) c * gdn_layout::stride; };

    // operand groups, each committed as one cp.async group: 1 = (W, u), 2 = (QG, A), 3 = (KL, exp(G_last))
    auto load1 = [&](int c) {
        const float * b = chunk(c);
        for (int i = tid * 4; i < D * BT; i += GDN_THREADS * 4) gdn_copy16(sW + i, b + gdn_layout::WT + i);
        for (int i = tid * 4; i < NV * BT; i += GDN_THREADS * 4) gdn_copy16(sU + i, b + gdn_layout::UT + n0 * BT + i);
    };
    auto load2 = [&](int c) {
        const float * b = chunk(c);
        for (int i = tid * 4; i < D * BT; i += GDN_THREADS * 4) gdn_copy16(sQ + i, b + gdn_layout::QGT + i);
        for (int i = tid * 4; i < BT * BT; i += GDN_THREADS * 4) gdn_copy16(sA + i, b + gdn_layout::A + i);
    };
    auto load3 = [&](int c) {
        const float * b = chunk(c);
        for (int i = tid * 4; i < D * BT; i += GDN_THREADS * 4) gdn_copy16(sK + i, b + gdn_layout::KLT + i);
        for (int i = tid * 4; i < D; i += GDN_THREADS * 4) gdn_copy16(sGL + i, b + gdn_layout::GL + i);
    };

    const float * sin_h  = s_in  + (int64_t) (seq * H + h) * D * D;
    float *       sout_h = s_out + (int64_t) (seq * H + h) * D * D;
    // state [col][row] per (seq, head): S[i][col] = state[col * D + i]
    for (int idx = tid; idx < D * NV; idx += GDN_THREADS) {
        const int nn = idx / D;
        const int i  = idx % D;
        sS[i * SP + nn] = sin_h[(int64_t) (n0 + nn) * D + i];
    }

    load1(0); gdn_commit();
    load2(0); gdn_commit();
    load3(0); gdn_commit();

    const int n  = lane;          // value column
    const int c0 = warp * 4;      // v' and o: 4 tokens per warp
    const int d0 = warp * 16;     // state update: 16 rows per warp

    for (int c = 0; c < n_chunks_run; c++) {
        const bool next = c + 1 < n_chunks_run;

        // v' = u - w S                                     [groups pending: 1(c) 2(c) 3(c)]
        gdn_wait<2>();
        __syncthreads();
        if (!(dbg & 1)) {
            float acc[4] = {};
            for (int i = 0; i < D; i++) {
                const float s  = sS[i * SP + n];
                const float4 w = *(const float4 *) (sW + i * BT + c0);
                acc[0] += w.x * s; acc[1] += w.y * s; acc[2] += w.z * s; acc[3] += w.w * s;
            }
#pragma unroll
            for (int r = 0; r < 4; r++) {
                sVN[(c0 + r) * NV + n] = sU[n * BT + c0 + r] - acc[r];
            }
        }
        __syncthreads();
        if (next && !(dbg & 8)) load1(c + 1);
        gdn_commit();

        // o = qg S + A v'                                  [pending: 2(c) 3(c) 1(c+1)]
        gdn_wait<2>();
        __syncthreads();
        if (!(dbg & 2)) {
            float acc[4] = {};
            for (int i = 0; i < D; i++) {
                const float s  = sS[i * SP + n];
                const float4 q = *(const float4 *) (sQ + i * BT + c0);
                acc[0] += q.x * s; acc[1] += q.y * s; acc[2] += q.z * s; acc[3] += q.w * s;
            }
            for (int j = 0; j < c0 + 4; j++) {
                const float vn = sVN[j * NV + n];
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    acc[r] += sA[(c0 + r) * BT + j] * vn; // A is zero above the diagonal
                }
            }
            const int64_t tc0 = t0 + (int64_t) c * BT;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int64_t tok = tc0 + c0 + r;
                if (tok < n_tokens) {
                    dst[((int64_t) seq * n_tokens * H + tok * H + h) * D + n0 + n] = acc[r];
                }
            }
        }
        __syncthreads();
        if (next && !(dbg & 8)) load2(c + 1);
        gdn_commit();

        // S = exp(G_last) S + kl^T v'                      [pending: 3(c) 1(c+1) 2(c+1)]
        gdn_wait<2>();
        __syncthreads();
        if (!(dbg & 4)) {
            float acc[16];
#pragma unroll
            for (int r = 0; r < 16; r++) {
                acc[r] = sGL[d0 + r] * sS[(d0 + r) * SP + n];
            }
            for (int j = 0; j < BT; j += 4) {
                const float v0 = sVN[(j + 0) * NV + n];
                const float v1 = sVN[(j + 1) * NV + n];
                const float v2 = sVN[(j + 2) * NV + n];
                const float v3 = sVN[(j + 3) * NV + n];
#pragma unroll
                for (int r = 0; r < 16; r++) {
                    const float4 kl = *(const float4 *) (sK + (d0 + r) * BT + j);
                    acc[r] += kl.x * v0 + kl.y * v1 + kl.z * v2 + kl.w * v3;
                }
            }
#pragma unroll
            for (int r = 0; r < 16; r++) {
                sS[(d0 + r) * SP + n] = acc[r];
            }
        }
        __syncthreads();
        if (next && !(dbg & 8)) load3(c + 1);
        gdn_commit();
    }
    gdn_wait<0>();
    __syncthreads();
    for (int idx = tid; idx < D * NV; idx += GDN_THREADS) {
        const int nn = idx / D;
        const int i  = idx % D;
        sout_h[(int64_t) (n0 + nn) * D + i] = sS[i * SP + nn];
    }
}

bool ggml_cuda_gated_delta_net_chunked(
        ggml_backend_cuda_context & ctx,
        const float * q, const float * k, const float * v, const float * g, const float * beta, const float * s0,
        float * dst, float * state_out,
        bool kda, int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, float scale, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    if (S_v != GDN_D || ggml_cuda_info().devices[device].warp_size != 32 ||
            ggml_cuda_info().devices[device].smpbo < std::max(GDN_INTRA_SMEM, GDN_INTER_SMEM)) {
        return false;
    }

    static bool smem_set[GGML_CUDA_MAX_DEVICES] = {};
    if (!smem_set[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(gdn_chunk_intra, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) GDN_INTRA_SMEM));
        CUDA_CHECK(cudaFuncSetAttribute(gdn_chunk_inter, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) GDN_INTER_SMEM));
        smem_set[device] = true;
    }

    const gdn_src src = { q, k, v, g, beta, kda, (int) H, (int) H_k, n_tokens, rq3,
                          sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, scale };

    const int64_t n_chunks   = (n_tokens + GDN_BT - 1) / GDN_BT;
    const int     seg_chunks = (int) std::min<int64_t>(n_chunks, GDN_SEG_CHUNKS);

    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), (size_t) n_seqs * H * seg_chunks * gdn_layout::stride);

    // GGML_CUDA_GDN_SKIP=1/2 skips kernel 1/2 (timing only, wrong results)
    static const int skip = getenv("GGML_CUDA_GDN_SKIP") ? atoi(getenv("GGML_CUDA_GDN_SKIP")) : 0;
    static const int dbg  = getenv("GGML_CUDA_GDN_DBG")  ? atoi(getenv("GGML_CUDA_GDN_DBG"))  : 0;

    for (int64_t c0 = 0; c0 < n_chunks; c0 += seg_chunks) {
        const int n_run = (int) std::min<int64_t>(seg_chunks, n_chunks - c0);
        const int64_t t0 = c0 * GDN_BT;

        if (skip != 1) {
            gdn_chunk_intra<<<dim3(n_run, H, n_seqs), GDN_THREADS, GDN_INTRA_SMEM, stream>>>(src, tmp.get(), t0, seg_chunks);
            CUDA_CHECK(cudaGetLastError());
        }
        // the first segment starts from s0, the next ones from the state the previous one wrote
        if (skip != 2) {
            gdn_chunk_inter<<<dim3(GDN_D / GDN_NV, H, n_seqs), GDN_THREADS, GDN_INTER_SMEM, stream>>>(
                tmp.get(), c0 == 0 ? s0 : state_out, state_out, dst, (int) H, n_tokens, t0, seg_chunks, n_run, dbg);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    return true;
}
