#include "gated_delta_net_chunked.cuh"

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
// Kernel 1 computes w, u, A and the gated q/k of every chunk in parallel; kernel 2 runs the chunks in order, one
// block per (head, group of value columns). exp(G_c - G_i) is evaluated per term (never exp(-G) alone), so any gate
// magnitude is safe. Everything is f32.
//
// The sequence is processed in segments of SEG_CHUNKS chunks so the temporary buffer stays small.

static constexpr int GDN_BT         = 16;  // tokens per chunk
static constexpr int GDN_NV         = 16;  // value columns per block of the sequential kernel
static constexpr int GDN_SEG_CHUNKS = 256; // chunks per segment (4096 tokens)

template <int D>
struct gdn_chunk_layout {
    static constexpr int W      = 0;
    static constexpr int QG     = W  + GDN_BT * D;
    static constexpr int KL     = QG + GDN_BT * D;
    static constexpr int U      = KL + GDN_BT * D;
    static constexpr int A      = U  + GDN_BT * D;
    static constexpr int GL     = A  + GDN_BT * GDN_BT;
    static constexpr int stride = GL + D;
};

template <int D>
__global__ void __launch_bounds__(D) gdn_chunk_intra(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta, float * __restrict__ tmp,
        const bool kda, const int H, const int H_k, const int64_t n_tokens, const int64_t t0, const int n_chunks_seg,
        const int64_t rq3,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3, const float scale) {
    using L = gdn_chunk_layout<D>;
    constexpr int BT = GDN_BT;

    const int c   = blockIdx.x;
    const int h   = blockIdx.y;
    const int seq = blockIdx.z;
    const int d   = threadIdx.x;
    const int lane = d % WARP_SIZE;
    const int warp = d / WARP_SIZE;
    constexpr int n_warps = D / WARP_SIZE;

    const int64_t tc0 = t0 + (int64_t) c * BT;
    const int nvalid = (int) min((int64_t) BT, n_tokens - tc0);
    const int hk = h % H_k;
    const int64_t sk = seq / rq3;

    __shared__ float s_q[BT][D];  // scale * q
    __shared__ float s_k[BT][D];
    __shared__ float s_G[BT][D];  // cumulative gate inside the chunk
    __shared__ float s_beta[BT];
    __shared__ float s_N[BT][BT];
    __shared__ float s_M[BT][BT];
    __shared__ float s_A[BT][BT];

    float * out = tmp + ((int64_t) (seq * H + h) * n_chunks_seg + c) * L::stride;

    float G = 0.0f;
    float vcol[BT];
#pragma unroll
    for (int t = 0; t < BT; t++) {
        float qv = 0.0f, kv = 0.0f, gv = 0.0f, bv = 0.0f;
        vcol[t] = 0.0f;
        if (t < nvalid) {
            const int64_t tok = tc0 + t;
            qv = q[sk * sq3 + tok * sq2 + hk * sq1 + d];
            kv = k[sk * sq3 + tok * sq2 + hk * sq1 + d];
            vcol[t] = v[seq * sv3 + tok * sv2 + h * sv1 + d];
            const int64_t gb = seq * sb3 + tok * sb2 + h * sb1;
            gv = kda ? g[gb * D + d] : g[gb];
            bv = beta[gb];
        }
        G += gv;
        s_q[t][d] = qv * scale;
        s_k[t][d] = kv;
        s_G[t][d] = G;
        if (d == 0) {
            s_beta[t] = bv;
        }
    }
    const float G_last = G;
    __syncthreads();

    // A[i][j] = sum_d q_i k_j exp(G_i - G_j) (j <= i), N[c][i] = -beta_c sum_d k_c k_i exp(G_c - G_i) (c > i):
    // 136 + 120 dot products of length D, one warp each
    for (int p = warp; p < 256; p += n_warps) {
        int i, j;
        bool is_A;
        if (p < 136) {
            // p = i*(i+1)/2 + j, j <= i
            i = 0;
            while ((i + 1) * (i + 2) / 2 <= p) i++;
            j = p - i * (i + 1) / 2;
            is_A = true;
        } else {
            // p - 136 = i*(i-1)/2 + j, j < i
            const int r = p - 136;
            i = 1;
            while (i * (i + 1) / 2 <= r) i++;
            j = r - i * (i - 1) / 2;
            is_A = false;
        }
        const float * a  = is_A ? s_q[i] : s_k[i];
        const float * b  = s_k[j];
        const float * Gi = s_G[i];
        const float * Gj = s_G[j];
        float sum = 0.0f;
#pragma unroll
        for (int dd = lane; dd < D; dd += WARP_SIZE) {
            sum += a[dd] * b[dd] * expf(Gi[dd] - Gj[dd]);
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            if (is_A) {
                s_A[i][j] = sum;
            } else {
                s_N[i][j] = -s_beta[i] * sum;
            }
        }
    }
    __syncthreads();

    // M = (I - N)^-1 diag(beta), by forward substitution: T = N + N T (strictly lower), M = (T + I) diag(beta)
    if (warp == 0 && lane < BT) {
        const int col = lane;
        float T[BT];
#pragma unroll
        for (int i = 0; i < BT; i++) {
            float t = 0.0f;
            if (i > col) {
                t = s_N[i][col];
#pragma unroll
                for (int m = 0; m < BT; m++) {
                    if (m > col && m < i) {
                        t += s_N[i][m] * T[m];
                    }
                }
            }
            T[i] = t;
            s_M[i][col] = (t + (i == col ? 1.0f : 0.0f)) * s_beta[col];
        }
    }
    __syncthreads();

    // w = M (k * exp(G)), u = M v (column d); gated q and k for the sequential kernel
    float kgd[BT];
#pragma unroll
    for (int j = 0; j < BT; j++) {
        kgd[j] = s_k[j][d] * expf(s_G[j][d]);
    }
#pragma unroll
    for (int cc = 0; cc < BT; cc++) {
        float w = 0.0f, u = 0.0f;
#pragma unroll
        for (int j = 0; j < BT; j++) {
            w += s_M[cc][j] * kgd[j];
            u += s_M[cc][j] * vcol[j];
        }
        out[L::W  + cc * D + d] = w;
        out[L::U  + cc * D + d] = u;
        out[L::QG + cc * D + d] = s_q[cc][d] * expf(s_G[cc][d]);
        out[L::KL + cc * D + d] = s_k[cc][d] * expf(G_last - s_G[cc][d]);
    }
    out[L::GL + d] = expf(G_last);
    for (int idx = d; idx < BT * BT; idx += D) {
        const int i = idx / BT;
        const int j = idx % BT;
        out[L::A + idx] = j <= i ? s_A[i][j] : 0.0f;
    }
}

template <int D>
__global__ void __launch_bounds__(D) gdn_chunk_inter(
        const float * __restrict__ tmp, const float * __restrict__ s_in, float * __restrict__ s_out,
        float * __restrict__ dst,
        const int H, const int64_t n_tokens, const int64_t t0, const int n_chunks_seg, const int n_chunks_run) {
    using L = gdn_chunk_layout<D>;
    constexpr int BT = GDN_BT;
    constexpr int NV = GDN_NV;

    const int cg  = blockIdx.x;
    const int h   = blockIdx.y;
    const int seq = blockIdx.z;
    const int tid = threadIdx.x;
    const int n0  = cg * NV;

    __shared__ float s_S[D][NV + 1]; // +1: the state update reads a row per thread
    __shared__ float s_W[BT][D];
    __shared__ float s_QG[BT][D];
    __shared__ float s_U[BT][NV];
    __shared__ float s_A[BT][BT];
    __shared__ float s_VN[BT][NV];

    // state [col][row] per (seq, head): S[i][col] = state[col * D + i]
    const float * sin_h = s_in  + (int64_t) (seq * H + h) * D * D;
    float *       sout_h = s_out + (int64_t) (seq * H + h) * D * D;
    for (int idx = tid; idx < D * NV; idx += D) {
        const int n = idx / D;
        const int i = idx % D;
        s_S[i][n] = sin_h[(int64_t) (n0 + n) * D + i];
    }

    // thread -> (row c, column n) for the BT x NV outputs: 2 per thread with D = 128
    static_assert(BT * NV % D == 0, "bad block size");
    constexpr int per_thread = BT * NV / D;

    for (int c = 0; c < n_chunks_run; c++) {
        const float * base = tmp + ((int64_t) (seq * H + h) * n_chunks_seg + c) * L::stride;
        __syncthreads(); // previous chunk done with the shared buffers
#pragma unroll
        for (int cc = 0; cc < BT; cc++) {
            s_W [cc][tid] = base[L::W  + cc * D + tid];
            s_QG[cc][tid] = base[L::QG + cc * D + tid];
        }
        for (int idx = tid; idx < BT * NV; idx += D) {
            s_U[idx / NV][idx % NV] = base[L::U + (idx / NV) * D + n0 + idx % NV];
        }
        for (int idx = tid; idx < BT * BT; idx += D) {
            s_A[idx / BT][idx % BT] = base[L::A + idx];
        }
        const float gl = base[L::GL + tid];
        float kl[BT];
#pragma unroll
        for (int cc = 0; cc < BT; cc++) {
            kl[cc] = base[L::KL + cc * D + tid];
        }
        __syncthreads();

        // v' = u - w S
#pragma unroll
        for (int r = 0; r < per_thread; r++) {
            const int idx = tid + r * D;
            const int cc = idx / NV;
            const int n  = idx % NV;
            float sum = 0.0f;
#pragma unroll 16
            for (int i = 0; i < D; i++) {
                sum += s_W[cc][i] * s_S[i][n];
            }
            s_VN[cc][n] = s_U[cc][n] - sum;
        }
        __syncthreads();

        // o = qg S + A v'
        const int64_t tc0 = t0 + (int64_t) c * BT;
#pragma unroll
        for (int r = 0; r < per_thread; r++) {
            const int idx = tid + r * D;
            const int cc = idx / NV;
            const int n  = idx % NV;
            float sum = 0.0f;
#pragma unroll 16
            for (int i = 0; i < D; i++) {
                sum += s_QG[cc][i] * s_S[i][n];
            }
#pragma unroll
            for (int j = 0; j < BT; j++) {
                if (j <= cc) {
                    sum += s_A[cc][j] * s_VN[j][n];
                }
            }
            const int64_t tok = tc0 + cc;
            if (tok < n_tokens) {
                dst[((int64_t) seq * n_tokens * H + tok * H + h) * D + n0 + n] = sum;
            }
        }
        __syncthreads();

        // S = gl * S + kl^T v'   (row tid)
#pragma unroll
        for (int n = 0; n < NV; n++) {
            float s = gl * s_S[tid][n];
#pragma unroll
            for (int cc = 0; cc < BT; cc++) {
                s += kl[cc] * s_VN[cc][n];
            }
            s_S[tid][n] = s;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < D * NV; idx += D) {
        const int n = idx / D;
        const int i = idx % D;
        sout_h[(int64_t) (n0 + n) * D + i] = s_S[i][n];
    }
}

bool ggml_cuda_gated_delta_net_chunked(
        ggml_backend_cuda_context & ctx,
        const float * q, const float * k, const float * v, const float * g, const float * beta, const float * s0,
        float * dst, float * state_out,
        bool kda, int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, float scale, cudaStream_t stream) {
    constexpr int D = 128;
    if (S_v != D || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32) {
        return false;
    }
    using L = gdn_chunk_layout<D>;

    const int64_t n_chunks   = (n_tokens + GDN_BT - 1) / GDN_BT;
    const int     seg_chunks = (int) std::min<int64_t>(n_chunks, GDN_SEG_CHUNKS);

    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), (size_t) n_seqs * H * seg_chunks * L::stride);

    // GGML_CUDA_GDN_SKIP=1/2 skips kernel 1/2 (timing only, wrong results)
    static const int skip = getenv("GGML_CUDA_GDN_SKIP") ? atoi(getenv("GGML_CUDA_GDN_SKIP")) : 0;
    for (int64_t c0 = 0; c0 < n_chunks; c0 += seg_chunks) {
        const int n_run = (int) std::min<int64_t>(seg_chunks, n_chunks - c0);
        const int64_t t0 = c0 * GDN_BT;

        if (skip != 1) gdn_chunk_intra<D><<<dim3(n_run, H, n_seqs), D, 0, stream>>>(
            q, k, v, g, beta, tmp.get(), kda, (int) H, (int) H_k, n_tokens, t0, seg_chunks, rq3,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, scale);
        CUDA_CHECK(cudaGetLastError());

        // the first segment starts from s0, the next ones from the state the previous one wrote
        if (skip != 2) gdn_chunk_inter<D><<<dim3(D / GDN_NV, H, n_seqs), D, 0, stream>>>(
            tmp.get(), c0 == 0 ? s0 : state_out, state_out, dst, (int) H, n_tokens, t0, seg_chunks, n_run);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
