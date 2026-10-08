#include "gated_delta_net_chunked.cuh"
#include "cp-async.cuh"

// Chunked prefill for GGML_OP_GATED_DELTA_NET (KDA per-channel gates, or one scalar gate per head).
//
// The per-token recurrence (gated_delta_net.cu)
//     S_t = diag(exp(g_t)) S_{t-1};  S_t += beta_t k_t (v_t - S_t^T k_t)^T;  o_t = scale * S_t^T q_t
// is evaluated in chunks of BT = 32 tokens with the WY representation (flash-linear-attention, chunk_kda):
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
// chunk's operands with cp.async while it works on the current one.
//
// The matrix products run on tensor cores (mma m16n8k8, TF32 operands, f32 accumulation; NVIDIA Ampere or newer); the
// reference implementation (flash-linear-attention) feeds the same products bf16 operands. The recurrent state stays
// f32. Gate factors are never above 1, for any gate magnitude: inside a 16-token sub-chunk exp(G_c - G_i) is evaluated
// per pair; across sub-chunks it is split as exp(G_c - G_16) * exp(G_16 - G_i) (c >= 16 > i), as in fla's chunk_kda.
//
// The sequence is processed in segments of SEG_CHUNKS chunks so the temporary buffer stays small.

static constexpr int GDN_D          = 128; // head size (S_v == S_k)
static constexpr int GDN_BT         = 32;  // tokens per chunk
static constexpr int GDN_NV         = 32;  // value columns per block of the sequential kernel
static constexpr int GDN_SEG_CHUNKS = 128; // chunks per segment (4096 tokens)
static constexpr int GDN_THREADS    = 256;

// per (seq, head, chunk) in the temporary buffer, token-major ([BT][D] rows of 128 floats)
struct gdn_layout {
    static constexpr int W      = 0;                          // w                 [BT][D]
    static constexpr int QG     = W  + GDN_BT * GDN_D;        // scale q e^G       [BT][D]
    static constexpr int KL     = QG + GDN_BT * GDN_D;        // k e^(Gl - G)      [BT][D]
    static constexpr int U      = KL + GDN_BT * GDN_D;        // u                 [BT][D]
    static constexpr int A      = U  + GDN_BT * GDN_D;        // A                 [BT][BT]
    static constexpr int GL     = A  + GDN_BT * GDN_BT;       // exp(G_last)       [D]
    static constexpr int stride = GL + GDN_D;
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

static __device__ __forceinline__ uint32_t gdn_tf32(const float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

// C += A B with A 16x8 (row), B 8x8 (col), TF32 operands, f32 accumulation (g = lane / 4, tq = lane % 4):
//   a0 (g, tq), a1 (g+8, tq), a2 (g, tq+4), a3 (g+8, tq+4);  b0 (k tq, n g), b1 (k tq+4, n g)
//   c0 (g, 2tq), c1 (g, 2tq+1), c2 (g+8, 2tq), c3 (g+8, 2tq+1)
static __device__ __forceinline__ void gdn_mma(float (&c)[4], const float a0, const float a1, const float a2,
        const float a3, const float b0, const float b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(gdn_tf32(a0)), "r"(gdn_tf32(a1)), "r"(gdn_tf32(a2)), "r"(gdn_tf32(a3)), "r"(gdn_tf32(b0)), "r"(gdn_tf32(b1)));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------------------------------
// kernel 1: one block per (chunk, head, seq). The dims are processed in passes of DP; k e^G and v stay in shared memory
// for w = M k e^G and u = M v.

static constexpr int GDN_DP = 16;            // dims per pass
static constexpr int GDN_PS = GDN_DP + 4;    // row stride of the pass buffers (conflict-free fragment loads)
static constexpr int GDN_RS = GDN_D + 8;     // row stride of the resident [BT][D] buffers
static constexpr int GDN_MS = GDN_BT + 4;    // row stride of N / M
static constexpr int GDN_NDIAG = 2 * 16 * 17 / 2; // (row, col) pairs of the two diagonal 16x16 blocks, col <= row

// pass buffers: q, k, G [BT][PS]; qa, ka (tokens 16..31), kb (tokens 0..15) [16][PS]
static constexpr int GDN_PASS_FLOATS = (3 * GDN_BT + 3 * 16) * GDN_PS;
static_assert(GDN_PASS_FLOATS >= GDN_BT * GDN_MS, "M fits in the pass buffers");
static constexpr size_t GDN_INTRA_SMEM = (2 * GDN_BT * GDN_RS + GDN_PASS_FLOATS + GDN_BT) * sizeof(float);

__global__ void __launch_bounds__(GDN_THREADS, 2) gdn_chunk_intra(const gdn_src src, float * __restrict__ tmp,
        const int64_t t0, const int n_chunks_seg) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int D  = GDN_D;
    constexpr int BT = GDN_BT;
    constexpr int DP = GDN_DP;
    constexpr int PS = GDN_PS;
    constexpr int RS = GDN_RS;
    constexpr int MS = GDN_MS;

    extern __shared__ __align__(16) float smem[];
    float * sKE   = smem;                 // [BT][RS]  k e^G
    float * sV    = sKE + BT * RS;        // [BT][RS]  v
    float * sQ    = sV  + BT * RS;        // [BT][PS]  q            (pass)
    float * sK    = sQ  + BT * PS;        // [BT][PS]  k            (pass)
    float * sG    = sK  + BT * PS;        // [BT][PS]  g, then G    (pass)
    float * sQa   = sG  + BT * PS;        // [16][PS]  q e^(G - G_16), tokens 16..31
    float * sKa   = sQa + 16 * PS;        // [16][PS]  k e^(G - G_16), tokens 16..31
    float * sKb   = sKa + 16 * PS;        // [16][PS]  k e^(G_16 - G), tokens 0..15
    float * sM    = sQ;                   // [BT][MS]  N, then M (after the passes)
    float * sBeta = sQ + GDN_PASS_FLOATS; // [BT]

    const int c    = blockIdx.x;
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int g    = lane / 4;
    const int tq   = lane % 4;

    const int64_t tc0 = t0 + (int64_t) c * BT;
    const int nvalid = (int) min((int64_t) BT, src.n_tokens - tc0);
    const int hk = h % src.H_k;
    const int64_t sk = seq / src.rq3;

    float * out = tmp + ((int64_t) (seq * src.H + h) * n_chunks_seg + c) * gdn_layout::stride;

    if (tid < BT) {
        sBeta[tid] = tid < nvalid ? src.beta[seq * src.sb3 + (tc0 + tid) * src.sb2 + h * src.sb1] : 0.0f;
    }

    // diagonal blocks (rows and cols in the same 16-token sub-chunk, col <= row): per pair, f32. Thread -> pairs
    // tid and tid + 256 (fewer than 2 * 256 pairs)
    int   pr[2], pc[2];
    float dA[2] = {}, dN[2] = {};
#pragma unroll
    for (int p = 0; p < 2; p++) {
        int idx = tid + p * GDN_THREADS;
        if (idx >= GDN_NDIAG) {
            idx = GDN_NDIAG - 1; // duplicate of a real pair, never written
        }
        const int blk = idx / (16 * 17 / 2);
        int rem = idx % (16 * 17 / 2);
        int r = 0;
        while (rem > r) { rem -= r + 1; r++; }
        pr[p] = blk * 16 + r;
        pc[p] = blk * 16 + rem;
    }
    const bool has2 = tid + GDN_THREADS < GDN_NDIAG;

    // off-diagonal block (rows 16..31, cols 0..15) on tensor cores: warps 0 and 1, cols warp * 8 .. + 7
    float accQ[4] = {};
    float accK[4] = {};

    // loader mapping: token lt, dims 2 * ldp, 2 * ldp + 1 of the pass
    const int lt  = tid / (DP / 2);
    const int ldp = tid % (DP / 2);

    for (int d0 = 0; d0 < D; d0 += DP) {
        const int dd = 2 * ldp;
        const int d  = d0 + dd;
        float2 qv = make_float2(0.0f, 0.0f), kv = qv, vv = qv, gv = qv;
        if (lt < nvalid) {
            const int64_t tok = tc0 + lt;
            const int64_t qo  = sk * src.sq3 + tok * src.sq2 + hk * src.sq1 + d;
            qv = *(const float2 *) (src.q + qo);
            kv = *(const float2 *) (src.k + qo);
            vv = *(const float2 *) (src.v + seq * src.sv3 + tok * src.sv2 + h * src.sv1 + d);
            const int64_t gb = seq * src.sb3 + tok * src.sb2 + h * src.sb1;
            gv = src.kda ? *(const float2 *) (src.g + gb * D + d) : make_float2(src.g[gb], src.g[gb]);
        }
        *(float2 *) (sQ + lt * PS + dd) = qv;
        *(float2 *) (sK + lt * PS + dd) = kv;
        *(float2 *) (sG + lt * PS + dd) = gv;
        *(float2 *) (sV + lt * RS + d)  = vv;
        __syncthreads();

        // inclusive cumsum over the tokens: warp w scans dims 2w, 2w+1 of the pass (lane = token)
        {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float x = sG[lane * PS + 2 * warp + i];
#pragma unroll
                for (int off = 1; off < WARP_SIZE; off *= 2) {
                    const float y = __shfl_up_sync(0xffffffff, x, off);
                    if (lane >= off) {
                        x += y;
                    }
                }
                sG[lane * PS + 2 * warp + i] = x;
            }
        }
        __syncthreads();

        // gated operands and the per-chunk outputs that need only q, k, G
        {
            float qg[2], kl[2], gl[2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const float qe  = i ? qv.y : qv.x;
                const float ke  = i ? kv.y : kv.x;
                const float G   = sG[lt * PS + dd + i];
                const float R   = sG[16 * PS + dd + i];
                const float Gl  = sG[(BT - 1) * PS + dd + i];
                const float eG  = expf(G);
                if (lt >= 16) {
                    const float e = expf(G - R);
                    sQa[(lt - 16) * PS + dd + i] = qe * e;
                    sKa[(lt - 16) * PS + dd + i] = ke * e;
                } else {
                    sKb[lt * PS + dd + i] = ke * expf(R - G);
                }
                sKE[lt * RS + d + i] = ke * eG;
                qg[i] = src.scale * qe * eG;
                kl[i] = ke * expf(Gl - G);
                gl[i] = expf(Gl);
            }
            *(float2 *) (out + gdn_layout::QG + lt * D + d) = make_float2(qg[0], qg[1]);
            *(float2 *) (out + gdn_layout::KL + lt * D + d) = make_float2(kl[0], kl[1]);
            if (lt == 0) {
                *(float2 *) (out + gdn_layout::GL + d) = make_float2(gl[0], gl[1]);
            }
        }
        __syncthreads();

        // diagonal blocks: exp(G_r - G_c) <= 1 per pair
#pragma unroll
        for (int p = 0; p < 2; p++) {
            if (p == 1 && !has2) {
                break;
            }
            const float * qr = sQ + pr[p] * PS;
            const float * kr = sK + pr[p] * PS;
            const float * kc = sK + pc[p] * PS;
            const float * gr = sG + pr[p] * PS;
            const float * gc = sG + pc[p] * PS;
            float a = 0.0f, n = 0.0f;
#pragma unroll
            for (int j = 0; j < DP; j++) {
                const float e  = expf(gr[j] - gc[j]);
                const float ke = kc[j] * e;
                a += qr[j] * ke;
                n += kr[j] * ke;
            }
            dA[p] += a;
            dN[p] += n;
        }

        // off-diagonal block
        if (warp < 2) {
#pragma unroll
            for (int k0 = 0; k0 < DP; k0 += 8) {
                const float b0 = sKb[(warp * 8 + g) * PS + k0 + tq];
                const float b1 = sKb[(warp * 8 + g) * PS + k0 + tq + 4];
                gdn_mma(accQ, sQa[g * PS + k0 + tq], sQa[(g + 8) * PS + k0 + tq],
                              sQa[g * PS + k0 + tq + 4], sQa[(g + 8) * PS + k0 + tq + 4], b0, b1);
                gdn_mma(accK, sKa[g * PS + k0 + tq], sKa[(g + 8) * PS + k0 + tq],
                              sKa[g * PS + k0 + tq + 4], sKa[(g + 8) * PS + k0 + tq + 4], b0, b1);
            }
        }
        __syncthreads(); // the next pass overwrites the buffers
    }

    // A (scaled, col <= row) to the output, N (col < row) to shared memory. Zero first (the block above the diagonal
    // and the upper triangles of the diagonal blocks), then the computed entries.
    for (int idx = tid; idx < BT * BT; idx += GDN_THREADS) {
        const int r = idx / BT;
        const int s = idx % BT;
        if ((r < 16) == (s < 16) ? s > r : r < 16) {
            out[gdn_layout::A + r * BT + s] = 0.0f;
            sM[r * MS + s] = 0.0f;
        }
    }
#pragma unroll
    for (int p = 0; p < 2; p++) {
        if (p == 1 && !has2) {
            break;
        }
        const int r = pr[p];
        const int s = pc[p];
        out[gdn_layout::A + r * BT + s] = src.scale * dA[p];
        sM[r * MS + s] = s < r ? -sBeta[r] * dN[p] : 0.0f;
    }
    if (warp < 2) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = 16 + g + (i / 2) * 8;
            const int s = warp * 8 + 2 * tq + (i % 2);
            out[gdn_layout::A + r * BT + s] = src.scale * accQ[i];
            sM[r * MS + s] = -sBeta[r] * accK[i];
        }
    }
    __syncthreads();

    // T = (I - N)^-1 - I by forward substitution in one warp (lane j owns column j):
    //     T[i][j] = N[i][j] + sum_{m < i} N[i][m] T[m][j]     (T and N strictly lower triangular)
    // then M = (T + I) diag(beta)
    if (warp == 0) {
        const int j = lane;
        float Tc[BT];
#pragma unroll
        for (int i = 0; i < BT; i++) {
            float val = sM[i * MS + j];   // zero for j >= i
#pragma unroll
            for (int m = 0; m < i; m++) {
                val += sM[i * MS + m] * Tc[m];
            }
            Tc[i] = j < i ? val : 0.0f;
        }
        __syncwarp();
        const float bj = sBeta[j];
#pragma unroll
        for (int i = 0; i < BT; i++) {
            sM[i * MS + j] = (Tc[i] + (i == j ? 1.0f : 0.0f)) * bj;
        }
    }
    __syncthreads();

    // w = M (k e^G), u = M v: [BT x D], warp -> rows (warp & 1) * 16 .. + 15, cols (warp >> 1) * 32 .. + 31
    {
        const int mt  = warp & 1;
        const int nc0 = (warp >> 1) * 32;
        const int r0  = mt * 16 + g;
        float accW[4][4] = {};
        float accU[4][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < BT; k0 += 8) {
            if (mt == 0 && k0 >= 16) {
                break; // M is lower triangular: rows 0..15 have no entries in columns 16..31
            }
            const float a0 = sM[r0 * MS + k0 + tq];
            const float a1 = sM[(r0 + 8) * MS + k0 + tq];
            const float a2 = sM[r0 * MS + k0 + tq + 4];
            const float a3 = sM[(r0 + 8) * MS + k0 + tq + 4];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int n = nc0 + nt * 8 + g;
                gdn_mma(accW[nt], a0, a1, a2, a3, sKE[(k0 + tq) * RS + n], sKE[(k0 + tq + 4) * RS + n]);
                gdn_mma(accU[nt], a0, a1, a2, a3, sV[(k0 + tq) * RS + n],  sV[(k0 + tq + 4) * RS + n]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int n = nc0 + nt * 8 + 2 * tq;
            *(float2 *) (out + gdn_layout::W + r0 * D + n)       = make_float2(accW[nt][0], accW[nt][1]);
            *(float2 *) (out + gdn_layout::W + (r0 + 8) * D + n) = make_float2(accW[nt][2], accW[nt][3]);
            *(float2 *) (out + gdn_layout::U + r0 * D + n)       = make_float2(accU[nt][0], accU[nt][1]);
            *(float2 *) (out + gdn_layout::U + (r0 + 8) * D + n) = make_float2(accU[nt][2], accU[nt][3]);
        }
    }
#else
    GGML_UNUSED_VARS(src, tmp, t0, n_chunks_seg);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------------------------------
// kernel 2: the chunks in order. One block per (group of NV value columns, head, seq); the state S[D][NV] of the
// block's columns stays in shared memory (f32).

static __device__ __forceinline__ void gdn_copy16(float * dst, const float * src) {
    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dst), src);
}

template <int n>
static __device__ __forceinline__ void gdn_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void gdn_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

static constexpr int GDN_WS = GDN_D + 4;   // row stride of W, QG, KL ([BT][D]): conflict-free A fragments
static constexpr int GDN_VS = GDN_NV + 8;  // row stride of u, v', S ([.][NV]): conflict-free B fragments
static constexpr int GDN_AS = GDN_BT + 4;  // row stride of A
static constexpr size_t GDN_INTER_SMEM =
    (3 * GDN_BT * GDN_WS       // W, QG, KL
     + GDN_BT * GDN_VS         // u (this block's columns)
     + GDN_BT * GDN_AS         // A
     + GDN_D                   // exp(G_last)
     + GDN_BT * GDN_VS         // v'
     + GDN_D * GDN_VS) * sizeof(float); // S

__global__ void __launch_bounds__(GDN_THREADS, 1) gdn_chunk_inter(
        const float * __restrict__ tmp, const float * __restrict__ s_in, float * __restrict__ s_out,
        float * __restrict__ dst, const int H, const int64_t n_tokens, const int64_t t0, const int n_chunks_seg,
        const int n_chunks_run) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int D  = GDN_D;
    constexpr int BT = GDN_BT;
    constexpr int NV = GDN_NV;
    constexpr int WS = GDN_WS;
    constexpr int VS = GDN_VS;
    constexpr int AS = GDN_AS;

    static_assert(GDN_THREADS / WARP_SIZE == 8 && D == 128 && BT == 32 && NV == 32, "the warp mapping assumes these sizes");

    extern __shared__ __align__(16) float smem[];
    float * sW  = smem;              // [BT][WS]
    float * sQ  = sW + BT * WS;      // [BT][WS]
    float * sK  = sQ + BT * WS;      // [BT][WS]
    float * sU  = sK + BT * WS;      // [BT][VS]
    float * sA  = sU + BT * VS;      // [BT][AS]
    float * sGL = sA + BT * AS;      // [D]
    float * sVN = sGL + D;           // [BT][VS]  v'
    float * sS  = sVN + BT * VS;     // [D][VS]   S[i][n]

    const int cg   = blockIdx.x;
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int g    = lane / 4;
    const int tq   = lane % 4;
    const int n0   = cg * NV;

    const float * head = tmp + (int64_t) (seq * H + h) * n_chunks_seg * gdn_layout::stride;
    auto chunk = [&](int c) { return head + (int64_t) c * gdn_layout::stride; };

    // operand groups, each committed as one cp.async group: 1 = (W, u), 2 = (QG, A), 3 = (KL, exp(G_last))
    auto load_rows = [&](float * dstm, const float * srcm) {
        for (int i = tid; i < BT * (D / 4); i += GDN_THREADS) {
            const int t = i / (D / 4);
            const int q = i % (D / 4);
            gdn_copy16(dstm + t * WS + q * 4, srcm + t * D + q * 4);
        }
    };
    auto load1 = [&](int c) {
        const float * b = chunk(c);
        load_rows(sW, b + gdn_layout::W);
        for (int i = tid; i < BT * (NV / 4); i += GDN_THREADS) {
            const int t = i / (NV / 4);
            const int q = i % (NV / 4);
            gdn_copy16(sU + t * VS + q * 4, b + gdn_layout::U + t * D + n0 + q * 4);
        }
    };
    auto load2 = [&](int c) {
        const float * b = chunk(c);
        load_rows(sQ, b + gdn_layout::QG);
        for (int i = tid; i < BT * (BT / 4); i += GDN_THREADS) {
            const int t = i / (BT / 4);
            const int q = i % (BT / 4);
            gdn_copy16(sA + t * AS + q * 4, b + gdn_layout::A + t * BT + q * 4);
        }
    };
    auto load3 = [&](int c) {
        const float * b = chunk(c);
        load_rows(sK, b + gdn_layout::KL);
        for (int i = tid; i < D / 4; i += GDN_THREADS) {
            gdn_copy16(sGL + i * 4, b + gdn_layout::GL + i * 4);
        }
    };

    const float * sin_h  = s_in  + (int64_t) (seq * H + h) * D * D;
    float *       sout_h = s_out + (int64_t) (seq * H + h) * D * D;
    // state [col][row] per (seq, head): S[i][col] = state[col * D + i]
    for (int idx = tid; idx < D * NV; idx += GDN_THREADS) {
        const int nn = idx / D;
        const int i  = idx % D;
        sS[i * VS + nn] = sin_h[(int64_t) (n0 + nn) * D + i];
    }

    load1(0); gdn_commit();
    load2(0); gdn_commit();
    load3(0); gdn_commit();

    // v' and o: [BT x NV] as 2 x 4 tiles of 16 x 8, one per warp
    const int mt = warp >> 2;
    const int r0 = mt * 16 + g;              // fragment rows r0, r0 + 8
    const int cb = (warp & 3) * 8 + g;       // B fragment column
    const int cc = (warp & 3) * 8 + 2 * tq;  // C fragment columns cc, cc + 1
    // S update: rows warp * 16 .. + 15, all NV columns
    const int d0 = warp * 16 + g;            // fragment rows d0, d0 + 8

    for (int c = 0; c < n_chunks_run; c++) {
        const bool next = c + 1 < n_chunks_run;

        // v' = u - w S                                     [groups pending: 1(c) 2(c) 3(c)]
        gdn_wait<2>();
        __syncthreads();
        {
            float acc[4] = {};
#pragma unroll
            for (int k0 = 0; k0 < D; k0 += 8) {
                gdn_mma(acc, sW[r0 * WS + k0 + tq], sW[(r0 + 8) * WS + k0 + tq],
                             sW[r0 * WS + k0 + tq + 4], sW[(r0 + 8) * WS + k0 + tq + 4],
                             sS[(k0 + tq) * VS + cb], sS[(k0 + tq + 4) * VS + cb]);
            }
            const float2 u0 = *(const float2 *) (sU + r0 * VS + cc);
            const float2 u1 = *(const float2 *) (sU + (r0 + 8) * VS + cc);
            *(float2 *) (sVN + r0 * VS + cc)       = make_float2(u0.x - acc[0], u0.y - acc[1]);
            *(float2 *) (sVN + (r0 + 8) * VS + cc) = make_float2(u1.x - acc[2], u1.y - acc[3]);
        }
        __syncthreads();
        if (next) {
            load1(c + 1);
        }
        gdn_commit();

        // o = qg S + A v'                                  [pending: 2(c) 3(c) 1(c+1)]
        gdn_wait<2>();
        __syncthreads();
        {
            float acc[4] = {};
#pragma unroll
            for (int k0 = 0; k0 < D; k0 += 8) {
                gdn_mma(acc, sQ[r0 * WS + k0 + tq], sQ[(r0 + 8) * WS + k0 + tq],
                             sQ[r0 * WS + k0 + tq + 4], sQ[(r0 + 8) * WS + k0 + tq + 4],
                             sS[(k0 + tq) * VS + cb], sS[(k0 + tq + 4) * VS + cb]);
            }
            // A is lower triangular: rows 0..15 only need keys 0..15
#pragma unroll
            for (int k0 = 0; k0 < BT; k0 += 8) {
                if (mt == 0 && k0 >= 16) {
                    break;
                }
                gdn_mma(acc, sA[r0 * AS + k0 + tq], sA[(r0 + 8) * AS + k0 + tq],
                             sA[r0 * AS + k0 + tq + 4], sA[(r0 + 8) * AS + k0 + tq + 4],
                             sVN[(k0 + tq) * VS + cb], sVN[(k0 + tq + 4) * VS + cb]);
            }
            const int64_t tok0 = t0 + (int64_t) c * BT + r0;
            const int64_t tok1 = tok0 + 8;
            if (tok0 < n_tokens) {
                *(float2 *) (dst + ((int64_t) seq * n_tokens * H + tok0 * H + h) * D + n0 + cc) = make_float2(acc[0], acc[1]);
            }
            if (tok1 < n_tokens) {
                *(float2 *) (dst + ((int64_t) seq * n_tokens * H + tok1 * H + h) * D + n0 + cc) = make_float2(acc[2], acc[3]);
            }
        }
        __syncthreads();
        if (next) {
            load2(c + 1);
        }
        gdn_commit();

        // S = exp(G_last) S + kl^T v' (each warp reads and writes only its own rows of S)   [pending: 3(c) 1(c+1) 2(c+1)]
        gdn_wait<2>();
        __syncthreads();
        {
            const float gl0 = sGL[d0];
            const float gl1 = sGL[d0 + 8];
            float acc[4][4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int col = j * 8 + 2 * tq;
                const float2 s0 = *(const float2 *) (sS + d0 * VS + col);
                const float2 s1 = *(const float2 *) (sS + (d0 + 8) * VS + col);
                acc[j][0] = gl0 * s0.x; acc[j][1] = gl0 * s0.y;
                acc[j][2] = gl1 * s1.x; acc[j][3] = gl1 * s1.y;
            }
#pragma unroll
            for (int k0 = 0; k0 < BT; k0 += 8) {
                // A fragment = KL^T: row d, column t  ->  KL[t][d]
                const float a0 = sK[(k0 + tq) * WS + d0];
                const float a1 = sK[(k0 + tq) * WS + d0 + 8];
                const float a2 = sK[(k0 + tq + 4) * WS + d0];
                const float a3 = sK[(k0 + tq + 4) * WS + d0 + 8];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    gdn_mma(acc[j], a0, a1, a2, a3, sVN[(k0 + tq) * VS + j * 8 + g], sVN[(k0 + tq + 4) * VS + j * 8 + g]);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int col = j * 8 + 2 * tq;
                *(float2 *) (sS + d0 * VS + col)       = make_float2(acc[j][0], acc[j][1]);
                *(float2 *) (sS + (d0 + 8) * VS + col) = make_float2(acc[j][2], acc[j][3]);
            }
        }
        __syncthreads();
        if (next) {
            load3(c + 1);
        }
        gdn_commit();
    }
    gdn_wait<0>();
    __syncthreads();
    for (int idx = tid; idx < D * NV; idx += GDN_THREADS) {
        const int nn = idx / D;
        const int i  = idx % D;
        sout_h[(int64_t) (n0 + nn) * D + i] = sS[i * VS + nn];
    }
#else
    GGML_UNUSED_VARS(tmp, s_in, s_out, dst, H, n_tokens, t0, n_chunks_seg, n_chunks_run);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

bool ggml_cuda_gated_delta_net_chunked(
        ggml_backend_cuda_context & ctx,
        const float * q, const float * k, const float * v, const float * g, const float * beta, const float * s0,
        float * dst, float * state_out,
        bool kda, int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs, int64_t rq3,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, float scale, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    // TF32 tensor cores and cp.async: NVIDIA Ampere or newer
    if (S_v != GDN_D || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE ||
            ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE ||
            ggml_cuda_info().devices[device].smpbo < std::max(GDN_INTRA_SMEM, GDN_INTER_SMEM)) {
        return false;
    }
    // the loaders read float pairs
    if (((uintptr_t) q | (uintptr_t) k | (uintptr_t) v) % 8 != 0 || (sq1 | sq2 | sq3 | sv1 | sv2 | sv3) % 2 != 0 ||
            (kda && (uintptr_t) g % 8 != 0)) {
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

    for (int64_t c0 = 0; c0 < n_chunks; c0 += seg_chunks) {
        const int n_run = (int) std::min<int64_t>(seg_chunks, n_chunks - c0);
        const int64_t t0 = c0 * GDN_BT;

        gdn_chunk_intra<<<dim3(n_run, H, n_seqs), GDN_THREADS, GDN_INTRA_SMEM, stream>>>(src, tmp.get(), t0, seg_chunks);
        CUDA_CHECK(cudaGetLastError());
        // the first segment starts from s0, the next ones from the state the previous one wrote
        gdn_chunk_inter<<<dim3(GDN_D / GDN_NV, H, n_seqs), GDN_THREADS, GDN_INTER_SMEM, stream>>>(
            tmp.get(), c0 == 0 ? s0 : state_out, state_out, dst, (int) H, n_tokens, t0, seg_chunks, n_run);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
