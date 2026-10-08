#include "moe-route.cuh"

// Grouped sigmoid MoE routing (Ling / BailingMoE2 style, used by Tilde) in one kernel. The graph that build_moe_ffn
// makes for it (26 nodes, 3 ARGSORTs, ~20 small launches per layer):
//
//     p  = sigmoid(logits)                       [E, T]
//     s  = p + bias                              selection scores
//     group score = sum of the top-2 s of each group of E / G experts
//     keep the top-n_group_used groups, s = -inf elsewhere
//     ids = top-k of s (descending)              [k, T]   (the ARGSORT + VIEW the MUL_MAT_IDs read)
//     w   = p[ids] / max(sum p[ids], clamp_min) * w_scale     [1, k, T]
//
// One warp per token; E = 128 experts (4 per lane), G = 8 groups of 16 (4 lanes per group), group top-2,
// n_group_used <= 8, k <= 16. Ties go to the lower index (the ARGSORTs leave their order unspecified).

static constexpr int ROUTE_E = 128;
static constexpr int ROUTE_G = 8;

static __device__ __forceinline__ float route_pick(const float (&v)[4], const int j) {
    return j == 0 ? v[0] : j == 1 ? v[1] : j == 2 ? v[2] : v[3];
}

// warp-wide argmax of (val, idx): larger val, then smaller idx
static __device__ __forceinline__ void route_argmax(float & val, int & idx) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, val, off);
        const int   oi = __shfl_xor_sync(0xffffffff, idx, off);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
}

template <int k>
static __global__ void moe_group_route_f32(const float * __restrict__ logits, const float * __restrict__ bias,
        int32_t * __restrict__ ids, float * __restrict__ weights, const int64_t n_tokens, const int64_t logits_nb1,
        const int64_t ids_nb1, const int64_t weights_nb2, const int n_group_used, const float clamp_min,
        const float w_scale) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (t >= n_tokens) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const float4 lg = *(const float4 *) ((const char *) logits + t * logits_nb1 + lane * 4 * sizeof(float));
    const float4 bs = *(const float4 *) (bias + lane * 4);
    float p[4], s[4];
    p[0] = 1.0f / (1.0f + expf(-lg.x));
    p[1] = 1.0f / (1.0f + expf(-lg.y));
    p[2] = 1.0f / (1.0f + expf(-lg.z));
    p[3] = 1.0f / (1.0f + expf(-lg.w));
    s[0] = p[0] + bs.x; s[1] = p[1] + bs.y; s[2] = p[2] + bs.z; s[3] = p[3] + bs.w;

    // top-2 of this lane's 4, then of the group's 16 (lanes 4g .. 4g+3)
    float m1 = fmaxf(s[0], s[1]), m2 = fminf(s[0], s[1]);
#pragma unroll
    for (int j = 2; j < 4; j++) {
        if (s[j] > m1) {
            m2 = m1;
            m1 = s[j];
        } else if (s[j] > m2) {
            m2 = s[j];
        }
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        const float o1 = __shfl_xor_sync(0xffffffff, m1, off);
        const float o2 = __shfl_xor_sync(0xffffffff, m2, off);
        const float n1 = fmaxf(m1, o1);
        const float n2 = m1 >= o1 ? fmaxf(m2, o1) : fmaxf(m1, o2);
        m1 = n1;
        m2 = n2;
    }
    const float gscore = m1 + m2;
    const int   group  = lane / 4;

    // the top n_group_used groups
    bool keep = false;
    for (int r = 0; r < n_group_used; r++) {
        float v = keep ? -INFINITY : gscore;
        int   g = keep ? ROUTE_G : group;
        route_argmax(v, g);
        keep = keep || g == group;
    }
    if (!keep) {
        s[0] = s[1] = s[2] = s[3] = -INFINITY;
    }

    // the top k experts, in descending order
    int   sel_id[k];
    float sel_p[k];
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < k; r++) {
        float v = s[0];
        int   j = 0;
#pragma unroll
        for (int jj = 1; jj < 4; jj++) {
            if (s[jj] > v) {
                v = s[jj];
                j = jj;
            }
        }
        int e = lane * 4 + j;
        route_argmax(v, e);
        const int owner = e / 4;
        const float pv = __shfl_sync(0xffffffff, route_pick(p, e % 4), owner);
        if (lane == owner) {
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                if (jj == e % 4) {
                    s[jj] = -INFINITY;
                }
            }
        }
        sel_id[r] = e;
        sel_p[r]  = pv;
        sum += pv;
    }
    const float denom = fmaxf(sum, clamp_min);
    if (lane < k) {
        int   e  = sel_id[0];
        float pv = sel_p[0];
#pragma unroll
        for (int r = 1; r < k; r++) {
            if (lane == r) {
                e  = sel_id[r];
                pv = sel_p[r];
            }
        }
        ids[t * (ids_nb1 / sizeof(int32_t)) + lane] = e;
        weights[t * (weights_nb2 / sizeof(float)) + lane] = w_scale * (pv / denom);
    }
}

// the 26-node routing subgraph starting at the SIGMOID at node_idx (see the top of the file)
static const std::initializer_list<enum ggml_op> moe_route_ops = {
    GGML_OP_UNARY,     // 0  p = sigmoid(logits)
    GGML_OP_RESHAPE,   // 1  p [1, E, T]
    GGML_OP_ADD,       // 2  s = p + bias
    GGML_OP_RESHAPE,   // 3  s [E/G, G, T]
    GGML_OP_RESHAPE,   // 4  s [1, E/G, G, T]
    GGML_OP_ARGSORT,   // 5
    GGML_OP_VIEW,      // 6  top 2 per group
    GGML_OP_GET_ROWS,  // 7
    GGML_OP_RESHAPE,   // 8
    GGML_OP_SUM_ROWS,  // 9  group scores
    GGML_OP_RESHAPE,   // 10
    GGML_OP_ARGSORT,   // 11
    GGML_OP_VIEW,      // 12 top groups
    GGML_OP_GET_ROWS,  // 13
    GGML_OP_FILL,      // 14
    GGML_OP_SET_ROWS,  // 15 masked scores
    GGML_OP_RESHAPE,   // 16
    GGML_OP_ARGSORT,   // 17
    GGML_OP_VIEW,      // 18 ids (output)
    GGML_OP_GET_ROWS,  // 19 p[ids]
    GGML_OP_RESHAPE,   // 20
    GGML_OP_SUM_ROWS,  // 21
    GGML_OP_CLAMP,     // 22
    GGML_OP_DIV,       // 23
    GGML_OP_RESHAPE,   // 24
    GGML_OP_SCALE,     // 25 weights (output)
};

static constexpr int MOE_ROUTE_IDS = 18;
static constexpr int MOE_ROUTE_W   = 25;

bool ggml_cuda_moe_route_match(const ggml_cgraph * cgraph, int node_idx) {
    static const bool disabled = getenv("GGML_CUDA_NO_MOE_ROUTE_FUSION") != nullptr;
    const int n = (int) moe_route_ops.size();
    if (disabled || node_idx + n > cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * const * nd = cgraph->nodes + node_idx;
    if (nd[0]->op != GGML_OP_UNARY || ggml_get_unary_op(nd[0]) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const int outputs[] = { node_idx + MOE_ROUTE_IDS, node_idx + MOE_ROUTE_W };
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, moe_route_ops, { outputs[0], outputs[1] })) {
        return false;
    }
    const ggml_tensor * logits = nd[0]->src[0];
    const int64_t E = nd[0]->ne[0];
    const int64_t T = nd[0]->ne[1];
    const ggml_tensor * bias = nd[2]->src[0] == nd[0] ? nd[2]->src[1] : nd[2]->src[0];
    const ggml_tensor * ids = nd[MOE_ROUTE_IDS];
    const ggml_tensor * w   = nd[MOE_ROUTE_W];
    const int k = (int) ids->ne[0];
    float fill = 0.0f, cmax = 0.0f, sc_bias = 0.0f;
    memcpy(&fill, nd[14]->op_params, sizeof(float));
    memcpy(&cmax, (const float *) nd[22]->op_params + 1, sizeof(float));
    memcpy(&sc_bias, (const float *) nd[25]->op_params + 1, sizeof(float));
    return E == ROUTE_E && logits->type == GGML_TYPE_F32 && logits->ne[0] == E && logits->ne[1] == T &&
        logits->nb[0] == sizeof(float) && logits->nb[1] % 16 == 0 && ggml_nrows(logits) == T &&
        nd[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(nd[0]) &&
        // s = p + bias, bias a constant vector [E]
        nd[1]->src[0] == nd[0] && (nd[2]->src[0] == nd[0] || nd[2]->src[1] == nd[0]) && bias->type == GGML_TYPE_F32 &&
        nd[4]->src[0] == nd[3] && nd[8]->src[0] == nd[7] && nd[10]->src[0] == nd[9] &&
        ggml_nelements(bias) == E && ggml_is_contiguous(bias) && bias->op == GGML_OP_NONE &&
        // groups: [E/G, G, T], top-2 per group
        nd[3]->src[0] == nd[2] && nd[3]->ne[0] == E / ROUTE_G && nd[3]->ne[1] == ROUTE_G &&
        nd[5]->src[0] == nd[3] && ggml_get_op_params_i32(nd[5], 0) == GGML_SORT_ORDER_DESC &&
        nd[6]->src[0] == nd[5] && nd[6]->ne[0] == 2 &&
        nd[7]->src[0] == nd[4] && nd[7]->src[1] == nd[6] && nd[9]->src[0] == nd[8] &&
        nd[11]->src[0] == nd[10] && ggml_get_op_params_i32(nd[11], 0) == GGML_SORT_ORDER_DESC &&
        nd[12]->src[0] == nd[11] && nd[12]->ne[0] >= 1 && nd[12]->ne[0] <= ROUTE_G &&
        nd[13]->src[0] == nd[3] && nd[13]->src[1] == nd[12] && nd[14]->src[0] == nd[3] && fill == -INFINITY &&
        nd[15]->src[0] == nd[13] && nd[15]->src[1] == nd[12] && nd[15]->src[2] == nd[14] &&
        nd[16]->src[0] == nd[15] && nd[16]->ne[0] == E &&
        nd[17]->src[0] == nd[16] && ggml_get_op_params_i32(nd[17], 0) == GGML_SORT_ORDER_DESC &&
        ids->src[0] == nd[17] && ids->view_offs == 0 && k >= 1 && k <= 16 && ids->ne[1] == T &&
        nd[17]->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
        // w = p[ids] / clamp(sum) * scale
        nd[19]->src[0] == nd[1] && nd[19]->src[1] == ids && nd[20]->src[0] == nd[19] && nd[21]->src[0] == nd[20] &&
        nd[22]->src[0] == nd[21] && cmax == INFINITY && nd[23]->src[0] == nd[20] && nd[23]->src[1] == nd[22] &&
        nd[24]->src[0] == nd[23] && w->src[0] == nd[24] && sc_bias == 0.0f &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == 1 && w->ne[1] == k && w->ne[2] == T;
}

bool ggml_cuda_moe_route_maybe(const ggml_cgraph * cgraph, int node_idx) {
    const int n = (int) moe_route_ops.size();
    if (node_idx + n > cgraph->n_nodes || cgraph->nodes[node_idx]->op != GGML_OP_UNARY ||
            ggml_get_unary_op(cgraph->nodes[node_idx]) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    int j = 0;
    for (const ggml_op op : moe_route_ops) {
        if (cgraph->nodes[node_idx + j++]->op != op) {
            return false;
        }
    }
    return true;
}

int ggml_cuda_moe_route_n_nodes() {
    return (int) moe_route_ops.size();
}

ggml_tensor * ggml_cuda_moe_route_logits(const ggml_cgraph * cgraph, int node_idx) {
    return cgraph->nodes[node_idx]->src[0];
}

ggml_tensor * ggml_cuda_moe_route_last(const ggml_cgraph * cgraph, int node_idx) {
    return cgraph->nodes[node_idx + MOE_ROUTE_W];
}

void ggml_cuda_moe_route(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * const * nd = cgraph->nodes + node_idx;
    const ggml_tensor * logits = nd[0]->src[0];
    const ggml_tensor * bias = nd[2]->src[0] == nd[0] ? nd[2]->src[1] : nd[2]->src[0];
    const ggml_tensor * ids = nd[MOE_ROUTE_IDS];
    ggml_tensor * w = nd[MOE_ROUTE_W];
    const int64_t T = nd[0]->ne[1];
    const int k = (int) ids->ne[0];
    const int n_group_used = (int) nd[12]->ne[0];
    float clamp_min = 0.0f, w_scale = 1.0f;
    memcpy(&clamp_min, (const float *) nd[22]->op_params + 0, sizeof(float));
    memcpy(&w_scale, (const float *) w->op_params + 0, sizeof(float));

    constexpr int rows_per_block = 4;
    const dim3 grid((unsigned) ((T + rows_per_block - 1) / rows_per_block));
    const dim3 block(rows_per_block * WARP_SIZE);
    cudaStream_t stream = ctx.stream();
#define MOE_ROUTE_K(K) \
    case K: moe_group_route_f32<K><<<grid, block, 0, stream>>>((const float *) logits->data, (const float *) bias->data, \
        (int32_t *) ids->data, (float *) w->data, T, logits->nb[1], ids->nb[1], w->nb[2], n_group_used, clamp_min, w_scale); break;
    switch (k) {
        MOE_ROUTE_K(1)  MOE_ROUTE_K(2)  MOE_ROUTE_K(3)  MOE_ROUTE_K(4)
        MOE_ROUTE_K(5)  MOE_ROUTE_K(6)  MOE_ROUTE_K(7)  MOE_ROUTE_K(8)
        MOE_ROUTE_K(9)  MOE_ROUTE_K(10) MOE_ROUTE_K(11) MOE_ROUTE_K(12)
        MOE_ROUTE_K(13) MOE_ROUTE_K(14) MOE_ROUTE_K(15) MOE_ROUTE_K(16)
        default: GGML_ABORT("unsupported k");
    }
#undef MOE_ROUTE_K
    CUDA_CHECK(cudaGetLastError());
}
