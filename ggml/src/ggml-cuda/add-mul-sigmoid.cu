#include "add-mul-sigmoid.cuh"

static __global__ void add_mul_sigmoid_scale_f32(
        const float * __restrict__ x, const char * __restrict__ b, const char * __restrict__ m, float * __restrict__ dst,
        const int64_t n, const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const int64_t b0, const int64_t b1, const int64_t b2, const int64_t b3,
        const int64_t bn0, const int64_t bn1, const int64_t bn2, const int64_t bn3,
        const int64_t m0, const int64_t m1, const int64_t m2, const int64_t m3,
        const int64_t mn0, const int64_t mn1, const int64_t mn2, const int64_t mn3, const float s) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t i0 = i % ne0;
    const int64_t i1 = (i / ne0) % ne1;
    const int64_t i2 = (i / (ne0 * ne1)) % ne2;
    const int64_t i3 = i / (ne0 * ne1 * ne2);
    const float bv = *(const float *) (b + (i0 % b0) * bn0 + (i1 % b1) * bn1 + (i2 % b2) * bn2 + (i3 % b3) * bn3);
    const float mv = *(const float *) (m + (i0 % m0) * mn0 + (i1 % m1) * mn1 + (i2 % m2) * mn2 + (i3 % m3) * mn3);
    const float z  = (x[i] + bv) * mv;
    dst[i] = s / (1.0f + expf(-z));
}

// the KDA layout: x [ne0, ne1, ...] contiguous, b [ne0, ne1] contiguous (per channel), m [1, ne1] (per head)
static __global__ void add_mul_sigmoid_scale_f32_chan_head(
        const float * __restrict__ x, const float * __restrict__ b, const float * __restrict__ m, float * __restrict__ dst,
        const uint32_t n, const uint3 ne0_fd, const uint3 ne01_fd, const float s) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const uint32_t ib = fastmodulo(i, ne01_fd);
    const uint32_t ih = fastdiv(ib, ne0_fd);
    const float z = (x[i] + b[ib]) * m[ih];
    dst[i] = s / (1.0f + expf(-z));
}

static const ggml_tensor * other_src(const ggml_tensor * op, const ggml_tensor * prev) {
    return op->src[0] == prev ? op->src[1] : op->src[0];
}

bool ggml_cuda_add_mul_sigmoid_scale_fusable(const ggml_tensor * add, const ggml_tensor * mul, const ggml_tensor * sigm,
        const ggml_tensor * scale) {
    if (sigm->op != GGML_OP_UNARY || ggml_get_unary_op(sigm) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    if (!(mul->src[0] == add || mul->src[1] == add) || sigm->src[0] != mul || scale->src[0] != sigm) {
        return false;
    }
    const ggml_tensor * x = add->src[0];
    const ggml_tensor * b = add->src[1];
    const ggml_tensor * m = other_src(mul, add);
    float bias = 0.0f;
    memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));
    // x is the full-shape operand; b and m broadcast into it
    return x->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F32 &&
        add->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 && bias == 0.0f &&
        ggml_is_contiguous(x) && ggml_are_same_shape(x, add) && ggml_are_same_shape(add, scale) &&
        ggml_can_repeat(b, x) && ggml_can_repeat(m, x) && ggml_is_contiguous(scale);
}

void ggml_cuda_op_add_mul_sigmoid_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * mul,
        const ggml_tensor * sigm, ggml_tensor * scale) {
    GGML_UNUSED(sigm);
    const ggml_tensor * x = add->src[0];
    const ggml_tensor * b = add->src[1];
    const ggml_tensor * m = other_src(mul, add);
    float s = 1.0f;
    memcpy(&s, (const float *) scale->op_params + 0, sizeof(float));
    const int64_t n = ggml_nelements(x);
    const int threads = 256;
    if (n < INT32_MAX && ggml_is_contiguous(b) && ggml_is_contiguous(m) && b->ne[0] == x->ne[0] && b->ne[1] == x->ne[1] &&
            ggml_nelements(b) == x->ne[0] * x->ne[1] && m->ne[0] == 1 && m->ne[1] == x->ne[1] && ggml_nelements(m) == x->ne[1]) {
        add_mul_sigmoid_scale_f32_chan_head<<<(n + threads - 1) / threads, threads, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) b->data, (const float *) m->data, (float *) scale->data, (uint32_t) n,
            init_fastdiv_values((uint32_t) x->ne[0]), init_fastdiv_values((uint32_t) (x->ne[0] * x->ne[1])), s);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    add_mul_sigmoid_scale_f32<<<(n + threads - 1) / threads, threads, 0, ctx.stream()>>>(
        (const float *) x->data, (const char *) b->data, (const char *) m->data, (float *) scale->data,
        n, x->ne[0], x->ne[1], x->ne[2],
        b->ne[0], b->ne[1], b->ne[2], b->ne[3], b->nb[0], b->nb[1], b->nb[2], b->nb[3],
        m->ne[0], m->ne[1], m->ne[2], m->ne[3], m->nb[0], m->nb[1], m->nb[2], m->nb[3], s);
    CUDA_CHECK(cudaGetLastError());
}
