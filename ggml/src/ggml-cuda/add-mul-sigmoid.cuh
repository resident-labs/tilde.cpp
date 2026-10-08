#pragma once

#include "common.cuh"

// fused ADD -> MUL -> SIGMOID -> SCALE: out = s * sigmoid((x + b) * m), with b and m broadcast like ggml_add/ggml_mul
// (the decay gate of KDA linear attention: lower_bound * sigmoid(A * (f + dt_bias)))
bool ggml_cuda_add_mul_sigmoid_scale_fusable(const ggml_tensor * add, const ggml_tensor * mul, const ggml_tensor * sigm,
        const ggml_tensor * scale);
void ggml_cuda_op_add_mul_sigmoid_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * mul,
        const ggml_tensor * sigm, ggml_tensor * scale);
