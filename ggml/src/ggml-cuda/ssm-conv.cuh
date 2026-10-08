#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// fused CONCAT(state, transpose(x)) -> SSM_CONV -> SILU
bool ggml_cuda_ssm_conv_concat_fusable(const ggml_tensor * concat, const ggml_tensor * conv, const ggml_tensor * silu);
void ggml_cuda_op_ssm_conv_concat_silu(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * conv, ggml_tensor * silu);
