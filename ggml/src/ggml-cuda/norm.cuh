#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// ADD -> RMS_NORM -> MUL by a weight vector, the ADD result kept (the residual stream), in one pass
bool ggml_cuda_add_rms_norm_mul_fusable(const ggml_tensor * add, const ggml_tensor * norm, const ggml_tensor * mul);
// q8, s8 (optional): also the int8 rows of the result (ggml_cuda_mmi8_pre_quant)
void ggml_cuda_op_add_rms_norm_mul_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * norm, ggml_tensor * mul,
        int8_t * q8 = nullptr, float * s8 = nullptr);

// RMS_NORM -> MUL (weight vector) that also writes the int8 rows of the result
bool ggml_cuda_rms_norm_mul_q8_ok(const ggml_tensor * norm, const ggml_tensor * mul);
void ggml_cuda_op_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul, int8_t * q8, float * s8);
