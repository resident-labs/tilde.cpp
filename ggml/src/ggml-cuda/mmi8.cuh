#pragma once

#include "common.cuh"

// MUL_MAT with GGML_TYPE_I8 weights (integer codes; the per-row scale is a following MUL by an [ne01] vector, fused
// here when present): per-token int8 activations, int8 tensor cores (CUTLASS, Ampere or newer), int32 accumulation,
// D[t][m] = acc * scale_x[t] * scale_w[m] in the epilogue.
bool ggml_cuda_mul_mat_i8_supported(int device, const ggml_tensor * dst);

// the MUL that applies the row scales of an I8 MUL_MAT, if it can be fused into the GEMM epilogue
bool ggml_cuda_mul_mat_i8_scale_fusable(const ggml_tensor * mm, const ggml_tensor * mul);

// dst = src0 (I8) x src1 (F32) [* w_scale per row of src0 when w_scale != nullptr, written to out]. With the graph
// and the node index, the int8 activations are kept for the next I8 MUL_MATs on the same src1 (until a later node
// writes to its memory).
void ggml_cuda_mul_mat_i8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * mm, const float * w_scale, ggml_tensor * out, const ggml_cgraph * cgraph, int node_idx);
