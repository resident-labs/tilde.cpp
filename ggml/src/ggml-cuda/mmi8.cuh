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

// the producer of `out` (node node_idx) can write its int8 rows (one scale per row) for the I8 products that read
// it or ranges of its rows: returns the buffers (q [rows][K], s [rows]) when one of the next nodes does
bool ggml_cuda_mmi8_pre_quant(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx,
        const ggml_tensor * out, int8_t * & q, float * & s);

// MUL_MAT_ID with I8 experts [K, N, E] and the per-row-per-expert scales [N, E] in src[3] (ggml_mul_mat_id_scaled):
// tokens sorted by expert, per-token int8 activations, one CUTLASS grouped int8 GEMM for all experts, then the
// scales and the scatter back to token order.
bool ggml_cuda_mul_mat_id_i8_supported(int device, const ggml_tensor * dst);
// what happens to the [N, n_used, T] result: written as is (out = the MUL_MAT_ID node); split into gate and up halves
// and SwiGLU'd (out = the GLU node, [N/2, n_used, T]); or summed over each token's experts with the router weights
// [1, n_used, T] (out = [N, T], the MoE output; with add1 and add2, out = add2 + (add1 + MoE output): the shared
// expert and the residual of the next two ADDs of the graph)
enum mmid_i8_epilogue {
    MMID_I8_SCATTER,
    MMID_I8_SWIGLU,
    MMID_I8_REDUCE,
};
void ggml_cuda_mul_mat_id_i8(ggml_backend_cuda_context & ctx, const ggml_tensor * mmid, const float * w_scale, ggml_tensor * out,
        mmid_i8_epilogue epilogue = MMID_I8_SCATTER, const ggml_tensor * weights = nullptr,
        const ggml_cgraph * cgraph = nullptr, int node_idx = -1, const ggml_tensor * add1 = nullptr,
        const ggml_tensor * add2 = nullptr);
