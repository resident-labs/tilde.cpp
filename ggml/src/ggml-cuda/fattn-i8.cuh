#pragma once

#include "common.cuh"

// int8 Q*K^T flash attention for long unmasked prefill (decompressed MLA: DKQ 192, DV 128), see fattn-i8.cu
bool ggml_cuda_flash_attn_ext_i8_applies(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// a CPY (f32 -> f16) whose only consumer is a FLASH_ATTN_EXT that takes the int8 path as K (or as V with int8 P*V):
// the copy can be skipped, the kernel reads the f32 source
bool ggml_cuda_fattn_i8_cast_elidable(int device, const ggml_cgraph * cgraph, int node_idx);
// the attention node that such a copy feeds (-1 if the copy cannot be skipped): the graph keeps the copy's f32 source
// allocated until that node (ggml_backend_cuda_graph_optimize)
int ggml_cuda_fattn_i8_cast_consumer(int device, const ggml_cgraph * cgraph, int node_idx);
