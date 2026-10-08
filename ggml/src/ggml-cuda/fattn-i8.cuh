#pragma once

#include "common.cuh"

// int8 Q*K^T flash attention for long unmasked prefill (decompressed MLA: DKQ 192, DV 128), see fattn-i8.cu
bool ggml_cuda_flash_attn_ext_i8_applies(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
