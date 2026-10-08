#pragma once

#include "common.cuh"

// grouped sigmoid MoE routing (sigmoid, bias, group top-2 scores, top groups, top-k, normalized weights) as one
// kernel: the 26-node subgraph of build_moe_ffn that starts at the router's SIGMOID (moe-route.cu)
bool ggml_cuda_moe_route_match(const ggml_cgraph * cgraph, int node_idx);
// the same op sequence (no shape checks, no data): graph_optimize keeps the logits allocated until the last node
bool ggml_cuda_moe_route_maybe(const ggml_cgraph * cgraph, int node_idx);
int  ggml_cuda_moe_route_n_nodes();
ggml_tensor * ggml_cuda_moe_route_logits(const ggml_cgraph * cgraph, int node_idx);
ggml_tensor * ggml_cuda_moe_route_last(const ggml_cgraph * cgraph, int node_idx);
void ggml_cuda_moe_route(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
