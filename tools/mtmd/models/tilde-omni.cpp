#include "models.h"

// Tilde Omni (tilde-4b-omni) media encoders. Every output token carries 3072 floats: the 2560-wide LM input
// embedding, then 512 values the LM's point head reads (the 4 x 128 sub-patch keys of an image token; zeros for audio).

// vision: TLive-Omni-4B's Qwen3.5 ViT up to its merger LayerNorm (the pre-merger cells, 4 x 1024 per 32 px cell), the
// cells of each image token averaged (blank merging, tilda.vision.merge), the frozen merger MLP, then Tilde's bridge:
// linear + residual MLP + position encoding of each token's geometry (tilda.vision.model.VisionBridge)
// ref: tilda/qwen35/model/online.py (TLiveVision), tilda/vision/model.py (VisionModule.image_tokens, VisionBridge)
ggml_cgraph * clip_graph_tilde_omni_v::build() {
    GGML_ASSERT(model.patch_bias != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int batch_size       = 1;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position
    const int n_cells          = n_pos / 4;
    const int n_tokens         = img.tilde_tokens > 0 ? img.tilde_tokens : n_cells;

    norm_type norm_t = NORM_TYPE_NORMAL;

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = build_inp_with_temporal_merge();

    // spatial merge order: the 2x2 patches of a cell are consecutive
    {
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    inp = ggml_add(ctx0, inp, model.patch_bias);
    cb(inp, "patch_bias", -1);

    ggml_tensor * learned_pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    learned_pos_embd = ggml_cont_4d(ctx0, learned_pos_embd, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
    learned_pos_embd = ggml_reshape_4d(ctx0, learned_pos_embd, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
    learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = ggml_cont_3d(ctx0, learned_pos_embd, n_embd, n_patches_x * n_patches_y, batch_size);
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(inp, "inp_pos_emb", -1);

    ggml_tensor * inpL = inp;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];

        ggml_tensor * cur = build_norm(inpL, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        {
            cur = build_mm(layer.qkv_w, cur);
            cur = ggml_add(ctx0, cur, layer.qkv_b);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    ggml_row_size(cur->type, d_head), cur->nb[1], 0);
            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    ggml_row_size(cur->type, d_head), cur->nb[1], ggml_row_size(cur->type, n_embd));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    ggml_row_size(cur->type, d_head), cur->nb[1], ggml_row_size(cur->type, 2 * n_embd));

            Qcur = ggml_rope_multi(ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // merger LayerNorm (per patch): the pre-merger cells [4 * n_embd, n_cells]
    inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    ggml_tensor * cells = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_cells);
    cb(cells, "tilde_cells", -1);

    // inputs: the cell -> token averaging matrix, the position features and the representative cell of each token
    ggml_tensor * merge = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_cells, n_tokens);
    ggml_set_name(merge, "tilde_merge");
    ggml_set_input(merge);
    ggml_tensor * pos_feat = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.tilde_pos_w->ne[0], n_tokens);
    ggml_set_name(pos_feat, "tilde_pos");
    ggml_set_input(pos_feat);
    ggml_tensor * rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(rep, "tilde_rep");
    ggml_set_input(rep);
    ggml_tensor * single = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    ggml_set_name(single, "tilde_single");
    ggml_set_input(single);

    // each token: the mean of its cells
    ggml_tensor * tok = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cells)), merge); // [4 * n_embd, n_tokens]

    // the frozen merger MLP (the bridge's projector), then the bridge
    ggml_tensor * proj = build_ffn(tok, model.mm_0_w, model.mm_0_b, nullptr, nullptr, model.mm_1_w, model.mm_1_b,
            FFN_GELU_ERF, -1);
    ggml_tensor * emb = ggml_add(ctx0, ggml_mul_mat(ctx0, model.tilde_lin_w, proj), model.tilde_lin_b);
    emb = ggml_add(ctx0, emb, build_ffn(proj, model.tilde_mlp0_w, model.tilde_mlp0_b, nullptr, nullptr,
            model.tilde_mlp2_w, model.tilde_mlp2_b, FFN_GELU_ERF, -1));
    emb = ggml_add(ctx0, emb, ggml_add(ctx0, ggml_mul_mat(ctx0, model.tilde_pos_w, pos_feat), model.tilde_pos_b));
    cb(emb, "tilde_embd", -1);

    // the point head's sub-patch keys: the 4 sub-patch features of each single-cell token, LayerNorm, linear
    ggml_tensor * sub = ggml_get_rows(ctx0, cells, rep);  // [4 * n_embd, n_tokens]
    sub = ggml_mul(ctx0, sub, single);
    sub = ggml_reshape_2d(ctx0, sub, n_embd, 4 * n_tokens);
    sub = build_norm(sub, model.tilde_sub_key_norm_w, model.tilde_sub_key_norm_b, NORM_TYPE_NORMAL, 1e-5f, -1);
    sub = ggml_mul_mat(ctx0, model.tilde_sub_key_w, sub); // [n_sub, 4 * n_tokens]
    sub = ggml_reshape_2d(ctx0, sub, sub->ne[0] * 4, n_tokens);

    ggml_tensor * out = ggml_concat(ctx0, emb, sub, 0);
    cb(out, "tilde_out", -1);

    ggml_build_forward_expand(gf, out);
    return gf;
}

// audio: TLive-Omni-4B's AuT (the qwen3a graph, keeping only the tokens of the window's valid frames, as the reference
// drops the padded positions of the last chunk before the transformer) and Tilde's native audio bridge (a linear map
// in the LM space), with zeros for the point head's side values
// ref: transformers Qwen3OmniMoeAudioEncoder.forward, tilda/qwen35/model/online.py (TLiveAudio), tilda/audio/model.py
ggml_cgraph * clip_graph_tilde_omni_a::build() {
    ggml_tensor * inp = build_inp_raw(1);

    const int64_t n_frames   = inp->ne[0]; // padded to a multiple of chunk_size
    const int64_t n_mel      = inp->ne[1];
    const int64_t chunk_size = 100;
    const int64_t n_chunks   = n_frames / chunk_size;
    GGML_ASSERT(n_frames % chunk_size == 0);

    inp = ggml_view_4d(ctx0, inp, chunk_size, n_mel, 1, n_chunks,
        n_frames * (int64_t) sizeof(float), chunk_size * (int64_t) sizeof(float), chunk_size * (int64_t) sizeof(float), 0);
    inp = ggml_cont(ctx0, inp);

    auto conv_block = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        x = ggml_conv_2d(ctx0, w, x, 2, 2, 1, 1, 1, 1);
        x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b, 1, 1, x->ne[2], 1));
        return ggml_gelu_erf(ctx0, x);
    };
    inp = conv_block(inp, model.conv2d_1_w, model.conv2d_1_b);
    inp = conv_block(inp, model.conv2d_2_w, model.conv2d_2_b);
    inp = conv_block(inp, model.conv2d_3_w, model.conv2d_3_b);

    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2] * inp->ne[3]);
    inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);

    const int64_t n_pos = inp->ne[1];
    {
        const int64_t tokens_per_chunk = n_pos / n_chunks; // 13
        ggml_tensor * pos_tmp = ggml_view_2d(ctx0, model.position_embeddings,
            model.position_embeddings->ne[0], tokens_per_chunk, model.position_embeddings->nb[1], 0);
        ggml_tensor * tgt = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.position_embeddings->ne[0], n_pos);
        inp = ggml_add(ctx0, inp, ggml_repeat(ctx0, pos_tmp, tgt));
    }

    // the valid tokens come first (the padding is at the end of the last chunk)
    const int64_t n_valid = img.tilde_tokens > 0 ? img.tilde_tokens : n_pos;
    GGML_ASSERT(n_valid <= n_pos);
    if (n_valid < n_pos) {
        inp = ggml_view_2d(ctx0, inp, inp->ne[0], n_valid, inp->nb[1], 0);
    }

    ggml_tensor * cur = build_vit(inp, n_valid, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, nullptr);
    cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_2_w, model.mm_2_b, FFN_GELU_ERF, -1);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.tilde_lin_w, cur), model.tilde_lin_b);
    cur = ggml_pad(ctx0, cur, 512, 0, 0, 0);
    cb(cur, "tilde_out", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
