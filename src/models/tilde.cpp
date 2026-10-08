#include "models.h"

#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

// Tilde: Ling-3.0 backbone (bailingmoe3: KDA linear attention + MLA + MoE) with non-causal MLA layers, and a Clef-style
// joint decision head. The whole request is one ubatch and no memory module is used: the KDA layers start from zero
// states and stay causal, the MLA layers attend over the whole request.
//
// The head follows JointSchemaHead (Cloudflare Clef, the "fast" vectorized forward of tilda.clef_joint_fast). The
// fitted temperature is folded into the stored scales by the converter, so the outputs are calibrated logits.

void llama_model_tilde::load_arch_hparams(llama_model_loader & ml) {
    llama_model_bailingmoe3::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DECISION_ROUTING_BLOCK_COUNT, n_layer_routing);
    ml.get_key(LLM_KV_DECISION_BLOCK_COUNT,         n_layer_joint);
    ml.get_key(LLM_KV_DECISION_HEAD_COUNT,          n_head_decision);

    if (n_head_decision == 0 || n_layer_routing > LLAMA_MAX_LAYERS || n_layer_joint > LLAMA_MAX_LAYERS) {
        throw std::runtime_error("invalid size of the decision head");
    }
    if (hparams.n_layer_nextn != 0) {
        throw std::runtime_error("tilde: MTP layers are not supported");
    }

    // used by the head
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    // the output is one score per token, see llama_batch_ext_set_decision_order()
    hparams.n_embd_out_impl = 1;
}

void llama_model_tilde::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    // the frozen output table of the base model: the head reads the rows of the option tokens
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, 0);

    const int64_t head_dim         = hparams.n_embd_head_kda;
    const int64_t d_inner          = head_dim * n_head;
    const int64_t d_conv           = hparams.ssm_d_conv;
    const int64_t kv_lora_rank     = hparams.n_lora_kv;
    const int64_t q_lora_rank      = hparams.n_lora_q;
    const int64_t qk_rope_head_dim = hparams.n_rot();
    const int64_t qk_head_dim      = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim       = hparams.n_embd_head_v_mla();
    const int64_t qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, 0);

        if (hparams.is_recr(il)) {
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", il), { d_conv, 1, d_inner, 1 }, 0);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", il), { d_conv, 1, d_inner, 1 }, 0);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", il), { d_conv, 1, d_inner, 1 }, 0);

            create_tensor_qkv(layer, il, n_embd, d_inner, d_inner, d_inner, 0);
            layer.ssm_f_a    = create_tensor(tn(LLM_TENSOR_SSM_F_A,  "weight", il), { n_embd, d_inner }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", il), { n_embd, n_head }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, il),       { 1, n_head }, 0);
            layer.ssm_dt_b   = create_tensor(tn(LLM_TENSOR_SSM_DT,   "bias",   il), { d_inner }, 0);
            layer.ssm_g_a    = create_tensor(tn(LLM_TENSOR_SSM_G_A,  "weight", il), { n_embd, d_inner }, 0);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", il), { head_dim }, 0);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { d_inner, n_embd }, 0);
        } else {
            if (q_lora_rank > 0) {
                layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", il), { n_embd, q_lora_rank }, 0);
                layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), { q_lora_rank }, 0);
                layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", il), { q_lora_rank, n_head * qk_head_dim }, 0);
            } else {
                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, n_head * qk_head_dim }, 0);
            }
            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", il), { n_embd, kv_lora_rank + qk_rope_head_dim }, 0);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), { kv_lora_rank }, 0);
            // per head: qk_nope rows of K, then v_head_dim rows of V
            layer.wkv_b          = create_tensor(tn(LLM_TENSOR_ATTN_KV_B,      "weight", il), { kv_lora_rank, n_head * (qk_nope_head_dim + v_head_dim) }, 0);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, n_head }, 0);
            layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", il), { n_head * v_head_dim, n_embd }, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, 0);
        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), { n_embd, n_ff }, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff, n_embd }, 0);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", il), { n_embd, n_expert }, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   il), { n_expert }, 0);
            // gate and up of the routed experts: one merged tensor (one expert matmul for both), or two
            layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", il), { n_embd, 2 * hparams.n_ff_exp(), n_expert }, TENSOR_NOT_REQUIRED);
            if (layer.ffn_gate_up_exps == nullptr) {
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, hparams.n_ff_exp(), n_expert }, 0);
                layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", il), { n_embd, hparams.n_ff_exp(), n_expert }, 0);
            }
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", il), { hparams.n_ff_exp(), n_embd, n_expert }, 0);
            layer.ffn_gate_shexp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,  "weight", il), { n_embd, hparams.n_ff_shexp }, 0);
            layer.ffn_up_shexp    = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,    "weight", il), { n_embd, hparams.n_ff_shexp }, 0);
            layer.ffn_down_shexp  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,  "weight", il), { hparams.n_ff_shexp, n_embd }, 0);
        }
    }

    // I8 weights carry one f32 scale per output row, `<name>.scale` (tilde/make_i8c.py)
    auto load_scale = [&](ggml_tensor * w, llm_tensor kind, int il) {
        if (w != nullptr && w->type == GGML_TYPE_I8) {
            // experts [K, N, E]: one scale per row of each expert, [N, E]
            row_scales[w] = w->ne[2] > 1 ? create_tensor(tn(kind, "scale", il), { w->ne[1], w->ne[2] }, 0)
                                         : create_tensor(tn(kind, "scale", il), { w->ne[1] }, 0);
        }
    };
    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        load_scale(layer.wq,             LLM_TENSOR_ATTN_Q,         il);
        load_scale(layer.wk,             LLM_TENSOR_ATTN_K,         il);
        load_scale(layer.wv,             LLM_TENSOR_ATTN_V,         il);
        load_scale(layer.wo,             LLM_TENSOR_ATTN_OUT,       il);
        load_scale(layer.ssm_f_a,        LLM_TENSOR_SSM_F_A,        il);
        load_scale(layer.ssm_g_a,        LLM_TENSOR_SSM_G_A,        il);
        load_scale(layer.ssm_beta,       LLM_TENSOR_SSM_BETA,       il);
        load_scale(layer.wq_a,           LLM_TENSOR_ATTN_Q_A,       il);
        load_scale(layer.wq_b,           LLM_TENSOR_ATTN_Q_B,       il);
        load_scale(layer.wkv_a_mqa,      LLM_TENSOR_ATTN_KV_A_MQA,  il);
        load_scale(layer.wkv_b,          LLM_TENSOR_ATTN_KV_B,      il);
        load_scale(layer.wqkv_gate,      LLM_TENSOR_ATTN_GATE,      il);
        load_scale(layer.ffn_gate,       LLM_TENSOR_FFN_GATE,       il);
        load_scale(layer.ffn_up,         LLM_TENSOR_FFN_UP,         il);
        load_scale(layer.ffn_down,       LLM_TENSOR_FFN_DOWN,       il);
        load_scale(layer.ffn_gate_shexp, LLM_TENSOR_FFN_GATE_SHEXP, il);
        load_scale(layer.ffn_up_shexp,   LLM_TENSOR_FFN_UP_SHEXP,   il);
        load_scale(layer.ffn_down_shexp, LLM_TENSOR_FFN_DOWN_SHEXP, il);
        load_scale(layer.ffn_gate_up_exps, LLM_TENSOR_FFN_GATE_UP_EXPS, il);
        load_scale(layer.ffn_down_exps,    LLM_TENSOR_FFN_DOWN_EXPS,    il);
        // the generic per-tensor scale pass of llama_model must not create these names again
        layer.wq_s             = row_scale(layer.wq);
        layer.wk_s             = row_scale(layer.wk);
        layer.wv_s             = row_scale(layer.wv);
        layer.wo_s             = row_scale(layer.wo);
        layer.wqkv_gate_s      = row_scale(layer.wqkv_gate);
        layer.ssm_beta_s       = row_scale(layer.ssm_beta);
        layer.ffn_gate_s       = row_scale(layer.ffn_gate);
        layer.ffn_up_s         = row_scale(layer.ffn_up);
        layer.ffn_down_s       = row_scale(layer.ffn_down);
        layer.ffn_gate_shexp_s = row_scale(layer.ffn_gate_shexp);
        layer.ffn_up_shexp_s   = row_scale(layer.ffn_up_shexp);
        layer.ffn_down_shexp_s = row_scale(layer.ffn_down_shexp);
        layer.ffn_up_exps_s    = row_scale(layer.ffn_gate_up_exps); // the merged gate_up path takes the up scale
        layer.ffn_down_exps_s  = row_scale(layer.ffn_down_exps);
    }

    // decision head (same tensors as clef)
    const auto * w_memory = ml.get_weight(tn(LLM_TENSOR_DECISION_PROJ_MEMORY, "weight").str().c_str());
    const auto * w_ffn    = ml.get_weight(tn(LLM_TENSOR_DEC_FFN_UP, "weight", 0).str().c_str());
    if (w_memory == nullptr || w_ffn == nullptr) {
        throw std::runtime_error("the decision head is missing");
    }
    const int64_t n_embd_h = w_memory->tensor->ne[1];
    const int64_t n_ff_h   = w_ffn->tensor->ne[1];

    if (n_embd_h % n_head_decision != 0) {
        throw std::runtime_error("invalid width of the decision head");
    }

    auto load_norm = [&](norm & n, llm_tensor type, int64_t size, int il = -1) {
        n.w = il < 0 ? create_tensor(tn(type, "weight"), {size}, 0) : create_tensor(tn(type, "weight", il), {size}, 0);
        n.b = il < 0 ? create_tensor(tn(type, "bias"),   {size}, 0) : create_tensor(tn(type, "bias",   il), {size}, 0);
    };

    auto load_attn = [&](attn & a, llm_tensor q, llm_tensor k, llm_tensor v, llm_tensor o, int il) {
        a.wq = create_tensor(tn(q, "weight", il), {n_embd_h, n_embd_h}, 0);
        a.bq = create_tensor(tn(q, "bias",   il), {n_embd_h}, 0);
        a.wk = create_tensor(tn(k, "weight", il), {n_embd_h, n_embd_h}, 0);
        a.bk = create_tensor(tn(k, "bias",   il), {n_embd_h}, 0);
        a.wv = create_tensor(tn(v, "weight", il), {n_embd_h, n_embd_h}, 0);
        a.bv = create_tensor(tn(v, "bias",   il), {n_embd_h}, 0);
        a.wo = create_tensor(tn(o, "weight", il), {n_embd_h, n_embd_h}, 0);
        a.bo = create_tensor(tn(o, "bias",   il), {n_embd_h}, 0);
    };

    head_layers.resize(n_layer_routing + n_layer_joint);
    for (int il = 0; il < (int) head_layers.size(); ++il) {
        auto & layer = head_layers[il];

        if (il < (int) n_layer_routing) {
            load_norm(layer.cross_norm_kv, LLM_TENSOR_DEC_CROSS_ATTN_NORM_KV, n_embd_h, il);
        } else {
            load_norm(layer.self_norm, LLM_TENSOR_DEC_ATTN_NORM, n_embd_h, il);
            load_attn(layer.self_attn, LLM_TENSOR_DEC_ATTN_Q, LLM_TENSOR_DEC_ATTN_K, LLM_TENSOR_DEC_ATTN_V, LLM_TENSOR_DEC_ATTN_OUT, il);
        }

        load_norm(layer.cross_norm, LLM_TENSOR_DEC_CROSS_ATTN_NORM, n_embd_h, il);
        load_attn(layer.cross_attn, LLM_TENSOR_DEC_CROSS_ATTN_Q, LLM_TENSOR_DEC_CROSS_ATTN_K, LLM_TENSOR_DEC_CROSS_ATTN_V, LLM_TENSOR_DEC_CROSS_ATTN_OUT, il);

        load_norm(layer.ffn_norm, LLM_TENSOR_DEC_FFN_NORM, n_embd_h, il);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "weight", il), {n_embd_h, n_ff_h}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "bias",   il), {n_ff_h}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", il), {n_ff_h, n_embd_h}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "bias",   il), {n_embd_h}, 0);
    }

    load_norm(hidden_norm,         LLM_TENSOR_DECISION_HIDDEN_NORM,         n_embd);
    load_norm(option_summary_norm, LLM_TENSOR_DECISION_OPTION_SUMMARY_NORM, n_embd_h);
    load_norm(field_norm,          LLM_TENSOR_DECISION_FIELD_NORM,          n_embd_h);
    load_norm(option_norm,         LLM_TENSOR_DECISION_OPTION_NORM,         n_embd_h);

    proj_memory          = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_MEMORY,          "weight"), {n_embd, n_embd_h}, 0);
    proj_question        = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_QUESTION,        "weight"), {n_embd, n_embd_h}, 0);
    proj_option_question = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_OPTION_QUESTION, "weight"), {n_embd, n_embd_h}, 0);
    proj_global          = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_GLOBAL,          "weight"), {n_embd, n_embd_h}, 0);
    proj_option_context  = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_OPTION_CONTEXT,  "weight"), {n_embd, n_embd_h}, 0);
    proj_option_lexical  = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_OPTION_LEXICAL,  "weight"), {n_embd, n_embd_h}, 0);

    scales    = create_tensor(tn(LLM_TENSOR_DECISION_SCALES),       {3}, 0);
    type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd_h, 3}, 0);

    scorer       = create_tensor(tn(LLM_TENSOR_DECISION_SCORER,     "weight"), {4 * n_embd_h, n_embd_h}, 0);
    scorer_b     = create_tensor(tn(LLM_TENSOR_DECISION_SCORER,     "bias"),   {n_embd_h}, 0);
    scorer_out   = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT, "weight"), {n_embd_h, 1}, 0);
    scorer_out_b = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT, "bias"),   {1}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_tilde::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

//
// spans read by the head
//

// [start, end) in ubatch token indices
struct tilde_span {
    int32_t start;
    int32_t end;
};

struct tilde_spans {
    std::vector<tilde_span> questions;
    std::vector<int32_t>    question_types; // noul, choice, score
    std::vector<tilde_span> options;
    std::vector<int32_t>    option_question;
    bool valid = false;
};

// a run of entries with the same decision order is one span; an option belongs to the last question before it
// if the batch has no usable order, returns one question with one option on the last token, and valid = false
static tilde_spans tilde_get_spans(const llama_ubatch & ubatch) {
    tilde_spans res;
    std::vector<bool> has_option;

    bool ok = ubatch.decision_order != nullptr && ubatch.token != nullptr && ubatch.n_seqs_unq == 1;

    const int32_t n_tokens = ubatch.n_tokens;
    for (int32_t i = 0; ok && i < n_tokens;) {
        const int32_t order = ubatch.decision_order[i];
        int32_t end = i + 1;
        while (end < n_tokens && ubatch.decision_order[end] == order) {
            end++;
        }
        switch (order) {
            case LLAMA_DECISION_ORDER_NONE:
                break;
            case LLAMA_DECISION_ORDER_QUESTION_NOUL:
            case LLAMA_DECISION_ORDER_QUESTION_CHOICE:
            case LLAMA_DECISION_ORDER_QUESTION_SCORE:
                res.questions.push_back({ i, end });
                res.question_types.push_back(order - LLAMA_DECISION_ORDER_QUESTION_NOUL);
                has_option.push_back(false);
                break;
            case LLAMA_DECISION_ORDER_OPTION:
                ok = !res.questions.empty();
                if (ok) {
                    res.options.push_back({ i, end });
                    res.option_question.push_back((int32_t) res.questions.size() - 1);
                    has_option.back() = true;
                }
                break;
            default:
                ok = false;
        }
        i = end;
    }

    // the head reads the token ids of the spans, they cannot be embeddings
    for (int32_t i = 0; ok && ubatch.is_mixed() && i < n_tokens; i++) {
        ok = !ubatch.type[i] || ubatch.decision_order[i] == LLAMA_DECISION_ORDER_NONE;
    }

    res.valid = ok && !res.questions.empty() && std::find(has_option.begin(), has_option.end(), false) == has_option.end();
    if (!res.valid) {
        res = tilde_spans();
        res.questions.push_back({ n_tokens - 1, n_tokens });
        res.question_types.push_back(0);
        res.options.push_back({ n_tokens - 1, n_tokens });
        res.option_question.push_back(0);
    }
    return res;
}

// Span means without a [n_tokens, n_spans] pooling matrix: the tokens of the spans are gathered, then pooled in
// chunks of consecutive spans. Each chunk is one matmul with a [chunk tokens, chunk spans] matrix of 1/length, so the
// pooling matrices stay small even with long questions and many options.
struct tilde_pool_plan {
    struct chunk {
        int32_t tok0;   // first gathered token of the chunk
        int32_t n_tok;
        int32_t span0;  // first span of the chunk
        int32_t n_span;
    };

    std::vector<int32_t> tokens; // ubatch indices of the gathered tokens, span by span
    std::vector<chunk>   chunks;

    explicit tilde_pool_plan(const std::vector<tilde_span> & spans) {
        // elements of one pooling matrix (16 MiB of f32); a single longer span gets a chunk of its own
        constexpr int64_t budget = 4 << 20;

        chunk cur = { 0, 0, 0, 0 };
        for (size_t i = 0; i < spans.size(); i++) {
            const int32_t len = spans[i].end - spans[i].start;
            if (cur.n_span > 0 && (int64_t) (cur.n_tok + len) * (cur.n_span + 1) > budget) {
                chunks.push_back(cur);
                cur = { (int32_t) tokens.size(), 0, (int32_t) i, 0 };
            }
            for (int32_t t = spans[i].start; t < spans[i].end; t++) {
                tokens.push_back(t);
            }
            cur.n_tok  += len;
            cur.n_span += 1;
        }
        if (cur.n_span > 0) {
            chunks.push_back(cur);
        }
    }

    // pooling matrix of chunk c: [n_tok, n_span], 1/length where the token belongs to the span
    void fill(size_t c, const std::vector<tilde_span> & spans, std::vector<float> & out) const {
        const auto & ch = chunks[c];
        out.assign((size_t) ch.n_tok * ch.n_span, 0.0f);
        int32_t t = 0;
        for (int32_t s = 0; s < ch.n_span; s++) {
            const auto & sp = spans[ch.span0 + s];
            const int32_t len = sp.end - sp.start;
            for (int32_t j = 0; j < len; j++, t++) {
                out[(size_t) s * ch.n_tok + t] = 1.0f / len;
            }
        }
    }
};

class llama_model_tilde::input_decision : public llm_graph_input_i {
public:
    input_decision(const llama_ubatch & ubatch) :
            n_tokens(ubatch.n_tokens),
            spans(tilde_get_spans(ubatch)),
            plan_q(spans.questions),
            plan_o(spans.options) {
    }

    void set_input(const llama_ubatch * ubatch) override {
        const auto cur = tilde_get_spans(*ubatch);
        GGML_ASSERT(cur.questions.size() == spans.questions.size() && cur.options.size() == spans.options.size());

        // the scores are NaN if the batch has a decision order that cannot be used
        const float status_data = cur.valid || ubatch->decision_order == nullptr ? 0.0f : NAN;
        ggml_backend_tensor_set(status, &status_data, 0, ggml_nbytes(status));

        const tilde_pool_plan pq(cur.questions);
        const tilde_pool_plan po(cur.options);
        GGML_ASSERT(pq.tokens.size() == plan_q.tokens.size() && po.tokens.size() == plan_o.tokens.size());

        ggml_backend_tensor_set(q_rows, pq.tokens.data(), 0, ggml_nbytes(q_rows));
        ggml_backend_tensor_set(o_rows, po.tokens.data(), 0, ggml_nbytes(o_rows));

        // token ids of the option tokens (a batch of embeddings has no token ids, and no usable spans)
        std::vector<int32_t> ids(po.tokens.size(), 0);
        if (ubatch->token) {
            for (size_t i = 0; i < ids.size(); i++) {
                ids[i] = ubatch->token[po.tokens[i]];
            }
        }
        ggml_backend_tensor_set(o_ids, ids.data(), 0, ggml_nbytes(o_ids));

        std::vector<float> buf;
        for (size_t c = 0; c < pq.chunks.size(); c++) {
            pq.fill(c, cur.questions, buf);
            ggml_backend_tensor_set(q_pools[c], buf.data(), 0, ggml_nbytes(q_pools[c]));
        }
        for (size_t c = 0; c < po.chunks.size(); c++) {
            po.fill(c, cur.options, buf);
            ggml_backend_tensor_set(o_pools[c], buf.data(), 0, ggml_nbytes(o_pools[c]));
        }

        const size_t n_q = cur.questions.size();
        const size_t n_o = cur.options.size();
        std::vector<float> mask(n_o * n_q, -INFINITY);
        for (size_t i = 0; i < n_o; i++) {
            mask[cur.option_question[i] * n_o + i] = 0.0f;
        }
        ggml_backend_tensor_set(question_type,   cur.question_types.data(),  0, ggml_nbytes(question_type));
        ggml_backend_tensor_set(option_question, cur.option_question.data(), 0, ggml_nbytes(option_question));
        ggml_backend_tensor_set(option_mask,     mask.data(),                0, ggml_nbytes(option_mask));
    }

    bool can_reuse(const llm_graph_params & params) override {
        // the shapes (and the chunking) depend on the spans
        const auto cur = tilde_get_spans(params.ubatch);
        if (cur.questions.size() != spans.questions.size() || cur.options.size() != spans.options.size()) {
            return false;
        }
        const tilde_pool_plan pq(cur.questions);
        const tilde_pool_plan po(cur.options);
        auto same = [](const tilde_pool_plan & a, const tilde_pool_plan & b) {
            if (a.tokens.size() != b.tokens.size() || a.chunks.size() != b.chunks.size()) {
                return false;
            }
            for (size_t i = 0; i < a.chunks.size(); i++) {
                if (a.chunks[i].n_tok != b.chunks[i].n_tok || a.chunks[i].n_span != b.chunks[i].n_span) {
                    return false;
                }
            }
            return true;
        };
        return same(pq, plan_q) && same(po, plan_o);
    }

    ggml_tensor * q_rows          = nullptr; // I32 [n_q_tokens], tokens of the questions
    ggml_tensor * o_rows          = nullptr; // I32 [n_o_tokens], tokens of the options
    ggml_tensor * o_ids           = nullptr; // I32 [n_o_tokens], their token ids
    std::vector<ggml_tensor *> q_pools;      // F32 [chunk tokens, chunk spans]
    std::vector<ggml_tensor *> o_pools;
    ggml_tensor * question_type   = nullptr; // I32 [n_questions]
    ggml_tensor * option_question = nullptr; // I32 [n_options]
    ggml_tensor * option_mask     = nullptr; // F32 [n_options, n_questions], 0 if the option belongs to the question, else -inf
    ggml_tensor * status          = nullptr; // F32 [1], added to the scores: 0, or NaN on invalid input

    const int64_t   n_tokens;
    const tilde_spans     spans;
    const tilde_pool_plan plan_q;
    const tilde_pool_plan plan_o;
};

//
// backbone
//

// Tokens per chunk of the per-token blocks (FFN, MoE). The whole request is one ubatch (the MLA layers attend over
// all of it), so without chunking the MoE intermediates of a 64k request take several GB. Chunks bound them; the
// results are written into one output tensor in place. At most 8192: the CUDA expert-id helper keeps 4 bytes per
// token of a MUL_MAT_ID call in shared memory (48 KiB on older GPUs).
// TILDE_CHUNK_TOKENS overrides the default (tilde-server sets it from the free memory).
static int64_t tilde_chunk_tokens() {
    static const int64_t value = [] {
        const char * env = std::getenv("TILDE_CHUNK_TOKENS");
        const int64_t v = env ? std::atoll(env) : 8192;
        return v <= 0 ? (int64_t) 8192 : std::min<int64_t>(v, 8192);
    }();
    return value;
}

// int8 Q*K^T in the MLA flash attention (TILDE_FA_INT8=0 turns it off)
// int8 P*V in the MLA flash attention (TILDE_FA_PV8=1; off by default)
static bool tilde_fa_pv8() {
    static const bool v = [] {
        const char * env = getenv("TILDE_FA_PV8");
        return env != nullptr && atoi(env) != 0;
    }();
    return v;
}

static bool tilde_fa_int8() {
    static const bool value = [] {
        const char * env = std::getenv("TILDE_FA_INT8");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return value;
}

// y = fn(x) over chunks of the token dimension (dim 1) of x, assembled in order
template <typename F>
static ggml_tensor * tilde_chunked(ggml_context * ctx, ggml_tensor * x, int64_t chunk, F && fn) {
    const int64_t n_tokens = x->ne[1];
    if (chunk <= 0 || n_tokens <= chunk) {
        return fn(x);
    }
    ggml_tensor * out = nullptr;
    for (int64_t t0 = 0; t0 < n_tokens; t0 += chunk) {
        const int64_t n = std::min(chunk, n_tokens - t0);
        ggml_tensor * xc = ggml_view_2d(ctx, x, x->ne[0], n, x->nb[1], t0 * x->nb[1]);
        ggml_tensor * yc = fn(xc);
        GGML_ASSERT(yc->ne[1] == n && ggml_is_contiguous(yc));
        if (out == nullptr) {
            // padded as 1D: a [ne0, n_tokens] pad puts n_tokens in a CUDA grid dimension capped at 65535
            out = ggml_pad(ctx, ggml_reshape_1d(ctx, yc, ggml_nelements(yc)), (int) (yc->ne[0] * (n_tokens - n)), 0, 0, 0);
            out = ggml_reshape_2d(ctx, out, yc->ne[0], n_tokens);
        } else {
            out = ggml_set_2d_inplace(ctx, out, yc, out->nb[1], t0 * out->nb[1]);
        }
    }
    return out;
}

llama_model_tilde::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(static_cast<const llama_model_tilde &>(model_base)) {
    // one request per batch: a batch with several sequences gets NaN scores (tilde_get_spans), like clef

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    auto inp_decision_ptr = std::make_unique<input_decision>(ubatch);
    auto * inp_decision   = inp_decision_ptr.get();
    res->add_input(std::move(inp_decision_ptr));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recr(il)) {
            cur = build_layer_kda(cur, il);
        } else {
            cur = build_layer_mla(cur, inp_pos, il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = tilde_chunked(ctx0, cur, tilde_chunk_tokens(), [&](ggml_tensor * x) {
                return build_ffn(x,
                    layer.ffn_up,   nullptr, model.row_scale(layer.ffn_up),
                    layer.ffn_gate, nullptr, model.row_scale(layer.ffn_gate),
                    layer.ffn_down, nullptr, model.row_scale(layer.ffn_down),
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            });
        } else {
            cur = tilde_chunked(ctx0, cur, tilde_chunk_tokens(), [&](ggml_tensor * cur) {
            ggml_tensor * moe = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il, nullptr, layer.ffn_gate_up_exps, layer.ffn_up_exps_s, nullptr, layer.ffn_down_exps_s);
            ggml_tensor * shared = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, model.row_scale(layer.ffn_up_shexp),
                    layer.ffn_gate_shexp, nullptr, model.row_scale(layer.ffn_gate_shexp),
                    layer.ffn_down_shexp, nullptr, model.row_scale(layer.ffn_down_shexp),
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            return ggml_add(ctx0, moe, shared);
            });
        }
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // the head is always evaluated, so that the graph has the same nodes for every batch
    cur = build_head(cur, inp_decision);

    // row i of the output is the score of option i (padded as 1D, see tilde_chunked)
    cur = ggml_reshape_2d(ctx0, ggml_pad(ctx0, ggml_reshape_1d(ctx0, cur, cur->ne[1]), n_tokens - cur->ne[1], 0, 0, 0), 1, n_tokens);
    cur = ggml_add(ctx0, cur, inp_decision->status);
    cb(cur, "result_decision", -1);

    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_tilde::graph::mm(ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, x);
    if (ggml_tensor * s = model.row_scale(w)) {
        res = ggml_mul(ctx0, res, s);
    }
    return res;
}

// KDA layer over the whole batch: causal, the conv and recurrent states start from zero and are not kept
ggml_tensor * llama_model_tilde::graph::build_layer_kda(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];

    const int64_t n_head   = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner  = n_head * head_dim;
    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t n_seqs   = 1;

    auto conv = [&](ggml_tensor * proj_w, ggml_tensor * conv_w) {
        ggml_tensor * x = mm(proj_w, cur);
        x = ggml_reshape_3d(ctx0, x, d_inner, n_tokens, n_seqs);
        // the conv weight view is expanded first, so that CONCAT, SSM_CONV and SILU are adjacent graph nodes and the
        // backends can fuse them (silu directly on the conv output)
        ggml_tensor * cw     = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
        ggml_build_forward_expand(gf, cw);
        ggml_tensor * zeros  = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_conv - 1, d_inner, n_seqs), 0.0f);
        ggml_tensor * conv_x = ggml_concat(ctx0, zeros, ggml_transpose(ctx0, x), 0);
        ggml_tensor * out    = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_x, cw));
        return ggml_reshape_4d(ctx0, out, head_dim, n_head, n_tokens, n_seqs);
    };

    ggml_tensor * q = conv(layer.wq, layer.ssm_q_conv);
    ggml_tensor * k = conv(layer.wk, layer.ssm_k_conv);
    ggml_tensor * v = conv(layer.wv, layer.ssm_v_conv);

    // safe gate: lower_bound * sigmoid(exp(A_log) * (f_proj(x) + dt_bias)). The broadcast operands are expanded first,
    // so that ADD, MUL, SIGMOID and SCALE are adjacent graph nodes (one fused pass on CUDA)
    ggml_tensor * dt_b = ggml_reshape_3d(ctx0, layer.ssm_dt_b, head_dim, n_head, 1);
    ggml_tensor * a    = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
    ggml_build_forward_expand(gf, dt_b);
    ggml_build_forward_expand(gf, a);
    ggml_tensor * gate = ggml_reshape_3d(ctx0, mm(layer.ssm_f_a, cur), head_dim, n_head, n_tokens);
    gate = ggml_add(ctx0, gate, dt_b);
    gate = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_mul(ctx0, gate, a)), hparams.kda_gate_lower_bound);
    gate = ggml_reshape_4d(ctx0, gate, head_dim, n_head, n_tokens, n_seqs);
    cb(gate, "kda_gate", il);

    ggml_tensor * beta = mm(layer.ssm_beta, cur);
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_head, n_tokens, n_seqs));

    // the norms run on [head_dim, n_head * n_tokens]: a [head_dim, n_head, n_tokens] launch puts n_tokens in a grid
    // dimension that CUDA caps at 65535
    q = ggml_reshape_4d(ctx0, build_gdn_l2_norm(ctx0, ggml_reshape_2d(ctx0, q, head_dim, n_head * n_tokens), hparams.f_norm_rms_eps),
            head_dim, n_head, n_tokens, n_seqs);
    k = ggml_reshape_4d(ctx0, build_gdn_l2_norm(ctx0, ggml_reshape_2d(ctx0, k, head_dim, n_head * n_tokens), hparams.f_norm_rms_eps),
            head_dim, n_head, n_tokens, n_seqs);

    ggml_tensor * state = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim, head_dim, n_head, n_seqs), 0.0f);

    ggml_tensor * out = build_delta_net(q, k, v, gate, beta, state, il).first;
    cb(out, "kda_attn", il);

    ggml_tensor * out_gate = mm(layer.ssm_g_a, cur);
    out_gate = ggml_reshape_3d(ctx0, out_gate, head_dim, n_head, n_tokens);
    out = ggml_reshape_2d(ctx0, ggml_is_contiguous(out) ? out : ggml_cont(ctx0, out), head_dim, n_head * n_tokens);
    out = build_norm(out, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    out = ggml_reshape_3d(ctx0, out, head_dim, n_head, n_tokens);
    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, out_gate));
    cur = mm(layer.wo, ggml_reshape_2d(ctx0, out, d_inner, n_tokens));
    cb(cur, "kda_out", il);

    return cur;
}

// MLA layer, non-causal over the whole batch. K and V are decompressed from the latent (kv_b) instead of attending in
// the latent space: for a long prefill this costs (192 + 128) instead of (576 + 512) per query-key pair and head.
ggml_tensor * llama_model_tilde::graph::build_layer_mla(ggml_tensor * cur, ggml_tensor * inp_pos, int il) {
    const auto & layer = model.layers[il];

    const int64_t n_head           = hparams.n_head();
    const int64_t qk_head_dim      = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim       = hparams.n_embd_head_v_mla();
    const int64_t qk_rope_head_dim = hparams.n_rot();
    const int64_t qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;
    const int64_t kv_lora_rank     = hparams.n_lora_kv;
    const float   kq_scale         = 1.0f / sqrtf((float) qk_head_dim);

    ggml_tensor * attn_input = cur;

    ggml_tensor * q_all;
    if (layer.wq_a) {
        q_all = mm(layer.wq_a, cur);
        q_all = build_norm(q_all, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        q_all = mm(layer.wq_b, q_all);
    } else {
        q_all = mm(layer.wq, cur);
    }
    cb(q_all, "q", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q_all, qk_nope_head_dim, n_head, n_tokens,
            ggml_row_size(q_all->type, qk_head_dim),
            ggml_row_size(q_all->type, qk_head_dim) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q_all, qk_rope_head_dim, n_head, n_tokens,
            ggml_row_size(q_all->type, qk_head_dim),
            ggml_row_size(q_all->type, qk_head_dim) * n_head,
            ggml_row_size(q_all->type, qk_nope_head_dim));

    ggml_tensor * kv_all = mm(layer.wkv_a_mqa, cur);
    ggml_tensor * kv = ggml_view_2d(ctx0, kv_all, kv_lora_rank, n_tokens,
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_all, qk_rope_head_dim, 1, n_tokens,
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
            ggml_row_size(kv_all->type, kv_lora_rank));

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    // [n_head * (qk_nope + v), n_tokens]: per head, K nope rows then V rows
    ggml_tensor * kvb = mm(layer.wkv_b, kv);
    cb(kvb, "kv_b", il);
    const size_t es = ggml_element_size(kvb);
    ggml_tensor * k_nope = ggml_view_3d(ctx0, kvb, qk_nope_head_dim, n_head, n_tokens,
            es * (qk_nope_head_dim + v_head_dim), kvb->nb[1], 0);
    ggml_tensor * v = ggml_view_3d(ctx0, kvb, v_head_dim, n_head, n_tokens,
            es * (qk_nope_head_dim + v_head_dim), kvb->nb[1], es * qk_nope_head_dim);

    // concatenations as [dim, n_head * n_tokens] (a 3D launch caps n_tokens at 65535 on CUDA)
    auto flat = [&](ggml_tensor * t) {
        return ggml_view_2d(ctx0, t, t->ne[0], t->ne[1] * t->ne[2], t->nb[1], 0);
    };
    ggml_tensor * k_pe_rep = ggml_repeat_4d(ctx0, k_pe, qk_rope_head_dim, n_head, n_tokens, 1);
    ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_concat(ctx0, flat(q_nope), flat(q_pe), 0), qk_head_dim, n_head, n_tokens);
    ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_concat(ctx0, flat(k_nope), flat(k_pe_rep), 0), qk_head_dim, n_head, n_tokens);
    v = ggml_cont(ctx0, v);
    cb(q, "q_full", il);
    cb(k, "k_full", il);

    // no mask: one sequence, every token attends to every token. Without flash attention the scores would be
    // materialized as [n_tokens, n_tokens, n_head]: refuse long requests instead of trying to allocate them.
    if (!cparams.flash_attn && n_tokens > 8192) {
        throw std::runtime_error("tilde: requests over 8192 tokens need flash attention (-fa on)");
    }
    if (cparams.flash_attn) {
        ggml_tensor * fq = ggml_permute(ctx0, q, 0, 2, 1, 3);
        ggml_tensor * fk = ggml_cast(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor * fv = ggml_cast(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3), GGML_TYPE_F16);
        cur = ggml_flash_attn_ext(ctx0, fq, fk, fv, nullptr, kq_scale, 0.0f, 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        if (tilde_fa_int8()) {
            // Q*K^T may run in int8 (SageAttention-style; CUDA, Ampere and newer: fattn-i8.cu). Other backends ignore
            // the hint.
            ggml_prec_set_src(cur, GGML_PREC_Q8, 0);
            ggml_prec_set_src(cur, GGML_PREC_Q8, 1);
            if (tilde_fa_pv8()) {
                ggml_prec_set_src(cur, GGML_PREC_Q8, 2); // int8 P*V too (speed preset, measured separately)
            }
        }
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    } else {
        cur = build_attn_mha(q, k, v, nullptr, nullptr, nullptr, nullptr, n_tokens, kq_scale, il);
    }
    cb(cur, "mla_attn", il);

    ggml_tensor * attn_gate = mm(layer.wqkv_gate, attn_input);
    attn_gate = ggml_sigmoid(ctx0, ggml_reshape_3d(ctx0, attn_gate, 1, n_head, n_tokens));
    cur = ggml_reshape_3d(ctx0, cur, v_head_dim, n_head, n_tokens);
    cur = ggml_mul(ctx0, cur, attn_gate);
    cur = mm(layer.wo, ggml_reshape_2d(ctx0, cur, v_head_dim * n_head, n_tokens));
    cb(cur, "mla_out", il);

    return cur;
}

//
// head
//

ggml_tensor * llama_model_tilde::graph::build_head_norm(ggml_tensor * cur, const norm & n) {
    return build_norm(cur, n.w, n.b, LLM_NORM, -1);
}

// nn.MultiheadAttention without a mask; q: [n_embd_h, n_q], kv: [n_embd_h, n_kv]
ggml_tensor * llama_model_tilde::graph::build_head_attn(ggml_tensor * q, ggml_tensor * kv, const attn & a) {
    const int64_t n_embd_h = q->ne[0];
    const int64_t n_head_h = model.n_head_decision;
    const int64_t d_head   = n_embd_h / n_head_h;

    ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wq, q),  a.bq);
    ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wk, kv), a.bk);
    ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wv, kv), a.bv);

    Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head_h, q->ne[1]);
    Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head_h, kv->ne[1]);
    Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head_h, kv->ne[1]);

    // flash attention when enabled: the memory can be the whole request (64k tokens)
    // (the head lives with the output layer; the last layer stands for its device in the fused-op checks)
    ggml_tensor * cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr, kv->ne[1],
            1.0f / sqrtf(float(d_head)), n_layer - 1);

    return ggml_add(ctx0, ggml_mul_mat(ctx0, a.wo, cur), a.bo);
}

ggml_tensor * llama_model_tilde::graph::build_head_ffn(ggml_tensor * cur, const head_layer & layer) {
    cur = build_head_norm(cur, layer.ffn_norm);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_up, cur), layer.ffn_up_b);
    cur = ggml_gelu_erf(ctx0, cur);
    return ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_down, cur), layer.ffn_down_b);
}

// mean of each span of rows of `table` (gathered by `ids`, I32 [n_gathered], span by span);
// returns [n_embd, n_spans]. Gathered per chunk, so the copies stay chunk-sized.
ggml_tensor * llama_model_tilde::graph::build_span_means(ggml_tensor * table, ggml_tensor * ids, ggml_tensor * const * pools, size_t n_pools) {
    ggml_tensor * res = nullptr;
    int64_t tok0 = 0;
    for (size_t c = 0; c < n_pools; c++) {
        ggml_tensor * pool = pools[c];
        const int64_t n_tok = pool->ne[0];
        ggml_tensor * idx  = ggml_view_1d(ctx0, ids, n_tok, tok0 * ggml_element_size(ids));
        ggml_tensor * part = ggml_get_rows(ctx0, table, idx);                                     // [n_embd, n_tok]
        part = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, part)), pool);            // [n_embd, chunk spans]
        res = res ? ggml_concat(ctx0, res, part, 1) : part;
        tok0 += n_tok;
    }
    GGML_ASSERT(tok0 == ids->ne[0]);
    return res;
}

// ref: tilda/clef_joint_fast.py (FastJointSchemaHead.forward) and JointSchemaHead of Cloudflare Clef
ggml_tensor * llama_model_tilde::graph::build_head(ggml_tensor * hidden, input_decision * inp) {
    const int64_t n_questions = inp->spans.questions.size();
    const int64_t n_options   = inp->spans.options.size();

    inp->q_rows          = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->plan_q.tokens.size());
    inp->o_rows          = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->plan_o.tokens.size());
    inp->o_ids           = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->plan_o.tokens.size());
    inp->question_type   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_questions);
    inp->option_question = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_options);
    inp->option_mask     = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_options, n_questions);
    inp->status          = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { inp->q_rows, inp->o_rows, inp->o_ids, inp->question_type, inp->option_question, inp->option_mask, inp->status }) {
        ggml_set_input(t);
    }
    for (const auto & ch : inp->plan_q.chunks) {
        inp->q_pools.push_back(ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ch.n_tok, ch.n_span));
        ggml_set_input(inp->q_pools.back());
    }
    for (const auto & ch : inp->plan_o.chunks) {
        inp->o_pools.push_back(ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ch.n_tok, ch.n_span));
        ggml_set_input(inp->o_pools.back());
    }

    hidden = build_head_norm(hidden, model.hidden_norm);
    cb(hidden, "decision_hidden", -1);

    ggml_tensor * memory = ggml_mul_mat(ctx0, model.proj_memory, hidden); // [n_embd_h, n_tokens]
    const int64_t n_embd_h = memory->ne[0];

    // hidden state of the last token
    ggml_tensor * global = ggml_view_2d(ctx0, hidden, n_embd, 1, hidden->nb[1], (n_tokens - 1) * hidden->nb[1]);

    // mean of the hidden states over each span
    ggml_tensor * question = build_span_means(hidden, inp->q_rows, inp->q_pools.data(), inp->q_pools.size());
    ggml_tensor * opt_ctx  = build_span_means(hidden, inp->o_rows, inp->o_pools.data(), inp->o_pools.size());

    // mean of the output embeddings of the tokens of each option
    ggml_tensor * lexical = build_span_means(model.output, inp->o_ids, inp->o_pools.data(), inp->o_pools.size());
    cb(question, "decision_question", -1);
    cb(lexical,  "decision_lexical", -1);

    // one query per option
    ggml_tensor * options = ggml_mul_mat(ctx0, model.proj_option_context, opt_ctx);
    options = ggml_add(ctx0, options, ggml_mul_mat(ctx0, model.proj_option_lexical, lexical));
    options = ggml_add(ctx0, options, ggml_get_rows(ctx0, ggml_mul_mat(ctx0, model.proj_option_question, question), inp->option_question));

    // the options read the request
    for (uint32_t il = 0; il < model.n_layer_routing; ++il) {
        const auto & layer = model.head_layers[il];

        ggml_tensor * cur = build_head_attn(
                build_head_norm(options, layer.cross_norm),
                build_head_norm(memory,  layer.cross_norm_kv),
                layer.cross_attn);
        options = ggml_add(ctx0, options, cur);
        options = ggml_add(ctx0, options, build_head_ffn(options, layer));
    }
    cb(options, "decision_options", -1);

    // one vector per question: its text, a summary of its options, the end of the request and its type
    ggml_tensor * fields = ggml_mul_mat(ctx0, model.proj_question, question); // [n_embd_h, n_questions]
    {
        // each question weights its own options
        ggml_tensor * weights = ggml_mul_mat(ctx0, options, fields); // [n_options, n_questions]
        weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_embd_h)));
        weights = ggml_soft_max(ctx0, ggml_add(ctx0, weights, inp->option_mask));

        ggml_tensor * summary = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, options)), weights);

        fields = ggml_add(ctx0, fields, build_head_norm(summary, model.option_summary_norm));
        fields = ggml_add(ctx0, fields, ggml_mul_mat(ctx0, model.proj_global, global));
        fields = ggml_add(ctx0, fields, ggml_get_rows(ctx0, model.type_embd, inp->question_type));
    }

    // the questions read each other and the request
    for (uint32_t il = model.n_layer_routing; il < model.head_layers.size(); ++il) {
        const auto & layer = model.head_layers[il];

        ggml_tensor * cur = build_head_norm(fields, layer.self_norm);
        fields = ggml_add(ctx0, fields, build_head_attn(cur, cur, layer.self_attn));

        cur = build_head_norm(fields, layer.cross_norm);
        fields = ggml_add(ctx0, fields, build_head_attn(cur, memory, layer.cross_attn));

        fields = ggml_add(ctx0, fields, build_head_ffn(fields, layer));
    }
    fields = build_head_norm(fields, model.field_norm);
    cb(fields, "decision_fields", -1);

    const float eps = 1e-12f; // torch.nn.functional.normalize

    // prior: the output embeddings of the option against the question and the end of the request
    ggml_tensor * anchor = ggml_l2_norm(ctx0, ggml_add(ctx0, question, global), eps);
    anchor = ggml_get_rows(ctx0, anchor, inp->option_question);
    ggml_tensor * prior = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_l2_norm(ctx0, lexical, eps), anchor)); // [1, n_options]

    // joint: each option against the vector of its question
    options = build_head_norm(options, model.option_norm);
    ggml_tensor * field = ggml_get_rows(ctx0, fields, inp->option_question); // [n_embd_h, n_options]

    ggml_tensor * cosine = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_l2_norm(ctx0, field, eps), ggml_l2_norm(ctx0, options, eps)));

    ggml_tensor * features = ggml_concat(ctx0, field, options, 0);
    features = ggml_concat(ctx0, features, ggml_mul(ctx0, field, options), 0);
    features = ggml_concat(ctx0, features, ggml_abs(ctx0, ggml_sub(ctx0, field, options)), 0);

    ggml_tensor * residual = ggml_add(ctx0, ggml_mul_mat(ctx0, model.scorer, features), model.scorer_b);
    residual = ggml_gelu_erf(ctx0, residual);
    residual = ggml_add(ctx0, ggml_mul_mat(ctx0, model.scorer_out, residual), model.scorer_out_b); // [1, n_options]

    auto scale = [&](int i) {
        return ggml_view_1d(ctx0, model.scales, 1, i * ggml_element_size(model.scales));
    };

    ggml_tensor * joint = ggml_add(ctx0, ggml_mul(ctx0, cosine, scale(1)), residual);

    return ggml_add(ctx0, ggml_mul(ctx0, prior, scale(0)), ggml_mul(ctx0, joint, scale(2)));
}
