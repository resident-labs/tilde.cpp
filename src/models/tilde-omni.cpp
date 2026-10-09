#include "models.h"

#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

// Tilde Omni: TLive-Omni-4B's Qwen3.5 text decoder (3 Gated DeltaNet layers : 1 gated full-attention layer) trained
// with the Tilde joint decision head. The whole request is one ubatch and no memory module is used: the Gated
// DeltaNet layers start from zero states and stay causal, the full-attention layers attend over the whole request
// (attention_mode bidirectional_mla). Rotary positions are 1D (the Qwen3.5 interleaved M-RoPE with the same position
// in every section, i.e. NEOX rope over the first n_rot dimensions).
//
// The head follows tilda.qwen35.model.head.PackedJointSchemaHead (FastJointSchemaHead's parameters). The fitted
// temperature is folded into the stored scales by the converter, so the outputs are calibrated logits.

// width of the embeddings output: the graph writes its outputs as one flat array of n_tokens * TILDE_OMNI_OUT floats,
// the option logits first, then the point fields' outputs (see build_points)
static constexpr uint32_t TILDE_OMNI_OUT = 4;

// input embeddings carry this many floats after the n_embd of the LM input: the point head's sub-patch keys of image
// tokens (4 x 128, computed with the image embeddings by the mmproj; zero for any other entry)
static constexpr uint32_t TILDE_OMNI_SIDE = 512;

void llama_model_tilde_omni::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DECISION_ROUTING_BLOCK_COUNT, n_layer_routing);
    ml.get_key(LLM_KV_DECISION_BLOCK_COUNT,         n_layer_joint);
    ml.get_key(LLM_KV_DECISION_HEAD_COUNT,          n_head_decision);

    if (n_head_decision == 0 || n_layer_routing > LLAMA_MAX_LAYERS || n_layer_joint > LLAMA_MAX_LAYERS) {
        throw std::runtime_error("invalid size of the decision head");
    }
    if (hparams.n_layer_nextn != 0) {
        throw std::runtime_error("tilde_omni: MTP layers are not supported");
    }

    // used by the head
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    hparams.n_embd_out_impl = TILDE_OMNI_OUT;
    hparams.n_embd_inp_impl = hparams.n_embd + TILDE_OMNI_SIDE;
}

void llama_model_tilde_omni::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    // I8 weights carry one f32 scale per output row, `<name>.scale` (tilde/make_i8c.py)
    auto load_scale = [&](ggml_tensor * w, llm_tensor kind, int il) {
        if (w != nullptr && w->type == GGML_TYPE_I8) {
            row_scales[w] = create_tensor(tn(kind, "scale", il), { w->ne[1] }, 0);
        }
    };
    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        load_scale(layer.wq,        LLM_TENSOR_ATTN_Q,    il);
        load_scale(layer.wk,        LLM_TENSOR_ATTN_K,    il);
        load_scale(layer.wv,        LLM_TENSOR_ATTN_V,    il);
        load_scale(layer.wqkv,      LLM_TENSOR_ATTN_QKV,  il);
        load_scale(layer.wo,        LLM_TENSOR_ATTN_OUT,  il);
        load_scale(layer.wqkv_gate, LLM_TENSOR_ATTN_GATE, il);
        load_scale(layer.ssm_beta,  LLM_TENSOR_SSM_BETA,  il);
        load_scale(layer.ssm_alpha, LLM_TENSOR_SSM_ALPHA, il);
        load_scale(layer.ssm_out,   LLM_TENSOR_SSM_OUT,   il);
        load_scale(layer.ffn_gate,  LLM_TENSOR_FFN_GATE,  il);
        load_scale(layer.ffn_up,    LLM_TENSOR_FFN_UP,    il);
        load_scale(layer.ffn_down,  LLM_TENSOR_FFN_DOWN,  il);
        // the generic per-tensor scale pass of llama_model must not create these names again
        layer.wq_s        = row_scale(layer.wq);
        layer.wk_s        = row_scale(layer.wk);
        layer.wv_s        = row_scale(layer.wv);
        layer.wqkv_s      = row_scale(layer.wqkv);
        layer.wo_s        = row_scale(layer.wo);
        layer.wqkv_gate_s = row_scale(layer.wqkv_gate);
        layer.ssm_beta_s  = row_scale(layer.ssm_beta);
        layer.ssm_alpha_s = row_scale(layer.ssm_alpha);
        layer.ssm_out_s   = row_scale(layer.ssm_out);
        layer.ffn_gate_s  = row_scale(layer.ffn_gate);
        layer.ffn_up_s    = row_scale(layer.ffn_up);
        layer.ffn_down_s  = row_scale(layer.ffn_down);
    }

    // decision head (same tensors as tilde)
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
    // noul, choice, score, point (the point head's field type)
    type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd_h, 4}, 0);

    scorer       = create_tensor(tn(LLM_TENSOR_DECISION_SCORER,     "weight"), {4 * n_embd_h, n_embd_h}, 0);
    scorer_b     = create_tensor(tn(LLM_TENSOR_DECISION_SCORER,     "bias"),   {n_embd_h}, 0);
    scorer_out   = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT, "weight"), {n_embd_h, 1}, 0);
    scorer_out_b = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT, "bias"),   {1}, 0);

    // point head: KEY_WIDTH 256, SUB_KEY_WIDTH 128, query from a field vector, 3 x 3 locations per sub-patch
    const int64_t n_key  = 256;
    const int64_t n_sub  = TILDE_OMNI_SIDE / 4;
    const int64_t n_pair = n_embd_h + n_embd;
    load_norm(point.query_norm, LLM_TENSOR_POINT_QUERY_NORM, n_embd_h);
    point.query       = create_tensor(tn(LLM_TENSOR_POINT_QUERY,     "weight"), {n_embd_h, n_key}, 0);
    load_norm(point.key_norm, LLM_TENSOR_POINT_KEY_NORM, n_embd);
    point.key         = create_tensor(tn(LLM_TENSOR_POINT_KEY,       "weight"), {n_embd, n_key}, 0);
    load_norm(point.pair_norm, LLM_TENSOR_POINT_PAIR_NORM, n_pair);
    point.sub_query   = create_tensor(tn(LLM_TENSOR_POINT_SUB_QUERY, "weight"), {n_pair, n_sub}, 0);
    point.size_up     = create_tensor(tn(LLM_TENSOR_POINT_SIZE_UP,   "weight"), {n_pair, 256}, 0);
    point.size_up_b   = create_tensor(tn(LLM_TENSOR_POINT_SIZE_UP,   "bias"),   {256}, 0);
    point.size_down   = create_tensor(tn(LLM_TENSOR_POINT_SIZE_DOWN, "weight"), {256, 2}, 0);
    point.size_down_b = create_tensor(tn(LLM_TENSOR_POINT_SIZE_DOWN, "bias"),   {2}, 0);
    point.loc_up      = create_tensor(tn(LLM_TENSOR_POINT_LOC_UP,    "weight"), {2 * n_sub, 256}, 0);
    point.loc_up_b    = create_tensor(tn(LLM_TENSOR_POINT_LOC_UP,    "bias"),   {256}, 0);
    point.loc_down    = create_tensor(tn(LLM_TENSOR_POINT_LOC_DOWN,  "weight"), {256, 9}, 0);
    point.loc_down_b  = create_tensor(tn(LLM_TENSOR_POINT_LOC_DOWN,  "bias"),   {9}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_tilde_omni::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

//
// spans read by the head
//

// [start, end) in ubatch token indices
struct tilde_omni_span {
    int32_t start;
    int32_t end;
};

// a point field: its question (type 3, the point type), its only option (the <point none> marker) and its image
struct tilde_omni_point {
    int32_t question;
    int32_t option;
    int32_t image;
};

struct tilde_omni_spans {
    std::vector<tilde_omni_span>  questions;
    std::vector<int32_t>          question_types; // noul, choice, score, point
    std::vector<tilde_omni_span>  options;
    std::vector<int32_t>          option_question;
    std::vector<tilde_omni_span>  images;         // by image index; start < 0: no such image
    std::vector<tilde_omni_point> points;
    bool valid = false;

    int32_t image_tokens(const tilde_omni_point & p) const {
        return images[p.image].end - images[p.image].start;
    }
};

// floats written per point field after its image's token logits: none + tokens, 4 sub-patch logits, 4 x 9 location
// logits, 2 log2 sizes, all at the most likely image token
static constexpr int32_t TILDE_OMNI_POINT_EXTRA = 1 + 4 + 36 + 2;

static constexpr uint32_t TILDE_OMNI_POINT_TYPE = 3;

// a run of entries with the same decision order is one span; an option belongs to the last question before it
// if the batch has no usable order, returns one question with one option on the last token, and valid = false
static tilde_omni_spans tilde_omni_get_spans(const llama_ubatch & ubatch, int64_t n_out) {
    tilde_omni_spans res;
    std::vector<bool> has_option;

    bool ok = ubatch.decision_order != nullptr && ubatch.n_seqs_unq == 1;

    const int32_t n_tokens = ubatch.n_tokens;
    for (int32_t i = 0; ok && i < n_tokens;) {
        const int32_t order = ubatch.decision_order[i];
        int32_t end = i + 1;
        while (end < n_tokens && ubatch.decision_order[end] == order) {
            end++;
        }
        if (order == LLAMA_DECISION_ORDER_NONE) {
            // not read by the head
        } else if (order >= LLAMA_DECISION_ORDER_QUESTION_NOUL && order <= LLAMA_DECISION_ORDER_QUESTION_SCORE) {
            res.questions.push_back({ i, end });
            res.question_types.push_back(order - LLAMA_DECISION_ORDER_QUESTION_NOUL);
            has_option.push_back(false);
        } else if (order == LLAMA_DECISION_ORDER_OPTION) {
            ok = !res.questions.empty();
            if (ok) {
                const int32_t q = (int32_t) res.questions.size() - 1;
                if (!res.points.empty() && res.points.back().question == q) {
                    // a point question has exactly one option, its <point none> marker (one token)
                    ok = !has_option.back() && end - i == 1;
                    res.points.back().option = (int32_t) res.options.size();
                }
                res.options.push_back({ i, end });
                res.option_question.push_back(q);
                has_option.back() = true;
            }
        } else if (order >= LLAMA_DECISION_ORDER_IMAGE && order < LLAMA_DECISION_ORDER_IMAGE + LLAMA_DECISION_ORDER_MAX_IMAGES) {
            const int32_t k = order - LLAMA_DECISION_ORDER_IMAGE;
            if ((int32_t) res.images.size() <= k) {
                res.images.resize(k + 1, { -1, -1 });
            }
            ok = res.images[k].start < 0; // one run per image
            res.images[k] = { i, end };
        } else if (order >= LLAMA_DECISION_ORDER_QUESTION_POINT && order < LLAMA_DECISION_ORDER_QUESTION_POINT + LLAMA_DECISION_ORDER_MAX_IMAGES) {
            res.points.push_back({ (int32_t) res.questions.size(), -1, order - LLAMA_DECISION_ORDER_QUESTION_POINT });
            res.questions.push_back({ i, end });
            res.question_types.push_back(TILDE_OMNI_POINT_TYPE);
            has_option.push_back(false);
        } else {
            ok = false;
        }
        i = end;
    }

    // the head reads the token ids of the options (lexical vectors), they cannot be embeddings
    for (int32_t i = 0; ok && i < n_tokens; i++) {
        const bool is_embd = ubatch.token == nullptr || (ubatch.is_mixed() && ubatch.type[i]);
        ok = !is_embd || ubatch.decision_order[i] != LLAMA_DECISION_ORDER_OPTION;
    }

    // every point field needs its image, and the outputs must fit the embeddings output
    int64_t n_needed = (int64_t) res.options.size();
    for (const auto & pt : res.points) {
        ok = ok && pt.option >= 0 && pt.image < (int32_t) res.images.size() && res.images[pt.image].start >= 0;
        if (ok) {
            n_needed += res.image_tokens(pt) + TILDE_OMNI_POINT_EXTRA;
        }
    }
    ok = ok && n_needed <= n_out;

    res.valid = ok && !res.questions.empty() && std::find(has_option.begin(), has_option.end(), false) == has_option.end();
    if (!res.valid) {
        res = tilde_omni_spans();
        res.questions.push_back({ n_tokens - 1, n_tokens });
        res.question_types.push_back(0);
        res.options.push_back({ n_tokens - 1, n_tokens });
        res.option_question.push_back(0);
    }
    return res;
}

// Span means without a [n_tokens, n_spans] pooling matrix: the tokens of the spans are gathered, then pooled in
// chunks of consecutive spans (one matmul with a [chunk tokens, chunk spans] matrix of 1/length per chunk).
struct tilde_omni_pool_plan {
    struct chunk {
        int32_t tok0;   // first gathered token of the chunk
        int32_t n_tok;
        int32_t span0;  // first span of the chunk
        int32_t n_span;
    };

    std::vector<int32_t> tokens; // ubatch indices of the gathered tokens, span by span
    std::vector<chunk>   chunks;

    explicit tilde_omni_pool_plan(const std::vector<tilde_omni_span> & spans) {
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
    void fill(size_t c, const std::vector<tilde_omni_span> & spans, std::vector<float> & out) const {
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

    bool same_shape(const tilde_omni_pool_plan & other) const {
        if (tokens.size() != other.tokens.size() || chunks.size() != other.chunks.size()) {
            return false;
        }
        for (size_t i = 0; i < chunks.size(); i++) {
            if (chunks[i].n_tok != other.chunks[i].n_tok || chunks[i].n_span != other.chunks[i].n_span) {
                return false;
            }
        }
        return true;
    }
};

class llama_model_tilde_omni::input_decision : public llm_graph_input_i {
public:
    input_decision(const llama_ubatch & ubatch) :
            n_tokens(ubatch.n_tokens),
            spans(tilde_omni_get_spans(ubatch, n_out(ubatch))),
            plan_q(spans.questions),
            plan_o(spans.options) {
    }

    static int64_t n_out(const llama_ubatch & ubatch) {
        return (int64_t) TILDE_OMNI_OUT * ubatch.n_tokens;
    }

    // the graph depends on the number of point fields and on the token count of each one's image
    static bool same_points(const tilde_omni_spans & a, const tilde_omni_spans & b) {
        if (a.points.size() != b.points.size()) {
            return false;
        }
        for (size_t i = 0; i < a.points.size(); i++) {
            if (a.points[i].question != b.points[i].question || a.image_tokens(a.points[i]) != b.image_tokens(b.points[i])) {
                return false;
            }
        }
        return true;
    }

    void set_input(const llama_ubatch * ubatch) override {
        const auto cur = tilde_omni_get_spans(*ubatch, n_out(*ubatch));
        GGML_ASSERT(cur.questions.size() == spans.questions.size() && cur.options.size() == spans.options.size());
        GGML_ASSERT(same_points(cur, spans));

        // the scores are NaN if the batch has a decision order that cannot be used
        const float status_data = cur.valid || ubatch->decision_order == nullptr ? 0.0f : NAN;
        ggml_backend_tensor_set(status, &status_data, 0, ggml_nbytes(status));

        const tilde_omni_pool_plan pq(cur.questions);
        const tilde_omni_pool_plan po(cur.options);
        GGML_ASSERT(pq.same_shape(plan_q) && po.same_shape(plan_o));

        ggml_backend_tensor_set(q_rows, pq.tokens.data(), 0, ggml_nbytes(q_rows));
        ggml_backend_tensor_set(o_rows, po.tokens.data(), 0, ggml_nbytes(o_rows));

        // token ids of the option tokens (0 where the batch has no token ids: the result is NaN then)
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

        // point fields: the rows the point head reads, the <point none> marker then the image's tokens
        for (size_t i = 0; i < cur.points.size(); i++) {
            const auto & pt  = cur.points[i];
            const auto & img = cur.images[pt.image];
            std::vector<int32_t> rows;
            rows.push_back(cur.options[pt.option].start);
            for (int32_t t = img.start; t < img.end; t++) {
                rows.push_back(t);
            }
            ggml_backend_tensor_set(point_rows[i], rows.data(), 0, ggml_nbytes(point_rows[i]));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        // the shapes (and the chunking) depend on the spans
        const auto cur = tilde_omni_get_spans(params.ubatch, n_out(params.ubatch));
        if (cur.questions.size() != spans.questions.size() || cur.options.size() != spans.options.size() || !same_points(cur, spans)) {
            return false;
        }
        return tilde_omni_pool_plan(cur.questions).same_shape(plan_q) && tilde_omni_pool_plan(cur.options).same_shape(plan_o);
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
    std::vector<ggml_tensor *> point_rows;   // I32 [1 + image tokens] per point field

    const int64_t              n_tokens;
    const tilde_omni_spans     spans;
    const tilde_omni_pool_plan plan_q;
    const tilde_omni_pool_plan plan_o;
};

// Input embeddings: token rows from the token table; the embedding rows of a mixed batch (media) are set over them.
// Only those rows are uploaded, and their 512 side values (the point head's sub-patch keys) only when the batch has
// point fields (llm_graph_context::build_inp_embd would pad every token row to n_embd_inp, on the host).
class llama_model_tilde_omni_input_embd : public llm_graph_input_i {
public:
    static bool is_embd(const llama_ubatch & ubatch, uint32_t i) {
        return ubatch.token == nullptr || (ubatch.is_mixed() && ubatch.type[i]);
    }

    static int64_t n_embd_rows(const llama_ubatch & ubatch) {
        int64_t n = 0;
        for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
            n += is_embd(ubatch, i);
        }
        return n;
    }

    explicit llama_model_tilde_omni_input_embd(const llama_ubatch & ubatch) :
            n_tokens(ubatch.n_tokens), n_e(n_embd_rows(ubatch)) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT((int64_t) ubatch->n_tokens == n_tokens && n_embd_rows(*ubatch) == n_e);
        std::vector<int32_t> ids(n_tokens, 0);
        std::vector<int64_t> slots;
        for (int64_t i = 0; i < n_tokens; i++) {
            if (is_embd(*ubatch, i)) {
                slots.push_back(i);
            } else {
                ids[i] = ubatch->token[i];
            }
        }
        ggml_backend_tensor_set(tokens, ids.data(), 0, ggml_nbytes(tokens));
        if (n_e == 0) {
            return;
        }
        GGML_ASSERT(ubatch->embd != nullptr);
        const int64_t n_lm   = embd_lm->ne[0];
        const int64_t n_side = TILDE_OMNI_SIDE;
        const int64_t n_row  = n_lm + n_side; // n_embd_inp
        std::vector<float> lm((size_t) n_lm * n_e), side(embd_side ? (size_t) n_side * n_e : 0);
        for (int64_t r = 0; r < n_e; r++) {
            const float * src = ubatch->embd + (size_t) slots[r] * n_row;
            std::copy(src, src + n_lm, lm.begin() + (size_t) r * n_lm);
            if (embd_side) {
                std::copy(src + n_lm, src + n_row, side.begin() + (size_t) r * n_side);
            }
        }
        ggml_backend_tensor_set(embd_lm,   lm.data(),    0, ggml_nbytes(embd_lm));
        ggml_backend_tensor_set(embd_slots, slots.data(), 0, ggml_nbytes(embd_slots));
        if (embd_side) {
            ggml_backend_tensor_set(embd_side, side.data(), 0, ggml_nbytes(embd_side));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return (int64_t) params.ubatch.n_tokens == n_tokens && n_embd_rows(params.ubatch) == n_e;
    }

    ggml_tensor * tokens     = nullptr; // I32 [n_tokens], 0 at embedding rows
    ggml_tensor * embd_lm    = nullptr; // F32 [n_embd, n_e]
    ggml_tensor * embd_side  = nullptr; // F32 [TILDE_OMNI_SIDE, n_e], only with point fields
    ggml_tensor * embd_slots = nullptr; // I64 [n_e], the batch rows of the embeddings

    const int64_t n_tokens;
    const int64_t n_e;
};

//
// backbone
//

// Tokens per chunk of the FFN: the whole request is one ubatch (the full-attention layers attend over all of it), so
// without chunking the 9216-wide FFN intermediates of a 64k request take several GB. TILDE_CHUNK_TOKENS overrides it.
static int64_t tilde_omni_chunk_tokens() {
    static const int64_t value = [] {
        const char * env = std::getenv("TILDE_CHUNK_TOKENS");
        const int64_t v = env ? std::atoll(env) : 8192;
        return v <= 0 ? (int64_t) 8192 : v;
    }();
    return value;
}

// resid += fn(x), chunk by chunk over the tokens, in place (see tilde_residual_chunked in tilde.cpp). Returns a view of
// resid that comes after all of its chunks are written.
template <typename F>
static ggml_tensor * tilde_omni_residual_chunked(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * resid, ggml_tensor * x,
        int64_t chunk, F && fn) {
    const int64_t n_tokens = x->ne[1];
    ggml_build_forward_expand(gf, x);
    if (chunk <= 0 || n_tokens <= chunk) {
        ggml_tensor * r = ggml_view_2d(ctx, resid, resid->ne[0], n_tokens, resid->nb[1], 0);
        ggml_build_forward_expand(gf, r);
        ggml_tensor * y = ggml_add_inplace(ctx, r, fn(x));
        ggml_build_forward_expand(gf, y);
        return y;
    }
    // every in place result needs a consumer, or the graph allocator keeps resid allocated for the rest of the graph:
    // each chunk's rows are a view of the previous chunk's result, and the returned full view takes the last one as
    // an extra (ordering) source
    ggml_tensor * prev = nullptr;
    int64_t prev_t0 = 0;
    for (int64_t t0 = 0; t0 < n_tokens; t0 += chunk) {
        const int64_t n = std::min(chunk, n_tokens - t0);
        ggml_tensor * r = prev == nullptr
            ? ggml_view_2d(ctx, resid, resid->ne[0], n, resid->nb[1], t0 * resid->nb[1])
            : ggml_view_2d(ctx, prev,  resid->ne[0], n, resid->nb[1], (t0 - prev_t0) * resid->nb[1]);
        ggml_tensor * xc = ggml_view_2d(ctx, x, x->ne[0], n, x->nb[1], t0 * x->nb[1]);
        ggml_build_forward_expand(gf, r);
        prev = ggml_add_inplace(ctx, r, fn(xc));
        prev_t0 = t0;
        ggml_build_forward_expand(gf, prev);
    }
    ggml_tensor * out = ggml_view_2d(ctx, resid, resid->ne[0], n_tokens, resid->nb[1], 0);
    out->src[1] = prev;
    return out;
}

llama_model_tilde_omni::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(static_cast<const llama_model_tilde_omni &>(model_base)) {
    // one request per batch: a batch with several sequences gets NaN scores (tilde_omni_get_spans)
    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    auto inp_embd_ptr = std::make_unique<llama_model_tilde_omni_input_embd>(ubatch);
    auto * inp_embd   = inp_embd_ptr.get();
    res->add_input(std::move(inp_embd_ptr));

    inp_embd->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_embd->tokens);
    res->t_inp_tokens = inp_embd->tokens;
    ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embd, inp_embd->tokens);
    ggml_tensor * side = nullptr;
    if (inp_embd->n_e > 0) {
        inp_embd->embd_lm    = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, inp_embd->n_e);
        inp_embd->embd_slots = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, inp_embd->n_e);
        ggml_set_input(inp_embd->embd_lm);
        ggml_set_input(inp_embd->embd_slots);
        inpL = ggml_set_rows(ctx0, inpL, inp_embd->embd_lm, inp_embd->embd_slots);
    }
    cb(inpL, "model.input_embed", -1);
    res->t_inp_embd = inpL;

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
            cur = build_layer_gdn(cur, il);
        } else {
            cur = build_layer_attn(cur, inp_pos, il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // ffn_inp += ffn(norm(ffn_inp)), chunk by chunk
        cur = tilde_omni_residual_chunked(ctx0, gf, ffn_inp, cur, tilde_omni_chunk_tokens(), [&](ggml_tensor * x) {
            return build_ffn(x,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        });
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // the head is always evaluated, so that the graph has the same nodes for every batch
    ggml_tensor * fields = nullptr;
    ggml_tensor * hidden = cur;
    cur = build_head(hidden, inp_decision, &fields);
    cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));

    if (!inp_decision->spans.points.empty()) {
        // the side values of the embedding rows (the sub-patch keys of image tokens), zero elsewhere; a point field
        // always has an image, so the batch has embedding rows
        GGML_ASSERT(inp_embd->n_e > 0);
        inp_embd->embd_side = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, TILDE_OMNI_SIDE, inp_embd->n_e);
        ggml_set_input(inp_embd->embd_side);
        side = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, TILDE_OMNI_SIDE, n_tokens), 0.0f);
        side = ggml_set_rows(ctx0, side, inp_embd->embd_side, inp_embd->embd_slots);
        cur = ggml_concat(ctx0, cur, build_points(hidden, fields, side, inp_decision), 0);
    }

    // one flat output: the option logits, the point outputs, then zeros up to n_tokens * TILDE_OMNI_OUT (padded as 1D:
    // a [ne0, n_tokens] pad puts n_tokens in a CUDA grid dimension capped at 65535)
    const int64_t n_out = (int64_t) TILDE_OMNI_OUT * n_tokens;
    GGML_ASSERT(ggml_nelements(cur) <= n_out);
    cur = ggml_add(ctx0, cur, inp_decision->status);
    cur = ggml_pad(ctx0, cur, (int) (n_out - ggml_nelements(cur)), 0, 0, 0);
    cur = ggml_reshape_2d(ctx0, cur, TILDE_OMNI_OUT, n_tokens);
    cb(cur, "result_decision", -1);

    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_tilde_omni::graph::mm(ggml_tensor * w, ggml_tensor * x) {
    return build_lora_mm(w, x, model.row_scale(w));
}

// gated full attention, non-causal over the whole batch
ggml_tensor * llama_model_tilde_omni::graph::build_layer_attn(ggml_tensor * cur, ggml_tensor * inp_pos, int il) {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head = hparams.n_embd_head_v();

    // the Q projection outputs query + gate, per head
    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);

    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // no mask: one sequence, every token attends to every token. Without flash attention the scores would be
    // materialized as [n_tokens, n_tokens, n_head]: refuse long requests instead of trying to allocate them.
    if (!cparams.flash_attn && n_tokens > 4096) {
        throw std::runtime_error("tilde_omni: requests over 4096 tokens need flash attention (-fa on)");
    }
    cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr, n_tokens, kq_scale, il);
    cb(cur, "attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));

    cur = mm(layer.wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

// gated delta net over the whole batch: causal, the conv and recurrent states start from zero and are not kept
ggml_tensor * llama_model_tilde_omni::graph::build_layer_gdn(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];

    const int64_t d_inner     = hparams.ssm_d_inner;
    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = d_inner / num_v_heads;
    const int64_t n_seqs      = 1;

    ggml_tensor * qkv_mixed = mm(layer.wqkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_tokens, n_seqs);

    ggml_tensor * z = mm(layer.wqkv_gate, cur);

    ggml_tensor * beta = mm(layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);

    ggml_tensor * alpha = mm(layer.ssm_alpha, cur);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_tokens, n_seqs);

    ggml_tensor * gate = ggml_softplus(ctx0, ggml_add(ctx0, alpha, layer.ssm_dt));
    gate = ggml_mul(ctx0, gate, layer.ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_tokens, n_seqs);

    const int64_t conv_kernel_size = layer.ssm_conv1d->ne[0];
    const int64_t conv_channels    = d_inner + 2 * num_k_heads * head_k_dim;

    ggml_tensor * conv_states = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, conv_kernel_size - 1, conv_channels, n_seqs), 0.0f);
    ggml_tensor * conv_input  = ggml_concat(ctx0, conv_states, ggml_transpose(ctx0, qkv_mixed), 0);

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, layer.ssm_conv1d));
    cb(conv_out, "conv_output_silu", il);

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_tokens,
            0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_out));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_tokens,
            ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    q_conv = build_gdn_l2_norm(ctx0, q_conv, hparams.f_norm_rms_eps);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, hparams.f_norm_rms_eps);

    // the V heads are stored in tiled order (convert): an explicit repeat is needed only without the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
    }

    ggml_tensor * state = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_v_dim, head_v_dim, num_v_heads, n_seqs), 0.0f);

    ggml_tensor * output = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il).first;
    cb(output, "attn_output", il);

    // gated normalization
    ggml_tensor * z_4d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_tokens, n_seqs);
    output = build_norm(output, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    output = ggml_mul(ctx0, output, ggml_silu(ctx0, z_4d));

    output = ggml_reshape_3d(ctx0, output, head_v_dim * num_v_heads, n_tokens, n_seqs);

    cur = mm(layer.ssm_out, output);
    cb(cur, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
}

//
// head
//

// the head's products over all tokens (memory projection, cross-attention K and V) may run with F16 operands and F32
// accumulation (CUDA tensor cores). TILDE_HEAD_F16=0 turns it off.
static ggml_tensor * tilde_omni_head_mm_f16(ggml_tensor * mm) {
    static const bool on = [] {
        const char * e = getenv("TILDE_HEAD_F16");
        return e == nullptr || atoi(e) != 0;
    }();
    if (on) {
        ggml_prec_set_src(mm, GGML_PREC_F16, 0);
        ggml_prec_set_src(mm, GGML_PREC_F16, 1);
    }
    return mm;
}

ggml_tensor * llama_model_tilde_omni::graph::build_head_norm(ggml_tensor * cur, const norm & n) {
    return build_norm(cur, n.w, n.b, LLM_NORM, -1);
}

// nn.MultiheadAttention without a mask; q: [n_embd_h, n_q], kv: [n_embd_h, n_kv]
ggml_tensor * llama_model_tilde_omni::graph::build_head_attn(ggml_tensor * q, ggml_tensor * kv, const attn & a) {
    const int64_t n_embd_h = q->ne[0];
    const int64_t n_head_h = model.n_head_decision;
    const int64_t d_head   = n_embd_h / n_head_h;

    ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wq, q),  a.bq);
    ggml_tensor * Kcur = ggml_add(ctx0, tilde_omni_head_mm_f16(ggml_mul_mat(ctx0, a.wk, kv)), a.bk);
    ggml_tensor * Vcur = ggml_add(ctx0, tilde_omni_head_mm_f16(ggml_mul_mat(ctx0, a.wv, kv)), a.bv);

    Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head_h, q->ne[1]);
    Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head_h, kv->ne[1]);
    Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head_h, kv->ne[1]);

    // flash attention when enabled: the memory can be the whole request (64k tokens)
    // (the head lives with the output layer; the last layer stands for its device in the fused-op checks)
    ggml_tensor * cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr, kv->ne[1],
            1.0f / sqrtf(float(d_head)), n_layer - 1);

    return ggml_add(ctx0, ggml_mul_mat(ctx0, a.wo, cur), a.bo);
}

ggml_tensor * llama_model_tilde_omni::graph::build_head_ffn(ggml_tensor * cur, const head_layer & layer) {
    cur = build_head_norm(cur, layer.ffn_norm);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_up, cur), layer.ffn_up_b);
    cur = ggml_gelu_erf(ctx0, cur);
    return ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_down, cur), layer.ffn_down_b);
}

// mean of each span of rows of `table` (gathered by `ids`, I32 [n_gathered], span by span); returns [n_embd, n_spans]
ggml_tensor * llama_model_tilde_omni::graph::build_span_means(ggml_tensor * table, ggml_tensor * ids, ggml_tensor * const * pools, size_t n_pools) {
    ggml_tensor * res = nullptr;
    int64_t tok0 = 0;
    for (size_t c = 0; c < n_pools; c++) {
        ggml_tensor * pool = pools[c];
        const int64_t n_tok = pool->ne[0];
        ggml_tensor * idx  = ggml_view_1d(ctx0, ids, n_tok, tok0 * ggml_element_size(ids));
        ggml_tensor * part = ggml_get_rows(ctx0, table, idx);                          // [n_embd, n_tok]
        part = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, part)), pool); // [n_embd, chunk spans]
        res = res ? ggml_concat(ctx0, res, part, 1) : part;
        tok0 += n_tok;
    }
    GGML_ASSERT(tok0 == ids->ne[0]);
    return res;
}

// ref: tilda/qwen35/model/head.py (PackedJointSchemaHead.forward_packed)
ggml_tensor * llama_model_tilde_omni::graph::build_head(ggml_tensor * hidden, input_decision * inp, ggml_tensor ** fields_out) {
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

    ggml_tensor * memory = tilde_omni_head_mm_f16(ggml_mul_mat(ctx0, model.proj_memory, hidden)); // [n_embd_h, n_tokens]
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
    *fields_out = fields;

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

// ref: tilda/vision/point.py (PointHead.field and point_decode), unified mode: the query is the point field's final
// field vector. Per point field: the logits of [not on screen, image tokens...], then at the most likely image token
// the 4 sub-patch logits, the 4 x 9 location logits and the 2 log2 sizes.
ggml_tensor * llama_model_tilde_omni::graph::build_points(ggml_tensor * hidden, ggml_tensor * fields, ggml_tensor * side, input_decision * inp) {
    const auto & ph = model.point;
    const int64_t n_embd_h = fields->ne[0];
    const int64_t n_key    = ph.key->ne[1];
    const int64_t n_sub    = ph.sub_query->ne[1];

    auto ln = [&](ggml_tensor * x, const norm & n) {
        return build_norm(x, n.w, n.b, LLM_NORM, -1);
    };
    auto mlp = [&](ggml_tensor * x, ggml_tensor * up, ggml_tensor * up_b, ggml_tensor * down, ggml_tensor * down_b) {
        x = ggml_gelu_erf(ctx0, ggml_add(ctx0, ggml_mul_mat(ctx0, up, x), up_b));
        return ggml_add(ctx0, ggml_mul_mat(ctx0, down, x), down_b);
    };

    ggml_tensor * out = nullptr;
    for (size_t i = 0; i < inp->spans.points.size(); i++) {
        const auto & pt = inp->spans.points[i];
        const int64_t n_img = inp->spans.image_tokens(pt);

        inp->point_rows.push_back(ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1 + n_img));
        ggml_tensor * rows_idx = inp->point_rows.back();
        ggml_set_input(rows_idx);

        ggml_tensor * q_in = ggml_view_2d(ctx0, fields, n_embd_h, 1, fields->nb[1], pt.question * fields->nb[1]);

        // token: q . k over [none, image tokens]
        ggml_tensor * q    = ggml_mul_mat(ctx0, ph.query, ln(q_in, ph.query_norm));             // [n_key, 1]
        ggml_tensor * rows = ggml_get_rows(ctx0, hidden, rows_idx);                               // [n_embd, 1 + n_img]
        ggml_tensor * keys = ggml_mul_mat(ctx0, ph.key, ln(rows, ph.key_norm));                  // [n_key, 1 + n_img]
        ggml_tensor * logits = ggml_scale(ctx0, ggml_mul_mat(ctx0, keys, q), 1.0f / sqrtf((float) n_key)); // [1 + n_img, 1]

        // the most likely image token
        ggml_tensor * tok_logits = ggml_view_2d(ctx0, logits, n_img, 1, logits->nb[1], ggml_element_size(logits));
        ggml_tensor * best = ggml_argmax(ctx0, ggml_cont(ctx0, tok_logits));                     // I32 [1]
        ggml_tensor * img_rows = ggml_view_2d(ctx0, rows, n_embd, n_img, rows->nb[1], rows->nb[1]);
        ggml_tensor * h_best = ggml_get_rows(ctx0, ggml_cont(ctx0, img_rows), best);             // [n_embd, 1]

        // sub-patch and location at that token, from its sub-patch keys
        ggml_tensor * img_idx = ggml_view_1d(ctx0, rows_idx, n_img, ggml_element_size(rows_idx));
        ggml_tensor * sub_k   = ggml_get_rows(ctx0, ggml_get_rows(ctx0, side, img_idx), best);   // [4 * n_sub, 1]
        sub_k = ggml_reshape_2d(ctx0, sub_k, n_sub, 4);

        ggml_tensor * pair  = ln(ggml_concat(ctx0, q_in, h_best, 0), ph.pair_norm);              // [n_embd_h + n_embd, 1]
        ggml_tensor * sub_q = ggml_mul_mat(ctx0, ph.sub_query, pair);                            // [n_sub, 1]
        ggml_tensor * size  = mlp(pair, ph.size_up, ph.size_up_b, ph.size_down, ph.size_down_b); // [2, 1]

        ggml_tensor * sub_logits = ggml_scale(ctx0, ggml_mul_mat(ctx0, sub_k, sub_q), 1.0f / sqrtf((float) n_sub)); // [4, 1]
        ggml_tensor * loc_in = ggml_concat(ctx0, ggml_repeat_4d(ctx0, sub_q, n_sub, 4, 1, 1), sub_k, 0);           // [2 n_sub, 4]
        ggml_tensor * loc    = mlp(loc_in, ph.loc_up, ph.loc_up_b, ph.loc_down, ph.loc_down_b);                    // [9, 4]

        ggml_tensor * cur = ggml_reshape_1d(ctx0, logits, 1 + n_img);
        cur = ggml_concat(ctx0, cur, ggml_reshape_1d(ctx0, sub_logits, 4), 0);
        cur = ggml_concat(ctx0, cur, ggml_reshape_1d(ctx0, loc, 36), 0);
        cur = ggml_concat(ctx0, cur, ggml_reshape_1d(ctx0, size, 2), 0);
        out = out ? ggml_concat(ctx0, out, cur, 0) : cur;
    }
    cb(out, "decision_points", -1);
    return out;
}
