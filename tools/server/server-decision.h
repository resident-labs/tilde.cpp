#pragma once

#include "server-common.h"
#include "server-task.h"
#include "tilde-omni-image.h"
#include "tilde-omni-point.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// typed decision models (TypeSafe /v1/systemone API)
// the model answers each question in one forward pass, no token is generated

enum server_decision_question_type {
    SERVER_DECISION_QUESTION_CHOICE,
    SERVER_DECISION_QUESTION_SCORE,
    SERVER_DECISION_QUESTION_NOUL,
    SERVER_DECISION_QUESTION_POINT, // tilde omni: a location on an image (or not on screen)
};

struct server_decision_option {
    std::string key;
    json description; // null if not provided
    std::string text; // tilde: the option text given to the model
};

struct server_decision_question {
    std::string id;
    server_decision_question_type type;
    json instructions;
    std::vector<server_decision_option> options; // in the order of the model outputs

    // tilde
    std::string              text;             // the instruction text given to the model
    std::vector<std::string> legend;           // score: the description of each level
    float                    noul_temperature = 0.0f; // noul: extra temperature for this answer, 0 if none
    int                      image = -1;              // point: the image index
};

// tilde omni: the images of a request, as stored for the model (tilda.vision.budget) and their token layouts
struct server_decision_media {
    struct image {
        std::string          label = "screenshot";
        bool                 has_time = false;
        double               time = 0.0;
        int                  width = 0;  // original pixels
        int                  height = 0;
        std::vector<uint8_t> rgb;        // original pixels, then released once stored
        tilde_omni::plan     plan;
        std::vector<uint8_t> stored;     // plan.width x plan.height x 3
        tilde_omni::layout   layout;
    };
    std::vector<image> images;
    // audio clips, after the images (tilda.data.parse_row); whole clips, "Audio k: <label>"
    struct audio {
        std::string label = "audio";
        raw_buffer  data;   // the file (wav / mp3 / flac)
    };
    std::vector<audio> audios;
};

struct server_decision_context {
    common_decision_type type = COMMON_DECISION_TYPE_NONE;

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
    void init(const llama_model * model);

    // true if the questions of a request start with the same tokens, and the model can continue from them
    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_LFM2_D1:
                return true;
            default:
                return false;
        }
    }

    // true if all the questions of a request go in one prompt, see fill_task_joint()
    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF || type == COMMON_DECISION_TYPE_TILDE;
    }

    // tilde: request limits, read from the model metadata (rejected, never truncated)
    size_t max_questions       = 0; // 0: no limit
    size_t max_request_tokens  = 0; // state + all the questions
    size_t max_question_tokens = 0; // state + one question

    // tilde omni: images (and the point question type) in the request
    bool tilde_omni = false;

    // tilde omni: the request's images, decoded, budgeted and stored (throws std::invalid_argument on bad input)
    server_decision_media parse_media_tilde(const json & body, mtmd_context * mctx) const;

    // tilde: the joint prompt with the request's media; the result is a flat array: every option's logit in order,
    // then the point head's outputs of every point question in order
    void fill_task_tilde(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_media & media,
            mtmd_context * mctx,
            server_task & task) const;

    // tilde: answers of all the questions from the flat result (and "images" for a request with images)
    json format_answers_tilde(
            const std::vector<server_decision_question> & questions,
            const std::vector<float> & scores,
            const server_decision_media & media) const;

    // true if the prompt of the model has a place for images
    bool can_use_images() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_CLEF:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_LFM2_D1:
                return true;
            default:
                return false;
        }
    }

    // throw std::invalid_argument on bad input
    std::vector<server_decision_question> parse_questions(const json & body) const;

    // returns the state without its images, they are appended to files in order
    // images come from "images" and from the image_url parts of a state made of chat messages
    json parse_state(const json & body, std::vector<raw_buffer> & files) const;

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const server_decision_question & question) const;

    // set the prompt of one variant of this question, and where to read its result
    // mctx is only used if there are files
    void fill_task(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    // set the prompt of all the questions, the result has the scores of all their options, in order
    // mctx is only used if there are files
    void fill_task_joint(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    // scores: the raw model outputs of each variant
    json format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]
    bool   choice_sorted   = false; // choice options are in the order of their keys

    // OPENJEV, LEV, NIMBLE, PPLX_DECIDER
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    // TILDE
    llama_token              tilde_bos = LLAMA_TOKEN_NULL;
    std::vector<llama_token> tilde_markers; // start, state, state end, question, options, decision
    std::vector<llama_token> tilde_media_ids; // omni: image start, image end, image token, point query, point none, audio start, audio end
    bool                     tilde_noul_normalize = false; // noul options are normalized (true first, default texts)

    std::vector<server_decision_question> parse_questions_tilde(const json & body) const;
    json format_answer_tilde(const server_decision_question & question, const std::vector<float> & scores, json & answer) const;

    std::string render(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            size_t n_images) const;
    json render_options(const server_decision_question & question, size_t variant) const;
    size_t n_outputs(const server_decision_question & question) const;
    // LFM2_D1: label text and tokens of each option
    void d1_labels(const server_decision_question & question, std::vector<std::string> & texts, std::vector<llama_tokens> & groups) const;
    void fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task) const;

    float get_temperature(const server_decision_question & question) const;
};

// group the tasks so that the common prefix of their prompts is evaluated only once
// each group is one parent and its children, it takes at most n_slots slots
// note: the order of the tasks is preserved
std::vector<server_task> server_decision_group_tasks(std::vector<server_task> && tasks, size_t n_slots);
