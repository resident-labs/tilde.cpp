#include "server-decision.h"

#include "../../src/llama-ext.h" // staging API: llama_decision_order

#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>
#include <stdexcept>

static const char * decision_question_type_name(server_decision_question_type type) {
    switch (type) {
        case SERVER_DECISION_QUESTION_CHOICE: return "choice";
        case SERVER_DECISION_QUESTION_SCORE:  return "score";
        case SERVER_DECISION_QUESTION_NOUL:   return "noul";
    }
    return "";
}

// lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes
static const size_t DECISION_LEV_N_RATINGS = 9;

static std::string decision_meta_str(const llama_model * model, const std::string & key) {
    char buf[256];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? "" : std::string(buf);
}

//
// model-specific setup
//

void server_decision_context::init(const llama_model * model) {
    *this = server_decision_context(); // the model can be reloaded

    const common_decision_type model_type = common_get_decision_type(model);
    if (model_type == COMMON_DECISION_TYPE_NONE) {
        return;
    }

    const std::string prefix    = decision_meta_str(model, "general.architecture") + ".decision.";
    const std::string type_name = decision_meta_str(model, prefix + "type");

    vocab = llama_model_get_vocab(model);

    const char * tmpl_src = llama_model_chat_template(model, "systemone");
    if (tmpl_src != nullptr) {
        tmpl = std::make_shared<const common_chat_template>(tmpl_src, "", "");
    } else if (model_type != COMMON_DECISION_TYPE_TILDE) { // tilde builds its prompt in fill_task_tilde()
        throw std::runtime_error("decision model has no \"systemone\" template");
    }

    const std::string prefix_temp = prefix + "temperature.";
    for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
        char key[256];
        char val[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_temp)) {
            continue;
        }
        if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
            continue;
        }
        const float temp = std::strtof(val, nullptr);
        if (temp <= 0.0f) {
            throw std::runtime_error(string_format("invalid decision temperature: %s = %s", key, val));
        }
        temperatures[key + prefix_temp.size()] = temp;
    }

    if (model_type == COMMON_DECISION_TYPE_OPENJEV) {
        // one letter per option, each must be a single token
        const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        for (const char c : letters) {
            const auto toks = common_tokenize(vocab, std::string(1, c), false, false);
            if (toks.size() != 1) {
                throw std::runtime_error(string_format("decision label '%c' is not a single token", c));
            }
            labels.push_back(toks[0]);
        }
        n_options_max   = labels.size();
        noul_true_first = true;
    } else if (model_type == COMMON_DECISION_TYPE_LEV || model_type == COMMON_DECISION_TYPE_NIMBLE || model_type == COMMON_DECISION_TYPE_PPLX_DECIDER) {
        // label codes are A..Z then AA..ZZ, only the ones that are a single token are used
        std::vector<std::string> codes;
        for (char a = 'A'; a <= 'Z'; a++) {
            codes.push_back(std::string(1, a));
        }
        for (char a = 'A'; a <= 'Z'; a++) {
            for (char b = 'A'; b <= 'Z'; b++) {
                codes.push_back(std::string{a, b});
            }
        }
        for (const auto & code : codes) {
            const auto toks = common_tokenize(vocab, code, false, false);
            if (toks.size() == 1 && labels.size() < 255) {
                labels.push_back(toks[0]);
                label_texts.push_back(code);
            }
        }
        n_options_max = labels.size();
    } else if (model_type == COMMON_DECISION_TYPE_KEV) {
        // the hidden state of an option is read at the token that ends it
        const auto toks = common_tokenize(vocab, "<|box_end|>", false, true);
        if (toks.size() != 1) {
            throw std::runtime_error("decision model has no <|box_end|> token");
        }
        token_marker  = toks[0];
        n_options_max = 255;
    } else if (model_type == COMMON_DECISION_TYPE_LAYA) {
        token_marker = llama_vocab_mask(vocab);
        token_sep    = llama_vocab_sep(vocab);
        if (token_marker == LLAMA_TOKEN_NULL || token_sep == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("decision model has no mask or sep token");
        }
        text_marker = common_token_to_piece(vocab, token_marker, true);

        const std::string val = decision_meta_str(model, prefix + "max_head_tokens");
        max_head_tokens = std::strtoul(val.c_str(), nullptr, 10);
        if (max_head_tokens == 0) {
            throw std::runtime_error("decision model has no valid max_head_tokens");
        }
        n_options_max = 255;
    } else if (model_type == COMMON_DECISION_TYPE_CLEF) {
        n_options_max   = 255;
        noul_true_first = true;
        choice_sorted   = true;
    } else if (model_type == COMMON_DECISION_TYPE_LFM2_D1) {
        n_options_max   = 255;
        noul_true_first = true;
    } else if (model_type == COMMON_DECISION_TYPE_TILDE) {
        // structure markers follow the vocabulary of the base model, see tilde/convert_tilde_to_gguf.py
        static const char * names[] = {
            "<|tilda_start|>", "<|tilda_state|>", "<|tilda_state_end|>", "<|tilda_question|>", "<|tilda_options|>", "<|tilda_decision|>",
        };
        const long marker_start = std::strtol(decision_meta_str(model, "tilde.marker_start").c_str(), nullptr, 10);
        for (size_t i = 0; i < 6; i++) {
            const llama_token id = (llama_token) (marker_start + i);
            if (marker_start <= 0 || id >= llama_vocab_n_tokens(vocab) || std::string(llama_vocab_get_text(vocab, id)) != names[i]) {
                throw std::runtime_error(string_format("tilde: marker %s is not at token %d", names[i], id));
            }
            tilde_markers.push_back(id);
        }
        const long bos = std::strtol(decision_meta_str(model, "tilde.bos_token_id").c_str(), nullptr, 10);
        tilde_bos = bos >= 0 ? (llama_token) bos : LLAMA_TOKEN_NULL;

        const std::string noul_format = decision_meta_str(model, "tilde.noul_format");
        if (noul_format == "clef_true_first_v1") {
            tilde_noul_normalize = true;
        } else if (noul_format != "legacy") {
            throw std::runtime_error("tilde: unsupported noul format: " + noul_format);
        }
        if (decision_meta_str(model, "tilde.input_layout") != "state_first") {
            throw std::runtime_error("tilde: only the state_first input layout is supported");
        }

        auto meta_size = [&](const char * key, size_t def) {
            const std::string val = decision_meta_str(model, key);
            return val.empty() ? def : (size_t) std::strtoull(val.c_str(), nullptr, 10);
        };
        n_options_max       = meta_size("tilde.max_options",         255);
        max_questions       = meta_size("tilde.max_questions",       32);
        max_request_tokens  = meta_size("tilde.max_request_tokens",  65536);
        max_question_tokens = meta_size("tilde.max_question_tokens", 32768);
        noul_true_first     = tilde_noul_normalize;
    } else {
        throw std::runtime_error("unsupported decision model type: " + type_name);
    }
    type = model_type;

    SRV_INF("decision model type: %s\n", type_name.c_str());
}

//
// request parsing
//

std::vector<server_decision_question> server_decision_context::parse_questions(const json & body) const {
    if (type == COMMON_DECISION_TYPE_TILDE) {
        return parse_questions_tilde(body);
    }
    // d1 accepts a null state (images only)
    if (!body.contains("state") || (body.at("state").is_null() && type != COMMON_DECISION_TYPE_LFM2_D1)) {
        throw std::invalid_argument("\"state\" must be provided");
    }
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        throw std::invalid_argument("\"questions\" must be a non-empty object");
    }

    std::vector<server_decision_question> questions;
    for (const auto & [id, q] : body.at("questions").items()) {
        auto err = [&id = id](const std::string & msg) {
            return std::invalid_argument("questions." + id + ": " + msg);
        };
        if (!q.is_object()) {
            throw err("must be an object");
        }
        if (!q.contains("instructions") || q.at("instructions").is_null()) {
            throw err("\"instructions\" must be provided");
        }

        server_decision_question question;
        question.id           = id;
        question.instructions = q.at("instructions");

        const std::string type_name = json_value(q, "type", std::string());
        const json        criteria  = q.contains("criteria") ? q.at("criteria") : json();

        if (type_name == "choice") {
            question.type = SERVER_DECISION_QUESTION_CHOICE;
            if (!criteria.is_object() || criteria.empty()) {
                throw err("\"criteria\" must be a non-empty object");
            }
            for (const auto & [key, description] : criteria.items()) {
                question.options.push_back({key, description});
            }
            if (choice_sorted) {
                std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                    return a.key < b.key;
                });
            }
        } else if (type_name == "score") {
            question.type = SERVER_DECISION_QUESTION_SCORE;
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10) {
                throw err("\"criteria\" must be an array of 2 to 10 levels");
            }
            for (size_t i = 0; i < criteria.size(); i++) {
                question.options.push_back({std::to_string(i), criteria.at(i)});
            }
        } else if (type_name == "noul") {
            question.type = SERVER_DECISION_QUESTION_NOUL;
            if (!criteria.is_null() && !criteria.is_object()) {
                throw err("\"criteria\" must be an object");
            }
            for (const char * key : {"false", "true"}) {
                question.options.push_back({key, criteria.is_object() && criteria.contains(key) ? criteria.at(key) : json()});
            }
            if (noul_true_first) {
                std::swap(question.options[0], question.options[1]);
            }
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
        }

        if (question.options.size() > n_options_max) {
            throw err(string_format("too many options (%zu), this model supports at most %zu", question.options.size(), n_options_max));
        }

        questions.push_back(std::move(question));
    }
    return questions;
}

//
// images
//

static const size_t DECISION_MAX_IMAGES = 8;

static void decision_load_image(const json & url, std::vector<raw_buffer> & files) {
    if (!url.is_string() || !string_starts_with(url.get<std::string>(), "data:image/")) {
        throw std::invalid_argument("images must be data URLs (data:image/...;base64,...)");
    }
    if (files.size() >= DECISION_MAX_IMAGES) {
        throw std::invalid_argument(string_format("too many images, the maximum is %zu", DECISION_MAX_IMAGES));
    }
    handle_media(files, url.get<std::string>(), "");
}

json server_decision_context::parse_state(const json & body, std::vector<raw_buffer> & files) const {
    if (type == COMMON_DECISION_TYPE_TILDE) {
        // tilde reads the state as given (text or JSON), like tilda/serve.py; media are refused rather than answered
        // from the text around them
        static const char * TEXT_ONLY = "this Tilde model is text-only";
        for (const char * key : { "images", "videos", "audio" }) {
            if (body.contains(key) && !body.at(key).is_null() && !(body.at(key).is_array() && body.at(key).empty())) {
                throw std::invalid_argument(string_format("%s: \"%s\" is not supported", TEXT_ONLY, key));
            }
        }
        const json state = body.contains("state") ? body.at("state") : json();
        const json & messages = state.is_object() && state.contains("messages") ? state.at("messages") : state;
        if (messages.is_array()) {
            for (const auto & msg : messages) {
                if (!msg.is_object() || !msg.contains("content") || !msg.at("content").is_array()) {
                    continue;
                }
                for (const auto & part : msg.at("content")) {
                    const std::string t = part.is_object() ? json_value(part, "type", std::string()) : std::string();
                    if (t == "image_url" || t == "image" || t == "input_image" || t == "input_audio" || t == "video_url") {
                        throw std::invalid_argument(string_format("%s: \"%s\" content is not supported", TEXT_ONLY, t.c_str()));
                    }
                }
            }
        }
        return state;
    }
    if (body.contains("videos") && !body.at("videos").is_null() && !body.at("videos").empty()) {
        throw std::invalid_argument("\"videos\" is not supported");
    }
    if (body.contains("images") && !body.at("images").is_null()) {
        if (!body.at("images").is_array()) {
            throw std::invalid_argument("\"images\" must be an array");
        }
        for (const auto & url : body.at("images")) {
            decision_load_image(url, files);
        }
    }

    const json & state = body.at("state");
    const bool is_wrapped = state.is_object() && state.contains("messages");
    const json & messages = is_wrapped ? state.at("messages") : state;
    if (!messages.is_array()) {
        return state;
    }

    // chat messages: take the image parts out of the content
    json messages_out = json::array();
    for (const auto & msg : messages) {
        if (!msg.is_object() || !msg.contains("content") || !msg.at("content").is_array()) {
            messages_out.push_back(msg);
            continue;
        }
        json content = json::array();
        for (const auto & part : msg.at("content")) {
            if (part.is_object() && json_value(part, "type", std::string()) == "image_url" && part.contains("image_url")) {
                const json & image_url = part.at("image_url");
                decision_load_image(image_url.is_object() && image_url.contains("url") ? image_url.at("url") : image_url, files);
            } else {
                content.push_back(part);
            }
        }
        json msg_out = msg;
        msg_out["content"] = content;
        messages_out.push_back(msg_out);
    }

    if (!is_wrapped) {
        return messages_out;
    }
    json state_out = state;
    state_out["messages"] = messages_out;
    return state_out;
}

//
// prompt
//

// replace text in all strings of a JSON value
static json decision_replace_text(const json & val, const std::string & search, const std::string & replace) {
    if (val.is_string()) {
        std::string str = val.get<std::string>();
        string_replace_all(str, search, replace);
        return str;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_replace_text(item, search, replace));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = decision_replace_text(item, search, replace);
        }
        return out;
    }
    return val;
}

// sort the keys of all objects of a JSON value
static json decision_sort_keys(const json & val) {
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_sort_keys(item));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = decision_sort_keys(item);
        }
        json out = json::object();
        for (const auto & [key, item] : sorted) {
            out[key] = item;
        }
        return out;
    }
    return val;
}

// kev flattens a JSON value into text, the keys of an object are kept as labels (kev/api.py: render)
static std::string decision_kev_render(const json & val, int indent = 0) {
    const std::string pad(2 * indent, ' ');
    if (val.is_null()) {
        return "";
    }
    if (val.is_string()) {
        return val.get<std::string>();
    }
    if (val.is_boolean()) {
        return val.get<bool>() ? "True" : "False";
    }
    if (val.is_array()) {
        std::string out;
        for (const auto & item : val) {
            const std::string text = decision_kev_render(item, indent + 1);
            out += (out.empty() ? "" : "\n") + pad + "- " + text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
        }
        return out;
    }
    if (val.is_object()) {
        std::string out;
        for (const auto & [key, item] : val.items()) {
            const bool is_nested = item.is_object() || item.is_array();
            out += (out.empty() ? "" : "\n") + pad + key + (is_nested ? ":\n" : ": ") + decision_kev_render(item, is_nested ? indent + 1 : 0);
        }
        return out;
    }
    return val.dump();
}

// kev text input: special tokens written in the text must not be parsed as such
static std::string decision_kev_text(const json & val) {
    static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
    return std::regex_replace(decision_kev_render(val), re_special, "<\xC2\xA6$1\xC2\xA6>");
}

size_t server_decision_context::n_variants(const server_decision_question & question) const {
    // lev shows the options of a choice in 2 orders, to cancel the preference for the first label
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_CHOICE && question.options.size() > 1) {
        return 2;
    }
    return 1;
}

size_t server_decision_context::n_outputs(const server_decision_question & question) const {
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_NOUL) {
        return DECISION_LEV_N_RATINGS;
    }
    return question.options.size();
}

// label codes follow prompt.py of the model repo
void server_decision_context::d1_labels(const server_decision_question & question, std::vector<std::string> & texts, std::vector<llama_tokens> & groups) const {
    const size_t n_options = question.options.size();

    auto get_single_tokens = [&](const std::vector<std::string> & forms) {
        llama_tokens out;
        for (const auto & form : forms) {
            const auto toks = common_tokenize(vocab, form, false, false);
            if (toks.size() == 1 && std::find(out.begin(), out.end(), toks[0]) == out.end()) {
                out.push_back(toks[0]);
            }
        }
        return out;
    };

    if (question.type != SERVER_DECISION_QUESTION_CHOICE) {
        for (const auto & opt : question.options) {
            llama_tokens group;
            if (question.type == SERVER_DECISION_QUESTION_SCORE) {
                group = get_single_tokens({opt.key});
            } else if (opt.key == "true") {
                group = get_single_tokens({"yes", "Yes", "YES"});
            } else {
                group = get_single_tokens({"no", "No", "NO"});
            }
            if (group.empty()) {
                throw std::runtime_error("decision label is not a single token: " + opt.key);
            }
            texts.push_back(opt.key);
            groups.push_back(group);
        }
        return;
    }

    bool is_letters = true;
    for (const auto & opt : question.options) {
        is_letters = is_letters && opt.key.size() == 1 && std::isalpha((unsigned char) opt.key[0]);
    }

    std::vector<std::string> codes;
    for (size_t i = 0; i < n_options; i++) {
        if (is_letters) {
            codes.push_back(question.options[i].key);
        } else if (n_options <= 26) {
            codes.push_back(std::string(1, 'A' + i));
        } else {
            codes.push_back(string_format("%02zu", i));
        }
    }

    std::vector<std::string> pool;
    for (char c = 'A'; c <= 'Z'; c++) {
        pool.push_back(std::string(1, c));
    }
    for (int i = 0; i < 100; i++) {
        pool.push_back(string_format("%02d", i));
    }
    for (char c = 'a'; c <= 'z'; c++) {
        pool.push_back(std::string(1, c));
    }
    for (int i = 0; i < 200; i++) {
        pool.push_back(string_format("#%d", i));
    }
    for (char a = 'A'; a <= 'Z'; a++) {
        for (char b = 'A'; b <= 'Z'; b++) {
            pool.push_back(std::string{a, b});
        }
    }

    llama_tokens used;
    auto take = [&](const std::string & code) {
        const auto toks = common_tokenize(vocab, code, false, false);
        if (toks.size() != 1 || std::find(used.begin(), used.end(), toks[0]) != used.end()) {
            return false;
        }
        used.push_back(toks[0]);
        llama_tokens group = {toks[0]};
        for (const llama_token tok : get_single_tokens({" " + code})) {
            if (tok != toks[0]) {
                group.push_back(tok);
            }
        }
        texts.push_back(code);
        groups.push_back(group);
        return true;
    };
    for (const auto & code : codes) {
        bool is_taken = take(code);
        for (size_t i = 0; !is_taken && i < pool.size(); i++) {
            is_taken = take(pool[i]);
        }
        if (!is_taken) {
            throw std::invalid_argument(string_format("no single-token label left for %zu options", n_options));
        }
    }
}

json server_decision_context::render_options(const server_decision_question & question, size_t variant) const {
    const size_t n_options = question.options.size();

    std::vector<std::string>  d1_texts;
    std::vector<llama_tokens> d1_groups;
    if (type == COMMON_DECISION_TYPE_LFM2_D1) {
        d1_labels(question, d1_texts, d1_groups);
    }

    // the second variant shows the options in the reverse order
    json options = json::array();
    for (size_t i = 0; i < n_options; i++) {
        const auto & opt = question.options[variant == 0 ? i : n_options - 1 - i];
        json option = json{
            {"key",         opt.key},
            {"description", opt.description},
        };
        if (type == COMMON_DECISION_TYPE_KEV) {
            option["key"] = decision_kev_text(opt.key);
            if (!opt.description.is_null()) {
                option["description"] = decision_kev_text(opt.description);
            }
        }
        if (!label_texts.empty()) {
            option["label"] = label_texts[i];
        }
        if (!d1_texts.empty()) {
            option["label"] = d1_texts[i];
        }
        options.push_back(option);
    }
    return options;
}

std::string server_decision_context::render(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        size_t n_images) const {
    // the template is given raw JSON values, it serializes the ones that are not strings
    json inp = json{
        {"id",           question.id},
        {"type",         decision_question_type_name(question.type)},
        {"instructions", question.instructions},
        {"state",        state},
        {"options",      render_options(question, variant)},
    };

    // the nimble prompt lists all the questions of the request
    if (type == COMMON_DECISION_TYPE_NIMBLE) {
        inp["questions"] = json::array();
        for (const auto & q : questions) {
            inp["questions"].push_back(json{
                {"id",           q.id},
                {"type",         decision_question_type_name(q.type)},
                {"instructions", q.instructions},
                {"options",      render_options(q, 0)},
            });
        }
    }

    // lev was trained with sorted keys
    if (type == COMMON_DECISION_TYPE_LEV) {
        inp = decision_sort_keys(inp);
    }

    // the kev template only takes text
    if (type == COMMON_DECISION_TYPE_KEV) {
        inp["state"]        = decision_kev_text(state);
        inp["instructions"] = decision_kev_text(question.instructions);
    }

    // the input must not contain the marker of the options
    if (!text_marker.empty()) {
        inp = decision_replace_text(inp, text_marker, " ");
    }

    // the template puts one media marker per image
    json images = json::array();
    if (n_images > 0) {
        inp = decision_replace_text(inp, get_media_marker(), " ");
        for (size_t i = 0; i < n_images; i++) {
            images.push_back(get_media_marker());
        }
    }
    inp["images"] = images;

    jinja::context ctx(tmpl->source());
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl->prog);
    return jinja::runtime::gather_string_parts(results)->as_string().str();
}

void server_decision_context::fill_task(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        const std::vector<raw_buffer> & files,
        mtmd_context * mctx,
        const mtmd_helper_init_opt & init_opt,
        server_task & task) const {
    const std::string prompt = render(state, questions, question, variant, files.size());

    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE || type == COMMON_DECISION_TYPE_PPLX_DECIDER) {
        // lev reads the ratings of a noul question at its first labels, not at the digits
        task.decision.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
    }
    if (type == COMMON_DECISION_TYPE_LFM2_D1) {
        std::vector<std::string>  texts;
        std::vector<llama_tokens> groups;
        d1_labels(question, texts, groups);
        for (const auto & group : groups) {
            task.decision.labels.insert(task.decision.labels.end(), group.begin(), group.end());
            task.decision.label_groups.push_back(group.size());
        }
    }
    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE || type == COMMON_DECISION_TYPE_PPLX_DECIDER || type == COMMON_DECISION_TYPE_LFM2_D1) {
        if (!files.empty()) {
            task.tokens = process_mtmd_prompt(mctx, prompt, files, init_opt);
            return;
        }
    }

    llama_tokens tokens = common_tokenize(vocab, prompt, false, true);
    if (type == COMMON_DECISION_TYPE_LAYA) {
        fill_task_laya(tokens, question, task);
    }
    if (type == COMMON_DECISION_TYPE_KEV) {
        // an option is read at its end token, the question at the last token
        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i] == token_marker) {
                task.decision.markers.push_back(i);
            }
        }
        if (task.decision.markers.size() != question.options.size()) {
            throw std::runtime_error("unexpected layout of the decision prompt");
        }
        task.decision.pointer = tokens.size() - 1;
    }
    task.tokens = server_tokens(tokens, false);
}

// the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
// options and question are cut to fit max_head_tokens, the same way the model was trained
void server_decision_context::fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task) const {
    const size_t n_options = question.options.size();

    std::vector<size_t> markers;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == token_marker) {
            markers.push_back(i);
        }
    }
    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");
    if (markers.size() != n_options || markers[0] < 2 || tokens[markers[0] - 1] != token_sep || tokens.back() != token_sep) {
        throw invalid;
    }
    const size_t head_end = markers[0] - 1;
    const size_t opts_end = std::find(tokens.begin() + markers.back(), tokens.end(), token_sep) - tokens.begin();
    if (opts_end + 1 >= tokens.size()) {
        throw invalid;
    }

    // marker + text of each option
    std::vector<llama_tokens> options;
    size_t n_options_tokens = 0;
    auto set_max = [&](size_t n_max) {
        n_options_tokens = 0;
        for (auto & opt : options) {
            opt.resize(std::min(opt.size(), n_max));
            n_options_tokens += opt.size();
        }
    };
    for (size_t i = 0; i < n_options; i++) {
        const size_t end = i + 1 < n_options ? markers[i + 1] : opts_end;
        options.emplace_back(tokens.begin() + markers[i], tokens.begin() + end);
    }
    set_max(max_option_tokens + 1);
    if (n_options_tokens + 16 > max_head_tokens) {
        // too many or too long options, shrink them evenly
        set_max(std::max((size_t) 4, (max_head_tokens - std::min(max_head_tokens, (size_t) 16)) / n_options));
    }
    const size_t n_question_max = std::max((size_t) 8, max_head_tokens - std::min(max_head_tokens, n_options_tokens));

    llama_tokens out;
    out.push_back(tokens[0]);
    out.insert(out.end(), tokens.begin() + 1, tokens.begin() + std::min(head_end, 1 + n_question_max));
    out.push_back(token_sep);
    for (const auto & opt : options) {
        task.decision.markers.push_back(out.size());
        out.insert(out.end(), opt.begin(), opt.end());
    }
    out.insert(out.end(), tokens.begin() + opts_end, tokens.end());
    tokens = std::move(out);

    // the output has one score per question type
    task.decision.column = question.type;
}

//
// joint prompt (clef)
//

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
static const std::string CLEF_MARKER        = "<<clef:";
static const std::string CLEF_SEP           = "<<clef:sep>>";
static const std::string CLEF_MARK_QUESTION = "<<clef:question>>";
static const std::string CLEF_MARK_OPTION   = "<<clef:option>>";

void server_decision_context::fill_task_joint(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const std::vector<raw_buffer> & files,
        mtmd_context * mctx,
        const mtmd_helper_init_opt & init_opt,
        server_task & task) const {
    if (type == COMMON_DECISION_TYPE_TILDE) {
        if (!files.empty()) {
            throw std::invalid_argument("tilde does not take images");
        }
        fill_task_tilde(state, questions, task);
        return;
    }
    json inp_questions = json::array();
    for (const auto & question : questions) {
        json options = json::array();
        for (const auto & opt : question.options) {
            options.push_back(json{
                {"key",         opt.key},
                {"description", opt.description},
            });
        }
        inp_questions.push_back(json{
            {"id",           question.id},
            {"type",         decision_question_type_name(question.type)},
            {"instructions", question.instructions},
            {"options",      options},
        });
    }

    // the template is given raw JSON values with sorted keys, and no marker in the input
    json inp = json{
        {"state",     state},
        {"questions", inp_questions},
    };
    inp = decision_replace_text(decision_sort_keys(inp), CLEF_MARKER, "<<clef ");

    // the template puts one media marker per image
    json images = json::array();
    if (!files.empty()) {
        inp = decision_replace_text(inp, get_media_marker(), " ");
        for (size_t i = 0; i < files.size(); i++) {
            images.push_back(get_media_marker());
        }
    }
    inp["images"]        = images;
    inp["sep"]           = CLEF_SEP;
    inp["mark_question"] = CLEF_MARK_QUESTION;
    inp["mark_option"]   = CLEF_MARK_OPTION;

    jinja::context ctx(tmpl->source());
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl->prog);
    const std::string prompt   = jinja::runtime::gather_string_parts(results)->as_string().str();

    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

    // the model was trained with the pieces tokenized one by one
    const std::vector<std::string> pieces = string_split(prompt, CLEF_SEP);
    size_t i_piece = 0;

    task.tokens = server_tokens(llama_tokens(), false);
    if (!files.empty()) {
        // the text before the images is tokenized with them, a vision token separates the pieces anyway
        std::string head;
        while (i_piece < pieces.size() && head.find(get_media_marker()) == std::string::npos) {
            head += pieces[i_piece++];
        }
        if (head.find(get_media_marker()) == std::string::npos || head.find(CLEF_MARKER) != std::string::npos) {
            throw invalid;
        }
        task.tokens = process_mtmd_prompt(mctx, head, files, init_opt);
        task.decision.order.resize(task.tokens.size(), LLAMA_DECISION_ORDER_NONE);
    }

    size_t i_question = 0;
    for (; i_piece < pieces.size(); i_piece++) {
        std::string piece = pieces[i_piece];
        int32_t order = LLAMA_DECISION_ORDER_NONE;
        if (string_starts_with(piece, CLEF_MARK_QUESTION)) {
            piece = piece.substr(CLEF_MARK_QUESTION.size());
            if (i_question >= questions.size()) {
                throw invalid;
            }
            switch (questions[i_question++].type) {
                case SERVER_DECISION_QUESTION_NOUL:   order = LLAMA_DECISION_ORDER_QUESTION_NOUL;   break;
                case SERVER_DECISION_QUESTION_CHOICE: order = LLAMA_DECISION_ORDER_QUESTION_CHOICE; break;
                case SERVER_DECISION_QUESTION_SCORE:  order = LLAMA_DECISION_ORDER_QUESTION_SCORE;  break;
            }
        } else if (string_starts_with(piece, CLEF_MARK_OPTION)) {
            piece = piece.substr(CLEF_MARK_OPTION.size());
            order = LLAMA_DECISION_ORDER_OPTION;
            task.decision.n_scores++;
        }

        const llama_tokens piece_tokens = common_tokenize(vocab, piece, false, true);
        if (order != LLAMA_DECISION_ORDER_NONE && piece_tokens.empty()) {
            throw std::invalid_argument("the instructions and the options of a question must not be empty");
        }
        for (const llama_token token : piece_tokens) {
            task.tokens.push_back(token);
        }
        task.decision.order.resize(task.tokens.size(), order);
    }

    size_t n_options = 0;
    for (const auto & question : questions) {
        n_options += question.options.size();
    }
    if (i_question != questions.size() || (size_t) task.decision.n_scores != n_options) {
        throw invalid;
    }
}

//
// answer
//

float server_decision_context::get_temperature(const server_decision_question & question) const {
    const size_t n = question.options.size();
    const std::string type_name = decision_question_type_name(question.type);

    // the temperature can depend on the number of options, the buckets are the ones used to fit it
    std::string bucket;
    if (type == COMMON_DECISION_TYPE_LEV) {
        bucket = n <= 8 ? "small" : n <= 26 ? "mid" : "large";
    } else {
        bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    }

    for (const auto & name : {type_name + "." + bucket, type_name}) {
        const auto it = temperatures.find(name);
        if (it != temperatures.end()) {
            return it->second;
        }
    }
    return 1.0f;
}

// confidence formulas are the ones published by TypeSafe

static double decision_confidence_choice(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const double uniform = 1.0 / probs.size();
    const double p_max   = *std::max_element(probs.begin(), probs.end());
    return std::max(0.0, (p_max - uniform) / (1.0 - uniform));
}

static double decision_confidence_score(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const size_t n    = probs.size();
    const size_t mode = std::max_element(probs.begin(), probs.end()) - probs.begin();

    // mean distance to the mode, relative to the one of a uniform distribution around its center
    double dist         = 0.0;
    double dist_uniform = 0.0;
    for (size_t i = 0; i < n; i++) {
        dist         += probs[i] * std::fabs((double) i - (double) mode);
        dist_uniform += std::fabs((double) i - (n - 1) / 2.0) / n;
    }
    return std::max(0.0, 1.0 - dist / dist_uniform);
}

json server_decision_context::format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const {
    const size_t n = n_outputs(question);
    if (scores.size() != n_variants(question)) {
        throw std::runtime_error("decision result does not match the number of variants");
    }

    // softmax over the outputs of each variant, then the average of the variants
    const float temperature = get_temperature(question);
    std::vector<double> probs(n, 0.0);
    for (size_t v = 0; v < scores.size(); v++) {
        const auto & s = scores[v];
        if (s.size() != n) {
            throw std::runtime_error("decision result does not match the number of options");
        }
        // a joint head returns NaN if it could not use the decision order
        if (std::any_of(s.begin(), s.end(), [](float v) { return std::isnan(v); })) {
            throw std::runtime_error("the model could not evaluate the decision");
        }
        const float score_max = *std::max_element(s.begin(), s.end());
        std::vector<double> p(n);
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            p[i] = std::exp((double) (s[i] - score_max) / temperature);
            sum += p[i];
        }
        for (size_t i = 0; i < n; i++) {
            // the second variant is in the reverse order
            probs[v == 0 ? i : n - 1 - i] += p[i] / sum / scores.size();
        }
    }

    json answer = json{{"type", decision_question_type_name(question.type)}};

    if (type == COMMON_DECISION_TYPE_TILDE) {
        return format_answer_tilde(question, scores[0], answer);
    }

    if (question.type == SERVER_DECISION_QUESTION_NOUL) {
        if (type == COMMON_DECISION_TYPE_LEV) {
            double expected = 0.0;
            for (size_t i = 0; i < n; i++) {
                expected += probs[i] * i / (n - 1);
            }
            answer["noul"] = expected;
            return answer;
        }
        for (size_t i = 0; i < n; i++) {
            if (question.options[i].key == "true") {
                answer["noul"] = probs[i];
            }
        }
        return answer;
    }

    json probabilities = json::object();
    for (size_t i = 0; i < n; i++) {
        probabilities[question.options[i].key] = probs[i];
    }

    if (question.type == SERVER_DECISION_QUESTION_CHOICE) {
        const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
        answer["choice"]        = question.options[best].key;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = decision_confidence_choice(probs);
    } else {
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < n; i++) {
            expected += i * probs[i];
            legend[question.options[i].key] = question.options[i].description;
        }
        answer["score"]         = expected;
        answer["legend"]        = legend;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = decision_confidence_score(probs);
    }
    return answer;
}

//
// shared prompt prefix
//

std::vector<server_task> server_decision_group_tasks(std::vector<server_task> && tasks, size_t n_slots) {
    n_slots = std::max(n_slots, (size_t) 1);

    std::vector<server_task> groups;
    for (size_t i = 0; i < tasks.size(); i += n_slots) {
        const size_t end = std::min(tasks.size(), i + n_slots);
        server_task & parent = tasks[i];

        // every task must have at least one token of its own to evaluate
        size_t n_shared = parent.tokens.size() - 1;
        for (size_t j = i + 1; j < end; j++) {
            n_shared = std::min(n_shared, parent.tokens.get_common_prefix(tasks[j].tokens));
            n_shared = std::min(n_shared, tasks[j].tokens.size() - 1);
        }

        if (end - i < 2 || n_shared == 0) {
            for (size_t j = i; j < end; j++) {
                groups.push_back(std::move(tasks[j]));
            }
            continue;
        }

        parent.n_tokens_shared = n_shared;
        for (size_t j = i + 1; j < end; j++) {
            tasks[j].id_parent = parent.id;
            parent.child_tasks.push_back(std::move(tasks[j]));
        }
        groups.push_back(std::move(parent));
    }
    return groups;
}

//
// tilde
//
// The request is encoded exactly as tilda.joint_serving.encode_request + tilda.clef_joint.encode_joint_record
// (serialization clef_joint_bare_v1, state_first layout): every text piece is tokenized on its own, structure markers
// are inserted by id, non-string values are given to the model as Python json.dumps text.
//

// Python's str.isspace() (and \s of the re module for str patterns)
static bool tilde_is_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// decode one UTF-8 code point at s[i], advance i (the text comes from a parsed JSON string, so it is valid UTF-8)
static uint32_t tilde_next_cp(const std::string & s, size_t & i) {
    const unsigned char c = s[i];
    const int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < len && i + k < s.size(); k++) {
        cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
    }
    i += len;
    return cp;
}

// Python str.strip(): the text without leading and trailing whitespace
static std::string tilde_strip(const std::string & s) {
    size_t start = 0, end = s.size();
    for (size_t i = 0; i < s.size();) {
        const size_t at = i;
        if (!tilde_is_space(tilde_next_cp(s, i))) {
            start = at;
            break;
        }
        start = i;
    }
    // scan forward and remember the end of the last non-space code point
    size_t last = start;
    for (size_t i = start; i < s.size();) {
        if (!tilde_is_space(tilde_next_cp(s, i))) {
            last = i;
        }
    }
    end = last;
    return start < end ? s.substr(start, end - start) : std::string();
}

// Python repr() of a float, as json.dumps writes it: the shortest round-trip digits, fixed notation for decimal
// exponents in [-4, 16), else d.ddde+XX
static std::string tilde_py_float(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    char buf[64];
    // shortest representation that round-trips, in scientific notation: [-]d[.ddd]e[+-]XX
    int n = 0;
    for (int prec = 1; prec <= 17; prec++) {
        n = snprintf(buf, sizeof(buf), "%.*e", prec - 1, v);
        if (std::strtod(buf, nullptr) == v) {
            break;
        }
    }
    std::string sci(buf, n);
    const bool neg = sci[0] == '-';
    if (neg) sci = sci.substr(1);
    const size_t epos = sci.find('e');
    std::string mant = sci.substr(0, epos);
    const int exp = std::atoi(sci.c_str() + epos + 1);
    std::string digits;
    for (char c : mant) if (c != '.') digits += c;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();

    std::string out;
    if (exp >= -4 && exp < 16) {
        if (exp >= 0) {
            if ((int) digits.size() <= exp + 1) {
                out = digits + std::string(exp + 1 - digits.size(), '0') + ".0";
            } else {
                out = digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
            }
        } else {
            out = "0." + std::string(-exp - 1, '0') + digits;
        }
    } else {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char e[16];
        snprintf(e, sizeof(e), "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += e;
    }
    return neg ? "-" + out : out;
}

// Python json.dumps(value, ensure_ascii=False): compact separators (",", ":") when indent < 0,
// else indent spaces per level with separators (",", ": ")
static std::string tilde_py_dumps(const json & v, int indent = -1, int level = 0) {
    if (v.is_null())    return "null";
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_number_float()) return tilde_py_float(v.get<double>());
    if (v.is_number() || v.is_string()) return v.dump(); // exact integers; strings: same escapes as Python
    const bool is_obj = v.is_object();
    if (v.empty()) return is_obj ? "{}" : "[]";
    const std::string nl  = indent < 0 ? "" : "\n" + std::string((size_t) indent * (level + 1), ' ');
    const std::string end = indent < 0 ? "" : "\n" + std::string((size_t) indent * level, ' ');
    std::string out = is_obj ? "{" : "[";
    bool first = true;
    if (is_obj) {
        for (const auto & [key, item] : v.items()) {
            out += (first ? "" : ",") + nl + json(key).dump() + (indent < 0 ? ":" : ": ") + tilde_py_dumps(item, indent, level + 1);
            first = false;
        }
    } else {
        for (const auto & item : v) {
            out += (first ? "" : ",") + nl + tilde_py_dumps(item, indent, level + 1);
            first = false;
        }
    }
    return out + end + (is_obj ? "}" : "]");
}

// tilda.joint_serving.encode_request: literal(value)
static std::string tilde_literal(const json & v) {
    return v.is_string() ? v.get<std::string>() : tilde_py_dumps(v);
}

// Python truthiness of a JSON value
static bool tilde_truthy(const json & v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0.0;
    if (v.is_string()) return !v.get<std::string>().empty();
    if (v.is_array() || v.is_object()) return !v.empty();
    return true;
}

// Python repr() of a JSON value, for error messages
static std::string tilde_py_repr(const json & v) {
    if (v.is_null()) return "None";
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_string()) return "'" + v.get<std::string>() + "'";
    return tilde_py_dumps(v);
}

// tilda.noul_format._BARE: a bare yes / no / true / false (any case, optional trailing .!?, surrounding whitespace)
// returns "true", "false" or "" (not bare)
static std::string tilde_bare_noul(const std::string & text) {
    std::string core = tilde_strip(text);
    while (!core.empty() && (core.back() == '.' || core.back() == '!' || core.back() == '?')) {
        core.pop_back();
    }
    std::string lower;
    for (char c : core) lower += (char) std::tolower((unsigned char) c);
    if (lower == "yes" || lower == "true") return "true";
    if (lower == "no" || lower == "false") return "false";
    return "";
}

// tilda.predict.description_text: the legend of a score level
static std::string tilde_description_text(const json & v) {
    return v.is_string() ? v.get<std::string>() : tilde_py_dumps(v, 2);
}

static float tilde_noul_temperature(const json & v, const std::string & where) {
    if (v.is_boolean() || !v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>() <= 0.0) {
        throw std::invalid_argument(where + " must be a finite number above 0");
    }
    return (float) v.get<double>();
}

std::vector<server_decision_question> server_decision_context::parse_questions_tilde(const json & body) const {
    // tilda/serve.py decide() and tilda.joint_serving.encode_request, with the same checks and messages
    const json state     = body.contains("state")     ? body.at("state")     : json();
    const json questions = body.contains("questions") ? body.at("questions") : json();
    if (!state.is_string() && !state.is_object() && !state.is_array()) {
        throw std::invalid_argument("state must be a string, object or array");
    }
    if (!questions.is_object() || questions.empty() || (max_questions > 0 && questions.size() > max_questions)) {
        throw std::invalid_argument(max_questions > 0 ? string_format("provide 1-%zu questions", max_questions)
                                                      : std::string("provide at least 1 question"));
    }

    std::vector<server_decision_question> out;
    for (const auto & [qid, spec] : questions.items()) {
        if (!spec.is_object()) {
            throw std::invalid_argument("question IDs must be strings and questions must be objects");
        }
        server_decision_question q;
        q.id           = qid;
        q.instructions = spec.contains("instructions") ? spec.at("instructions") : json();
        q.text         = tilde_literal(tilde_truthy(q.instructions) ? q.instructions : json(qid));

        const json kind     = spec.contains("type") ? spec.at("type") : json();
        const json criteria = spec.contains("criteria") ? spec.at("criteria") : json();
        const std::string kind_name = kind.is_string() ? kind.get<std::string>() : "";

        if (kind_name == "choice") {
            q.type = SERVER_DECISION_QUESTION_CHOICE;
            if (!criteria.is_object() || criteria.empty() || criteria.size() > n_options_max) {
                throw std::invalid_argument(string_format("%s: choice criteria require 1 to %zu options", qid.c_str(), n_options_max));
            }
            for (const auto & [key, value] : criteria.items()) {
                q.options.push_back({key, value, value.is_null() ? key : tilde_literal(value)});
            }
        } else if (kind_name == "noul") {
            q.type = SERVER_DECISION_QUESTION_NOUL;
            if (!criteria.is_null() && !criteria.is_object()) {
                throw std::invalid_argument(qid + ": noul criteria must be an object or null");
            }
            // literal(criteria.get(key, key)) for key in ("false", "true")
            std::vector<server_decision_option> opts;
            for (const char * key : {"false", "true"}) {
                const bool has = criteria.is_object() && criteria.contains(key);
                opts.push_back({key, has ? criteria.at(key) : json(), has ? tilde_literal(criteria.at(key)) : std::string(key)});
            }
            if (tilde_noul_normalize) {
                // tilda.noul_format.normalize_noul_example: true first, a bare yes/no text becomes the default text
                static const std::map<std::string, std::string> defaults = {
                    {"true",  "The proposition is true or the answer is yes."},
                    {"false", "The proposition is false or the answer is no."},
                };
                std::swap(opts[0], opts[1]);
                for (auto & opt : opts) {
                    if (tilde_strip(opt.text).empty()) {
                        throw std::invalid_argument("noul descriptions must be nonempty text");
                    }
                    const std::string bare = tilde_bare_noul(opt.text);
                    if (!bare.empty()) {
                        if (bare != opt.key) {
                            throw std::invalid_argument("bare noul description contradicts its semantic key");
                        }
                        opt.text = defaults.at(opt.key);
                    }
                }
            }
            q.options = opts;
        } else if (kind_name == "score") {
            q.type = SERVER_DECISION_QUESTION_SCORE;
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10) {
                throw std::invalid_argument(qid + ": score criteria require 2 to 10 levels");
            }
            for (size_t i = 0; i < criteria.size(); i++) {
                const json & level = criteria.at(i);
                if (!level.is_string() && !level.is_object() && !level.is_array()) {
                    throw std::invalid_argument("criteria descriptions must be strings, objects, or arrays");
                }
                q.options.push_back({std::to_string(i), level, tilde_literal(level)});
                q.legend.push_back(tilde_description_text(level));
            }
        } else {
            throw std::invalid_argument(qid + ": unsupported question type " + tilde_py_repr(kind));
        }

        out.push_back(std::move(q));
    }

    // yes/no temperatures (question field > body field), checked after the questions like tilda/serve.py does
    float default_noul_temperature = 0.0f;
    if (body.contains("noul_temperature") && !body.at("noul_temperature").is_null()) {
        default_noul_temperature = tilde_noul_temperature(body.at("noul_temperature"), "noul_temperature");
    }
    for (auto & q : out) {
        const json & spec = questions.at(q.id);
        q.noul_temperature = q.type == SERVER_DECISION_QUESTION_NOUL ? default_noul_temperature : 0.0f;
        if (spec.contains("noul_temperature") && !spec.at("noul_temperature").is_null()) {
            if (q.type != SERVER_DECISION_QUESTION_NOUL) {
                throw std::invalid_argument(q.id + ": noul_temperature applies to noul questions only");
            }
            q.noul_temperature = tilde_noul_temperature(spec.at("noul_temperature"), q.id + ": noul_temperature");
        }
    }
    return out;
}

void server_decision_context::fill_task_tilde(
        const json & state,
        const std::vector<server_decision_question> & questions,
        server_task & task) const {
    // encode_joint_record, state_first layout, no questions-first prefix:
    // [bos] <start> <state> STATE <state_end> "\n\nSCHEMA FIELDS:\n" FIELD... <decision>
    llama_tokens tokens;
    std::vector<int32_t> order;

    auto raw = [&](const std::string & text, int32_t ord = LLAMA_DECISION_ORDER_NONE) {
        // tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
        const llama_tokens toks = common_tokenize(vocab, text, false, false);
        tokens.insert(tokens.end(), toks.begin(), toks.end());
        order.resize(tokens.size(), ord);
        return toks.size();
    };
    auto marker = [&](size_t i) {
        tokens.push_back(tilde_markers.at(i));
        order.push_back(LLAMA_DECISION_ORDER_NONE);
    };

    if (tilde_bos != LLAMA_TOKEN_NULL) {
        tokens.push_back(tilde_bos);
        order.push_back(LLAMA_DECISION_ORDER_NONE);
    }
    marker(0);
    marker(1);
    raw(tilde_literal(state));
    marker(2);
    raw("\n\nSCHEMA FIELDS:\n");
    const size_t n_base = tokens.size() + 1; // + the decision marker

    size_t n_scores = 0;
    for (size_t i = 0; i < questions.size(); i++) {
        const auto & q = questions[i];
        const size_t field_start = tokens.size();
        int32_t q_order = LLAMA_DECISION_ORDER_QUESTION_NOUL;
        switch (q.type) {
            case SERVER_DECISION_QUESTION_NOUL:   q_order = LLAMA_DECISION_ORDER_QUESTION_NOUL;   break;
            case SERVER_DECISION_QUESTION_CHOICE: q_order = LLAMA_DECISION_ORDER_QUESTION_CHOICE; break;
            case SERVER_DECISION_QUESTION_SCORE:  q_order = LLAMA_DECISION_ORDER_QUESTION_SCORE;  break;
        }
        raw(string_format("\nFIELD %zu\nID: field_%zu\nTYPE: %s\nINSTRUCTION: ", i + 1, i + 1, decision_question_type_name(q.type)));
        marker(3);
        if (raw(q.text, q_order) == 0) {
            throw std::invalid_argument("empty instruction token span");
        }
        raw("\nALLOWED OPTIONS:\n");
        marker(4);
        for (size_t j = 0; j < q.options.size(); j++) {
            raw(string_format("OPTION %zu: ", j + 1));
            // json.dumps({"option_id": key, "description": text}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            raw("{\"description\":" + json(q.options[j].text).dump() + ",\"option_id\":" + json(q.options[j].key).dump() + "}",
                LLAMA_DECISION_ORDER_OPTION);
            raw("\n");
            n_scores++;
        }
        raw("END FIELD\n");

        const size_t n_question = n_base + tokens.size() - field_start;
        if (max_question_tokens > 0 && n_question > max_question_tokens) {
            throw std::invalid_argument(string_format("%s: state and question are %zu tokens; this server accepts at most %zu tokens per question",
                    q.id.c_str(), n_question, max_question_tokens));
        }
    }
    marker(5);

    if (max_request_tokens > 0 && tokens.size() > max_request_tokens) {
        throw std::invalid_argument(string_format("request is %zu tokens (state + schema); this server accepts at most %zu tokens per request",
                tokens.size(), max_request_tokens));
    }

    task.tokens = server_tokens(tokens, false);
    task.decision.order    = std::move(order);
    task.decision.n_scores = (int32_t) n_scores;
}

// tilda.systemone.answer: the response shape of tilda/serve.py
json server_decision_context::format_answer_tilde(const server_decision_question & question, const std::vector<float> & scores, json & answer) const {
    const size_t n = question.options.size();
    if (scores.size() != n) {
        throw std::runtime_error("decision result does not match the number of options");
    }
    if (std::any_of(scores.begin(), scores.end(), [](float v) { return !std::isfinite(v); })) {
        throw std::runtime_error("joint head returned nonfinite logits");
    }
    // the logits are calibrated (the temperature is folded into the head); a decision temperature stored in the
    // file and the yes/no profile of the request divide them once more
    float temperature = get_temperature(question);
    if (question.type == SERVER_DECISION_QUESTION_NOUL && question.noul_temperature > 0.0f) {
        temperature *= question.noul_temperature;
    }
    const float score_max = *std::max_element(scores.begin(), scores.end());
    std::vector<double> probs(n);
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        probs[i] = std::exp((double) (scores[i] - score_max) / temperature);
        sum += probs[i];
    }
    for (auto & p : probs) {
        p /= sum;
    }

    auto confidence = [&]() {
        if (n < 2) return 1.0;
        const double p_max = *std::max_element(probs.begin(), probs.end());
        return std::max(0.0, std::min(1.0, (n * p_max - 1.0) / (n - 1.0)));
    };

    if (question.type == SERVER_DECISION_QUESTION_NOUL) {
        double p_true = 0.0;
        for (size_t i = 0; i < n; i++) {
            if (question.options[i].key == "true") {
                p_true = probs[i];
            }
        }
        answer["noul"] = p_true;
        answer["probabilities"] = json{{"true", p_true}, {"false", 1.0 - p_true}};
        answer["confidence"] = std::fabs(2.0 * p_true - 1.0);
        return answer;
    }

    json probabilities = json::object();
    for (size_t i = 0; i < n; i++) {
        probabilities[question.options[i].key] = probs[i];
    }
    if (question.type == SERVER_DECISION_QUESTION_CHOICE) {
        const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
        answer["choice"]        = question.options[best].key;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = confidence();
        return answer;
    }
    double expected = 0.0;
    json legend = json::object();
    for (size_t i = 0; i < n; i++) {
        expected += i * probs[i];
        legend[std::to_string(i)] = i < question.legend.size() ? question.legend[i] : std::string();
    }
    answer["score"]         = expected;
    answer["legend"]        = legend;
    answer["probabilities"] = probabilities;
    answer["confidence"]    = confidence();
    return answer;
}
