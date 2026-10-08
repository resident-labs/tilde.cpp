// tilde-eval: run a Tilde GGUF over pre-encoded records and write the raw head logits.
//
// Input: JSON lines, one record each:
//   {"id": "...", "input_ids": [...], "questions": [{"type": 0|1|2, "span": [s, e], "options": [[s, e], ...]}, ...]}
// (type: 0 noul, 1 choice, 2 score; spans are [start, end) token indices, exactly tilda.clef_joint.EncodedRecord.)
// Output: JSON lines {"id": ..., "n_tokens": N, "ms": ..., "logits": [[...], ...]} (one list per question).
//
// This is the parity harness: the records come from the reference encoder, so the comparison isolates the model.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../../src/llama-ext.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct record {
    std::string id;
    std::vector<llama_token> ids;
    struct question {
        int type;
        int start, end;
        std::vector<std::pair<int, int>> options;
    };
    std::vector<question> questions;
};

static record parse_record(const json & j) {
    record r;
    r.id = j.contains("id") ? (j["id"].is_string() ? j["id"].get<std::string>() : j["id"].dump()) : "";
    for (const auto & t : j.at("input_ids")) {
        r.ids.push_back(t.get<int32_t>());
    }
    for (const auto & q : j.at("questions")) {
        record::question qq;
        qq.type  = q.at("type").get<int>();
        qq.start = q.at("span")[0].get<int>();
        qq.end   = q.at("span")[1].get<int>();
        for (const auto & o : q.at("options")) {
            qq.options.push_back({ o[0].get<int>(), o[1].get<int>() });
        }
        r.questions.push_back(std::move(qq));
    }
    return r;
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf --records IN.jsonl --out OUT.jsonl [-ngl N] [-t THREADS] [-fa on|off|auto]\n"
        "          [--max-tokens N] [--limit N] [--repeat N]\n", argv0);
}

int main(int argc, char ** argv) {
    std::string model_path, records_path, out_path, fa = "auto";
    int ngl = 999, threads = -1, max_tokens = 0, limit = -1, repeat = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "--records") records_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "-ngl") ngl = std::stoi(next());
        else if (a == "-t") threads = std::stoi(next());
        else if (a == "-fa") fa = next();
        else if (a == "--max-tokens") max_tokens = std::stoi(next());
        else if (a == "--limit") limit = std::stoi(next());
        else if (a == "--repeat") repeat = std::stoi(next());
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || records_path.empty() || out_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::vector<record> records;
    {
        std::ifstream in(records_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            records.push_back(parse_record(json::parse(line)));
            if (limit > 0 && (int) records.size() >= limit) break;
        }
    }
    int longest = 0;
    for (const auto & r : records) longest = std::max(longest, (int) r.ids.size());
    if (max_tokens <= 0) max_tokens = longest;
    fprintf(stderr, "%zu records, longest %d tokens\n", records.size(), longest);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx        = max_tokens;
    cparams.n_batch      = max_tokens;
    cparams.n_ubatch     = max_tokens;
    cparams.n_seq_max    = 1;
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (threads > 0) {
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;
    }
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    std::ofstream out(out_path);
    common_batch batch(ctx.get());
    double total_ms = 0;
    for (const auto & r : records) {
        if ((int) r.ids.size() > max_tokens) {
            fprintf(stderr, "%s: %zu tokens > %d, skipped\n", r.id.c_str(), r.ids.size(), max_tokens);
            continue;
        }
        std::vector<int32_t> order(r.ids.size(), LLAMA_DECISION_ORDER_NONE);
        size_t n_options = 0;
        for (const auto & q : r.questions) {
            for (int t = q.start; t < q.end; t++) order[t] = LLAMA_DECISION_ORDER_QUESTION_NOUL + q.type;
            for (const auto & o : q.options) {
                for (int t = o.first; t < o.second; t++) order[t] = LLAMA_DECISION_ORDER_OPTION;
                n_options++;
            }
        }
        std::vector<float> scores;
        double ms = 0;
        for (int rep = 0; rep < repeat; rep++) {
            batch.clear();
            for (size_t i = 0; i < r.ids.size(); i++) {
                const int32_t idx = batch.add(r.ids[i], (llama_pos) i, 0, true);
                batch.tokens[idx].decision_order = order[i];
            }
            const auto t0 = std::chrono::steady_clock::now();
            const int ret = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                fprintf(stderr, "%s: llama_process failed (%d)\n", r.id.c_str(), ret);
                return 1;
            }
            const float * embd = llama_get_embeddings(ctx.get());
            scores.assign(embd, embd + n_options);
            const auto t1 = std::chrono::steady_clock::now();
            ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        total_ms += ms;
        json logits = json::array();
        size_t k = 0;
        for (const auto & q : r.questions) {
            json row = json::array();
            for (size_t j = 0; j < q.options.size(); j++) row.push_back(scores[k++]);
            logits.push_back(row);
        }
        out << json{{"id", r.id}, {"n_tokens", r.ids.size()}, {"ms", ms}, {"logits", logits}}.dump() << "\n";
        out.flush();
    }
    fprintf(stderr, "done: %.1f s total model time\n", total_ms / 1000.0);
    llama_backend_free();
    return 0;
}
