// tilde-encode: encode /v1/systemone request bodies exactly as llama-server does for a tilde model, and print the
// token ids, the decision order of each token and the error (if the request is rejected). Test harness for the
// request encoder against tilda.joint_serving.encode_request (tilde/parity/encoder_parity.py).
//
//   tilde-encode -m MODEL.gguf < requests.jsonl > encoded.jsonl      (one JSON body per line)

#include "server-decision.h"

#include "llama.h"

#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    std::string model_path;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m MODEL.gguf < requests.jsonl\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        return 1;
    }
    server_decision_context decision;
    decision.init(model);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        json out = json::object();
        try {
            const json body = json::parse(line);
            const auto questions = decision.parse_questions(body);
            std::vector<raw_buffer> files;
            const json state = decision.parse_state(body, files);
            server_task task(SERVER_TASK_TYPE_DECISION);
            decision.fill_task_joint(state, questions, files, nullptr, mtmd_helper_init_opt(), task);
            json ids = json::array();
            for (size_t i = 0; i < task.tokens.size(); i++) {
                ids.push_back(task.tokens[i]);
            }
            json order = json::array();
            for (const int32_t o : task.decision.order) {
                order.push_back(o);
            }
            out["input_ids"] = ids;
            out["order"]     = order;
            out["n_scores"]  = task.decision.n_scores;
        } catch (const std::exception & e) {
            out["error"] = e.what();
        }
        std::cout << out.dump() << "\n";
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
