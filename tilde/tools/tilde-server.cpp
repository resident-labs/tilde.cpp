// tilde-server: llama-server with the defaults Tilde needs, chosen from the hardware.
//
//   tilde-server                                   # resident-labs/tilde-8b-a1b-GGUF, quant picked by free memory
//   tilde-server -hf resident-labs/tilde-8b-a1b-GGUF:Q4_KE
//   tilde-server -m tilde-8b-a1b-Q8_0.gguf --port 8080
//
// Every llama-server option still works and wins over the defaults below. What tilde-server adds:
// - the model: the GGUF repo when no -m/-hf is given, and the quant preset by free memory when -hf has no tag;
// - one ubatch that holds the largest request (-c/-b/-ub = the model's request limit), one slot, the model name;
// - TILDE_CHUNK_TOKENS (tokens per chunk of the MoE/FFN blocks) from the memory left after the weights, so that the
//   compute buffer reserved at startup fits and no request can run out of memory later; if even the smallest chunk
//   does not fit, the request limit is lowered (requests above it are rejected, never truncated) and it says so.

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

int llama_server(int argc, char ** argv);

static const char * DEFAULT_REPO = "resident-labs/tilde-8b-a1b-GGUF";
static const double GiB = 1024.0 * 1024.0 * 1024.0;

// approximate sizes of the presets (weights loaded on the GPU; token embeddings stay in host memory)
struct preset {
    const char * tag;
    double       gpu_gib;  // weights on the device
    const char * use;
};
static const preset PRESETS[] = {
    { "Q8_0",  8.6, "default" },
    { "Q4_KE", 5.0, "8-12 GB GPUs" },
};

// compute buffer of the worst-case (largest) request, measured on CUDA (see STATUS.md, "Memory"):
// fixed + per_token * n_tokens + per_chunk_token * min(chunk, n_tokens)
struct compute_model {
    double fixed_gib       = 0.30;
    double per_token_mib   = 0.045; // MLA K/V, KDA projections, head memory: grows with the request
    double per_chunk_mib   = 0.085; // MoE/FFN intermediates: grow with the chunk
};

static bool has_arg(const std::vector<std::string> & args, std::initializer_list<const char *> names) {
    for (const auto & a : args) {
        for (const char * n : names) {
            if (a == n || a.rfind(std::string(n) + "=", 0) == 0) {
                return true;
            }
        }
    }
    return false;
}

static std::string arg_value(const std::vector<std::string> & args, std::initializer_list<const char *> names) {
    for (size_t i = 0; i < args.size(); i++) {
        for (const char * n : names) {
            if (args[i] == n && i + 1 < args.size()) {
                return args[i + 1];
            }
            if (args[i].rfind(std::string(n) + "=", 0) == 0) {
                return args[i].substr(strlen(n) + 1);
            }
        }
    }
    return "";
}

struct device_info {
    std::string name;
    double free_gib  = 0.0;
    double total_gib = 0.0;
    bool   gpu       = false;
};

static device_info best_device() {
    ggml_backend_load_all();
    device_info best;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        if (free / GiB > best.free_gib) {
            best = { ggml_backend_dev_description(dev), free / GiB, total / GiB, true };
        }
    }
    if (!best.gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(dev, &free, &total);
                best = { ggml_backend_dev_description(dev), free / GiB, total / GiB, false };
                break;
            }
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    const bool dry_run = has_arg(args, {"--tilde-plan"});
    args.erase(std::remove(args.begin(), args.end(), std::string("--tilde-plan")), args.end());

    const device_info dev = best_device();
    // keep room for the desktop, the driver and fragmentation
    const double margin_gib = dev.gpu ? 0.75 : 2.0;
    const double budget = std::max(0.0, dev.free_gib - margin_gib);

    // model: an explicit -m wins; -hf without a tag gets the preset that fits
    double weights_gib = 0.0;
    std::string chosen;
    if (!has_arg(args, {"-m", "--model", "-hf", "-hfr", "--hf-repo", "-mu", "--model-url"})) {
        args.insert(args.begin(), {"-hf", DEFAULT_REPO});
    }
    std::string hf = arg_value(args, {"-hf", "-hfr", "--hf-repo"});
    if (!hf.empty() && hf.find(':') == std::string::npos) {
        // the largest preset that leaves room for the compute buffer of a full-size request at the smallest chunk
        const compute_model cm;
        const double need_min = cm.fixed_gib + (cm.per_token_mib * 65536 + cm.per_chunk_mib * 2048) / 1024.0;
        const preset * pick = &PRESETS[sizeof(PRESETS) / sizeof(PRESETS[0]) - 1];
        for (const auto & p : PRESETS) {
            if (p.gpu_gib + need_min <= budget) {
                pick = &p;
                break;
            }
        }
        chosen = pick->tag;
        weights_gib = pick->gpu_gib;
        for (size_t i = 0; i + 1 < args.size(); i++) {
            if (args[i] == "-hf" || args[i] == "-hfr" || args[i] == "--hf-repo") {
                args[i + 1] = hf + ":" + chosen;
            }
        }
    } else {
        const std::string m = arg_value(args, {"-m", "--model"});
        std::error_code ec;
        if (!m.empty() && std::filesystem::exists(m, ec)) {
            weights_gib = std::filesystem::file_size(m, ec) / GiB - 0.9; // token embeddings (bf16) stay on the host
        } else {
            weights_gib = 8.6;
        }
    }

    // request limit and chunk size: the largest request (64k tokens) at the largest chunk that fits
    int64_t n_tokens = 65536;
    int64_t chunk = 0;
    if (const char * env = std::getenv("TILDE_CHUNK_TOKENS")) {
        chunk = std::atoll(env);
    } else {
        const compute_model cm;
        const double left = budget - weights_gib;
        bool fits = false;
        for (int64_t t = n_tokens; t >= 4096 && !fits; t /= 2) {
            for (const int64_t c : {8192, 4096, 2048}) {
                const double need = cm.fixed_gib + (cm.per_token_mib * t + cm.per_chunk_mib * std::min(c, t)) / 1024.0;
                if (need <= left) {
                    n_tokens = t;
                    chunk = c;
                    fits = true;
                    break;
                }
            }
        }
        if (!fits) {
            n_tokens = 4096;
            chunk = 2048;
        }
    }
    char chunk_str[32];
    snprintf(chunk_str, sizeof(chunk_str), "%lld", (long long) chunk);
#ifdef _WIN32
    _putenv_s("TILDE_CHUNK_TOKENS", chunk_str);
#else
    setenv("TILDE_CHUNK_TOKENS", chunk_str, 1);
#endif

    const std::string n_str = std::to_string(n_tokens);
    if (!has_arg(args, {"-c", "--ctx-size"}))       args.insert(args.end(), {"-c", n_str});
    if (!has_arg(args, {"-b", "--batch-size"}))     args.insert(args.end(), {"-b", n_str});
    if (!has_arg(args, {"-ub", "--ubatch-size"}))   args.insert(args.end(), {"-ub", n_str});
    if (!has_arg(args, {"-np", "--parallel"}))      args.insert(args.end(), {"-np", "1"});
    // flash attention is required above 8k tokens; "on" also keeps it when a GPU lacks the kernel (that op then
    // runs on the CPU) instead of turning it off for every layer as "auto" would
    if (!has_arg(args, {"-fa", "--flash-attn"}))    args.insert(args.end(), {"-fa", "on"});
    if (!has_arg(args, {"-ngl", "--gpu-layers", "--n-gpu-layers"})) args.insert(args.end(), {"-ngl", "999"});
    if (!has_arg(args, {"-a", "--alias"}))          args.insert(args.end(), {"--alias", "tilde-8b-a1b"});
    if (n_tokens < 65536 && !has_arg(args, {"--tilde-max-request-tokens"})) {
        // the server rejects requests above the ubatch (never truncates them)
        fprintf(stderr, "tilde-server: %.1f GiB free on %s: requests are limited to %lld tokens\n",
                dev.free_gib, dev.name.c_str(), (long long) n_tokens);
    }

    fprintf(stderr, "tilde-server: device %s (%s, %.1f of %.1f GiB free), preset %s, max request %lld tokens, "
            "chunk %lld tokens\n", dev.name.c_str(), dev.gpu ? "GPU" : "CPU", dev.free_gib, dev.total_gib,
            chosen.empty() ? "(given)" : chosen.c_str(), (long long) n_tokens, (long long) chunk);

    if (dry_run) {
        fprintf(stderr, "tilde-server: llama-server");
        for (const auto & a : args) {
            fprintf(stderr, " %s", a.c_str());
        }
        fprintf(stderr, "\n");
        return 0;
    }

    std::vector<char *> argv2;
    argv2.push_back(argv[0]);
    for (auto & a : args) {
        argv2.push_back(a.data());
    }
    argv2.push_back(nullptr);
    return llama_server((int) argv2.size() - 1, argv2.data());
}
