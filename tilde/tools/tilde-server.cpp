// tilde-server: llama-server with the defaults Tilde needs, chosen from the hardware.
//
//   tilde-server                                   # resident-labs/tilde-8b-a1b-GGUF, quant picked by free memory
//   tilde-server -hf resident-labs/tilde-8b-a1b-GGUF:Q4_KE
//   tilde-server -m tilde-8b-a1b-Q8_0.gguf --port 8080
//
// Every llama-server option still works and wins over the defaults below. What tilde-server adds:
// - the model: the GGUF repo when no -m/-hf is given, and the quant preset by free memory when -hf has no tag;
// - one ubatch that holds the largest request (-c/-b/-ub = the request limit), one slot, flash attention, the model name;
// - the request limit from the memory left after the weights. Tilde reads the whole request at once (the MLA layers
//   are bidirectional), so it cannot be split into smaller micro-batches; the compute buffer for the largest request
//   is reserved at startup, so no request within the limit can run out of memory later. When the memory is too small
//   for 65,536 tokens, the limit is lowered (and printed); longer requests are rejected with 400, never truncated.

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

// device memory of the presets' weights (MiB, measured on CUDA: the token embeddings stay in host memory)
struct preset {
    const char * tag;
    double       weights_mib;
    const char * use;
};
static const preset PRESETS[] = {
    { "Q8_0",  8389.0, "default" },
    { "Q4_KE", 5077.0, "8-12 GB GPUs, Macs" },
};

// device memory beyond the weights for a request limit of n tokens (MiB), measured on a 3090 Ti (STATUS.md, "Memory"):
// the compute buffer reserved at startup (~0.0675 MiB per token) plus the CUDA context and the attention/delta-net
// temporaries (~440 MiB + ~0.004 MiB per token), rounded up
static double runtime_mib(int64_t n_tokens) {
    return 640.0 + 0.0715 * (double) n_tokens;
}

// the file bytes that are not loaded on the device (token embeddings, metadata), MiB
static const double HOST_ONLY_MIB = 470.0;

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

static device_info best_device(bool cpu_only) {
    ggml_backend_load_all();
    device_info best;
    for (size_t i = 0; i < ggml_backend_dev_count() && !cpu_only; i++) {
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
#ifdef __linux__
        // the CPU device reports the physical memory; what can be used is MemAvailable
        if (FILE * f = fopen("/proc/meminfo", "r")) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                long long kb = 0;
                if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) {
                    best.free_gib = kb * 1024.0 / GiB;
                    break;
                }
            }
            fclose(f);
        }
#endif
    }
    return best;
}

int main(int argc, char ** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    const bool dry_run = has_arg(args, {"--tilde-plan"});
    args.erase(std::remove(args.begin(), args.end(), std::string("--tilde-plan")), args.end());

    // -ngl 0 or --device none: the model runs on the CPU, so plan with host memory
    const std::string ngl_arg = arg_value(args, {"-ngl", "--gpu-layers", "--n-gpu-layers"});
    const std::string dev_arg = arg_value(args, {"-dev", "--device"});
    const bool cpu_only = (!ngl_arg.empty() && std::atoi(ngl_arg.c_str()) == 0 && ngl_arg != "auto" && ngl_arg != "all") ||
                          dev_arg == "none";
    const device_info dev = best_device(cpu_only);
    // keep room for fragmentation (and, on the CPU, for the rest of the system)
    const double margin_mib = dev.gpu ? 256.0 : 2048.0;
    const double budget_mib = std::max(0.0, dev.free_gib * 1024.0 - margin_mib);
    auto max_tokens = [&](double weights_mib) -> int64_t {
        const double left = budget_mib - weights_mib - runtime_mib(0);
        const int64_t n = left > 0 ? (int64_t) (left / (runtime_mib(1) - runtime_mib(0))) : 0;
        return std::min<int64_t>(65536, n / 1024 * 1024);
    };

    // model: an explicit -m wins; -hf without a tag gets the largest preset that takes 65,536-token requests, else the
    // smallest preset (with a lower request limit)
    double weights_mib = 0.0;
    std::string chosen;
    if (!has_arg(args, {"-m", "--model", "-hf", "-hfr", "--hf-repo", "-mu", "--model-url"})) {
        args.insert(args.begin(), {"-hf", DEFAULT_REPO});
    }
    const std::string m  = arg_value(args, {"-m", "--model"});
    const std::string hf = arg_value(args, {"-hf", "-hfr", "--hf-repo"});
    // on the CPU the whole file is in memory (the device figures leave the token embeddings out)
    const double host_extra = dev.gpu ? 0.0 : HOST_ONLY_MIB;
    std::error_code ec;
    const auto m_size = m.empty() ? 0 : std::filesystem::file_size(m, ec);
    if (!m.empty() && !ec) {
        // a local file (wins over -hf when both are given)
        weights_mib = m_size / (1024.0 * 1024.0) - (dev.gpu ? HOST_ONLY_MIB : 0.0);
    } else if (!hf.empty() && hf.find(':') == std::string::npos) {
        const size_t n_presets = sizeof(PRESETS) / sizeof(PRESETS[0]);
        const preset * pick = &PRESETS[n_presets - 1];
        for (const auto & p : PRESETS) {
            if (max_tokens(p.weights_mib + host_extra) >= 65536) {
                pick = &p;
                break;
            }
        }
        chosen = pick->tag;
        weights_mib = pick->weights_mib + host_extra;
        for (size_t i = 0; i + 1 < args.size(); i++) {
            if (args[i] == "-hf" || args[i] == "-hfr" || args[i] == "--hf-repo") {
                args[i + 1] = hf + ":" + chosen;
            }
        }
    } else {
        // a tagged -hf (repo:PRESET): the size of that preset, the default one for an unknown tag
        const std::string tag = hf.find(':') != std::string::npos ? hf.substr(hf.find(':') + 1) : "";
        weights_mib = PRESETS[0].weights_mib + host_extra;
        for (const auto & p : PRESETS) {
            if (tag == p.tag) {
                weights_mib = p.weights_mib + host_extra;
                chosen = p.tag;
            }
        }
    }

    int64_t n_tokens = max_tokens(weights_mib);
    if (n_tokens < 1024) {
        fprintf(stderr, "tilde-server: %.1f GiB free on %s is not enough for the model (%.1f GiB) and a 1,024-token "
                "request\n", dev.free_gib, dev.name.c_str(), weights_mib / 1024.0);
        if (!dry_run) {
            return 1;
        }
    }

    // a smaller -c, -b or -ub given by the user bounds the request size (the server rejects longer requests)
    for (const auto & a : { arg_value(args, {"-c", "--ctx-size"}), arg_value(args, {"-b", "--batch-size"}),
                            arg_value(args, {"-ub", "--ubatch-size"}) }) {
        if (!a.empty() && std::atoll(a.c_str()) > 0) {
            n_tokens = std::min<int64_t>(n_tokens, std::atoll(a.c_str()));
        }
    }
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
    if (n_tokens < 65536) {
        // the server rejects requests above the ubatch (never truncates them)
        fprintf(stderr, "tilde-server: requests are limited to %lld tokens (%.1f GiB free on %s, or a smaller -c/-b/-ub)\n",
                (long long) n_tokens, dev.free_gib, dev.name.c_str());
    }

    fprintf(stderr, "tilde-server: device %s (%s, %.1f of %.1f GiB free), preset %s (%.1f GiB), max request %lld "
            "tokens\n", dev.name.c_str(), dev.gpu ? "GPU" : "CPU", dev.free_gib, dev.total_gib,
            chosen.empty() ? "(given)" : chosen.c_str(), weights_mib / 1024.0, (long long) n_tokens);

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
