// tilde-server: llama-server with the defaults Tilde needs, chosen from the hardware.
//
//   tilde-server                                   # resident-labs/tilde-8b-a1b-GGUF, preset picked from the hardware
//   tilde-server -hf resident-labs/tilde-8b-a1b-GGUF:Q4_KE
//   tilde-server -m tilde-8b-a1b-Q8_0.gguf --port 8080
//
// Every llama-server option still works and wins over the defaults below. What tilde-server adds:
// - the model: the GGUF repo when no -m/-hf is given, and the preset when -hf has no tag: I8CX with the
//   SageAttention2-style attention (TILDE_FA_FAST=1) on a GPU that runs the int8 GEMMs (NVIDIA Ampere or newer, CUDA
//   build with CUTLASS) and has the memory for 65,536-token requests; otherwise Q8_0, or Q4_KE when memory is short.
//   The choice and the reason are logged and reported by GET /health. A tag, -m and TILDE_FA_FAST=0 override it;
//   --tilde-models DIR (or TILDE_MODELS) takes the presets from local files DIR/tilde-8b-a1b-<PRESET>.gguf;
// - one ubatch that holds the largest request (-c/-b/-ub = the request limit), one slot, flash attention, the model name;
// - the request limit from the memory left after the weights. Tilde reads the whole request at once (the MLA layers
//   are bidirectional), so it cannot be split into smaller micro-batches; the compute buffer for the largest request
//   is reserved at startup, so no request within the limit can run out of memory later. When the memory is too small
//   for 65,536 tokens, the limit is lowered (and printed); longer requests are rejected with 400, never truncated.

#include "ggml.h"
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
    double       extra_mib;  // runtime memory beyond runtime_mib() at a 65,536-token limit (int8 activation caches)
    bool         auto_pick;  // chosen by free memory for an untagged -hf (I8CX needs a CUDA build with CUTLASS)
    const char * use;
};
static const preset PRESETS[] = {
    { "Q8_0",  8389.0,   0.0, true,  "default" },
    { "Q4_KE", 5077.0,   0.0, true,  "8-12 GB GPUs, Macs" },
    { "I8CX",  7977.0, 600.0, false, "CUDA (Ampere or newer, built with GGML_CUDA_CUTLASS_DIR): fastest; TILDE_FA_FAST=1" },
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
    std::string backend;     // CUDA, Vulkan, Metal, ROCm, CPU, ...
    bool   int8_gemm = false; // runs I8 MUL_MAT and MUL_MAT_ID (the I8CX preset): NVIDIA Ampere+ with CUTLASS
};

// whether the device runs the I8CX products: an I8 MUL_MAT and an I8 MUL_MAT_ID with row scales (CUDA: Ampere or
// newer, built with GGML_CUDA_CUTLASS_DIR; no other backend has them)
static bool device_runs_i8(ggml_backend_dev_t dev) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor * w    = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 256, 256);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 512);
    ggml_tensor * mm   = ggml_mul_mat(ctx, w, x);
    ggml_tensor * we   = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, 256, 256, 8);
    ggml_tensor * ws   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 8);
    ggml_tensor * xe   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 512);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 512);
    ggml_tensor * mmid = ggml_mul_mat_id_scaled(ctx, we, ws, xe, ids);
    const bool ok = ggml_backend_dev_supports_op(dev, mm) && ggml_backend_dev_supports_op(dev, mmid);
    ggml_free(ctx);
    return ok;
}

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
            best = { ggml_backend_dev_description(dev), free / GiB, total / GiB, true,
                     ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), device_runs_i8(dev) };
        }
    }
    if (!best.gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(dev, &free, &total);
                best = { ggml_backend_dev_description(dev), free / GiB, total / GiB, false, "CPU", false };
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

    // --tilde-models DIR (or TILDE_MODELS): the presets are local files DIR/tilde-8b-a1b-<PRESET>.gguf; the pick is the
    // same as for the GGUF repo, falling back to the next preset that is on disk
    std::string models_dir;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--tilde-models" && i + 1 < args.size()) {
            models_dir = args[i + 1];
            args.erase(args.begin() + i, args.begin() + i + 2);
            break;
        }
    }
    if (models_dir.empty() && getenv("TILDE_MODELS") != nullptr) {
        models_dir = getenv("TILDE_MODELS");
    }

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

    // model: an explicit -m wins; -hf without a tag gets I8CX when the device runs its int8 GEMMs and has the memory for
    // 65,536-token requests, else the largest of Q8_0 / Q4_KE that takes them, else Q4_KE (with a lower request limit)
    double weights_mib = 0.0;
    std::string chosen;
    std::string reason;
    char buf[512];
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
        if (m.find("I8C") != std::string::npos) {
            weights_mib += PRESETS[2].extra_mib; // int8 activation caches of the I8 GEMMs
            chosen = "I8CX";
        }
        reason = "given with -m";
    } else if (!hf.empty() && hf.find(':') == std::string::npos) {
        const preset & i8cx = PRESETS[2];
        const preset * pick = &PRESETS[1];
        if (dev.int8_gemm && max_tokens(i8cx.weights_mib + i8cx.extra_mib + host_extra) >= 65536) {
            pick = &i8cx;
            snprintf(buf, sizeof(buf), "%s device runs the int8 GEMMs (Ampere or newer, CUTLASS build) and %.1f GiB free holds "
                     "I8CX for 65,536-token requests", dev.backend.c_str(), dev.free_gib);
        } else {
            for (const auto & p : PRESETS) {
                if (p.auto_pick && max_tokens(p.weights_mib + p.extra_mib + host_extra) >= 65536) {
                    pick = &p;
                    break;
                }
            }
            snprintf(buf, sizeof(buf), "%s %s; %.1f GiB free %s", dev.backend.c_str(),
                     !dev.gpu ? "(no GPU)" : dev.int8_gemm ? "runs the int8 GEMMs but has too little memory for I8CX at 65,536 tokens"
                                                           : "does not run the I8CX int8 GEMMs (needs NVIDIA Ampere or newer and a CUTLASS build)",
                     dev.free_gib, pick == &PRESETS[0] ? "holds Q8_0 for 65,536-token requests" :
                                   "is short for Q8_0 at 65,536 tokens: the smaller Q4_KE");
        }
        reason = buf;
        if (!models_dir.empty()) {
            // local presets: the pick if its file is there, else Q8_0, else Q4_KE (whichever exists first)
            auto file = [&](const char * tag) { return models_dir + "/tilde-8b-a1b-" + tag + ".gguf"; };
            const preset * local = nullptr;
            for (const preset * c : { pick, &PRESETS[0], &PRESETS[1] }) {
                if (std::filesystem::exists(file(c->tag))) {
                    local = c;
                    break;
                }
            }
            if (local == nullptr) {
                fprintf(stderr, "tilde-server: no tilde-8b-a1b-{%s,Q8_0,Q4_KE}.gguf in %s\n", pick->tag, models_dir.c_str());
                return 1;
            }
            if (local != pick) {
                reason += std::string("; ") + pick->tag + " is not in " + models_dir + ", using " + local->tag;
                pick = local;
            }
            for (size_t i = 0; i + 1 < args.size(); i++) {
                if (args[i] == "-hf" || args[i] == "-hfr" || args[i] == "--hf-repo") {
                    args[i] = "-m";
                    args[i + 1] = file(pick->tag);
                }
            }
        }
        chosen = pick->tag;
        weights_mib = pick->weights_mib + pick->extra_mib + host_extra;
        for (size_t i = 0; i + 1 < args.size() && models_dir.empty(); i++) {
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
                weights_mib = p.weights_mib + p.extra_mib + host_extra;
                chosen = p.tag;
            }
        }
        reason = "given with a preset tag";
    }

    // I8CX runs the SageAttention2-style attention unless TILDE_FA_FAST is set (TILDE_FA_FAST=0 keeps the int8 Q*K^T
    // with f16 P*V)
    std::string attention = "default (int8 Q*K^T where supported)";
    if (chosen == "I8CX" && dev.int8_gemm) {
        setenv("TILDE_FA_FAST", "1", /*overwrite =*/ 0);
    }
    if (const char * f = getenv("TILDE_FA_FAST"); f != nullptr && atoi(f) != 0 && dev.int8_gemm) {
        attention = "fast (SageAttention2-style: int8 Q*K^T, f16 P*V)";
    } else if (const char * p8 = getenv("TILDE_FA_PV8"); p8 != nullptr && atoi(p8) != 0 && dev.int8_gemm) {
        attention = "int8 Q*K^T and P*V";
    }
    if (chosen == "I8CX" && !dev.int8_gemm) {
        fprintf(stderr, "tilde-server: warning: I8CX needs NVIDIA Ampere or newer and a CUDA build with CUTLASS; %s cannot "
                "run its int8 GEMMs\n", dev.name.c_str());
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

    fprintf(stderr, "tilde-server: device %s (%s, %s, %.1f of %.1f GiB free), preset %s (%.1f GiB), attention %s, max "
            "request %lld tokens\n", dev.name.c_str(), dev.gpu ? "GPU" : "CPU", dev.backend.c_str(), dev.free_gib,
            dev.total_gib, chosen.empty() ? "(given)" : chosen.c_str(), weights_mib / 1024.0, attention.c_str(),
            (long long) n_tokens);
    fprintf(stderr, "tilde-server: preset %s: %s\n", chosen.empty() ? "(given)" : chosen.c_str(), reason.c_str());
    // reported by GET /health
    setenv("TILDE_SERVER_PRESET", chosen.empty() ? "(given)" : chosen.c_str(), 1);
    setenv("TILDE_SERVER_PRESET_REASON", reason.c_str(), 1);
    setenv("TILDE_SERVER_ATTENTION", attention.c_str(), 1);
    setenv("TILDE_SERVER_DEVICE", (dev.name + " (" + dev.backend + ")").c_str(), 1);
    setenv("TILDE_SERVER_MAX_TOKENS", std::to_string(n_tokens).c_str(), 1);

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
