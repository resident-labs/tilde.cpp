#!/usr/bin/env bash
# Tilde on a fresh GPU box (made for the RTX 5090 test): build tilde.cpp from a pushed commit, check the CUDA kernels,
# DEV-slice parity per preset, and the 1k..64k latency sweep per preset. Everything comes from the private staging
# repo on Hugging Face; results go back there (results/<gpu>-<date>/) and to $WORK/results.
#
#   export HF_TOKEN=...            # read + write on the staging repo
#   bash run_gpu_box.sh            # or: curl/hf download the script, then run it
#
# Settings (environment):
#   STAGING   resident-labs/tilde-8b-a1b-GGUF-staging
#   COMMIT    tilde.cpp commit of the source tarball in the staging repo (src/tilde.cpp-<COMMIT>.tar.gz); default: the
#             one named in src/LATEST there
#   ARCH      CMAKE_CUDA_ARCHITECTURES (120a-real for the 5090: the FP4 tensor-core instructions need the "a" target)
#   PRESETS   GGUF presets to run ("Q8_0 Q4_KE I8CX NVFP4E"); I8CX also runs with int8 P*V (I8CX+PV8)
#   WORK      working directory ($HOME/tilde-box)
#   JOBS      build parallelism (nproc)
#   UPLOAD    1 = upload the results to the staging repo (default 1)
#   GGUF_DIR  (dry runs) take the GGUFs from this directory instead of downloading them
# Needs: NVIDIA driver + CUDA toolkit >= 12.8 (nvcc), git, python3 (pip is used for cmake / hf if they are missing).
set -euo pipefail

STAGING=${STAGING:-resident-labs/tilde-8b-a1b-GGUF-staging}
COMMIT=${COMMIT:-}
ARCH=${ARCH:-120a-real}
PRESETS=${PRESETS:-"Q8_0 Q4_KE I8CX NVFP4E"}
WORK=${WORK:-$HOME/tilde-box}
JOBS=${JOBS:-$(nproc)}
UPLOAD=${UPLOAD:-1}
CUTLASS_TAG=v3.9.2

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "error: $*" >&2; exit 1; }

mkdir -p "$WORK" && cd "$WORK"
RES=$WORK/results
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
T0=$(date +%s)

# --- tools -----------------------------------------------------------------------------------------------------------
log "box: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1), $(nproc) cpus"
[ -n "${HF_TOKEN:-}" ] || die "set HF_TOKEN (read/write on $STAGING)"
# the toolkit's own nvcc (a bare nvcc on PATH may be a wrapper without the toolkit headers next to it)
NVCC=${NVCC:-}
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
[ -z "$NVCC" ] && NVCC=$(command -v nvcc || true)
[ -n "$NVCC" ] || die "no nvcc (CUDA toolkit >= 12.8 needed)"
export PATH=$(dirname "$NVCC"):$PATH
CUDA_VER=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
log "nvcc $CUDA_VER ($NVCC)"
python3 - "$CUDA_VER" <<'PY' || die "CUDA >= 12.8 needed for sm_120"
import sys
major, minor = map(int, sys.argv[1].split("."))
sys.exit(0 if (major, minor) >= (12, 8) else 1)
PY
PIP="python3 -m pip install --quiet --user"
command -v cmake > /dev/null || { log "installing cmake"; $PIP cmake; export PATH=$HOME/.local/bin:$PATH; }
command -v hf > /dev/null || { log "installing huggingface_hub"; $PIP -U "huggingface_hub[cli]"; export PATH=$HOME/.local/bin:$PATH; }
command -v git > /dev/null || die "no git"

# --- inputs ----------------------------------------------------------------------------------------------------------
# one pattern per call: hf download ignores --include when a file name is given
fetch() { hf download "$STAGING" --include "$1" --local-dir "$WORK/staging" > /dev/null; }
log "downloading source, data and GGUFs from $STAGING"
if [ -z "$COMMIT" ]; then
    fetch "src/LATEST"
    COMMIT=$(tr -d ' \n' < "$WORK/staging/src/LATEST")
fi
log "commit $COMMIT"
fetch "src/tilde.cpp-$COMMIT.tar.gz"
fetch "data/*"
for p in $PRESETS; do
    if [ -n "${GGUF_DIR:-}" ]; then
        # dry runs: GGUFs already on disk
        mkdir -p "$WORK/staging/gguf" && ln -sf "$GGUF_DIR/tilde-8b-a1b-$p.gguf" "$WORK/staging/gguf/"
    else
        fetch "gguf/tilde-8b-a1b-$p.gguf"
    fi
done
S=$WORK/staging
[ -f "$S/src/tilde.cpp-$COMMIT.tar.gz" ] || die "source tarball missing"
for p in $PRESETS; do [ -s "$S/gguf/tilde-8b-a1b-$p.gguf" ] || die "gguf $p missing"; done
log "inputs ok ($(du -sh "$S" | cut -f1))"

# --- build -----------------------------------------------------------------------------------------------------------
rm -rf "$WORK/src" && mkdir -p "$WORK/src"
tar -xzf "$S/src/tilde.cpp-$COMMIT.tar.gz" -C "$WORK/src"
[ -d "$WORK/cutlass" ] || git clone --quiet --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass "$WORK/cutlass"
log "building $COMMIT for $ARCH with $JOBS jobs"
cmake -S "$WORK/src" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DGGML_CUDA_CUTLASS_DIR="$WORK/cutlass" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON > "$RES/cmake.log" 2>&1 \
    || { tail -30 "$RES/cmake.log"; die "cmake failed"; }
cmake --build "$WORK/build" -j "$JOBS" --target tilde-eval tilde-server test-backend-ops > "$RES/build.log" 2>&1 \
    || { grep -E "error" "$RES/build.log" | head -30; die "build failed"; }
B=$WORK/build/bin
log "build done ($(( $(date +%s) - T0 )) s since start)"

# --- kernel tests (CUDA vs CPU) --------------------------------------------------------------------------------------
log "kernel tests"
TB=$B/test-backend-ops
{
    $TB test -b CUDA0 -o "MOE_GROUP_ROUTE,NORM_I8,RMS_NORM_MUL,RMS_NORM,GATED_DELTA_NET,SSM_CONV"
    $TB test -b CUDA0 -o "FLASH_ATTN_EXT" -p "acc16|pv8"
    $TB test -b CUDA0 -o "ADD,MUL_MAT_ID" -p "n_mats|scaled"
    $TB test -b CUDA0 -o "MUL_MAT" -p "type_a=(nvfp4|mxfp4|q8_0|q4_K),.*n=(1|8|300)[,)]"
} > "$RES/tests.log" 2>&1 || true
grep -E "tests passed|FAIL" "$RES/tests.log" | sort | uniq -c | tee "$RES/tests_summary.txt"

# --- parity and latency per preset ----------------------------------------------------------------------------------
RUNS=""
for p in $PRESETS; do
    RUNS="$RUNS $p"
    [ "$p" = I8CX ] && RUNS="$RUNS I8CX+PV8"
done
# peak VRAM per run: sample nvidia-smi while each preset runs
vram_watch() { while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 0.5; done; }
for r in $RUNS; do
    p=${r%+PV8}
    pv=0; [ "$r" != "$p" ] && pv=1
    M=$S/gguf/tilde-8b-a1b-$p.gguf
    log "$r: DEV slice parity"
    TILDE_FA_PV8=$pv $B/tilde-eval -m "$M" --records "$S/data/dev_slice.jsonl" --out "$RES/par_$r.jsonl" -ngl 999 \
        > "$RES/par_$r.log" 2>&1 || { tail -20 "$RES/par_$r.log"; die "tilde-eval failed ($r parity)"; }
    python3 "$WORK/src/tilde/parity/reference.py" compare --records "$S/data/dev_slice.jsonl" --ref "$S/data/ref_dev_slice.jsonl" \
        --test "$RES/par_$r.jsonl" --json "$RES/par_$r.json" > /dev/null
    log "$r: 1k..64k sweep"
    vram_watch > "$RES/vram_$r.txt" & W=$!
    TILDE_FA_PV8=$pv $B/tilde-eval -m "$M" --records "$S/data/bench_records.jsonl" --out "$RES/time_$r.jsonl" -ngl 999 \
        --repeat 3 --max-tokens 65536 > "$RES/time_$r.log" 2>&1 || { kill $W; tail -20 "$RES/time_$r.log"; die "tilde-eval failed ($r sweep)"; }
    kill $W 2>/dev/null || true
done

# --- summary ---------------------------------------------------------------------------------------------------------
GATE_ARGS=""
for r in $RUNS; do GATE_ARGS="$GATE_ARGS $RES/par_$r.json:$r"; done
{
    echo "# tilde.cpp $COMMIT on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo
    echo "CUDA $CUDA_VER, arch $ARCH, driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), $(nproc) cpus, $(date -u +%Y-%m-%dT%H:%MZ)"
    echo
    echo "## Kernel tests"
    echo '```'; cat "$RES/tests_summary.txt"; echo '```'
    echo
    echo "## DEV slice parity ($(wc -l < "$S/data/dev_slice.jsonl") records) vs the bf16 PyTorch reference"
    echo
    python3 "$WORK/src/tilde/parity/gates.py" $GATE_ARGS
    echo
    echo "## Latency (tilde-eval warm, ms, best of 3) and peak VRAM"
    echo
    RES=$RES RUNS="$RUNS" S=$S python3 - <<'PY'
import json, os
res, runs, s = os.environ["RES"], os.environ["RUNS"].split(), os.environ["S"]
def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}
t = {r: load(f"{res}/time_{r}.jsonl") for r in runs}
ref3090 = load(f"{s}/data/3090ti_I8CX_bench.jsonl")
ids = list(next(iter(t.values())).keys())
print("| request | tokens | " + " | ".join(runs) + " | 3090 Ti I8CX |")
print("|---|---:|" + "---:|" * (len(runs) + 1))
for i in ids:
    n = next(iter(t.values()))[i]["n_tokens"]
    row = [f"{t[r][i]['ms']:,.0f}" if i in t[r] else "-" for r in runs]
    ref = f"{ref3090[i]['ms']:,.0f}" if i in ref3090 else "-"
    print(f"| {i} | {n:,} | " + " | ".join(row) + f" | {ref} |")
print()
peaks = []
for r in runs:
    try:
        v = [int(x) for x in open(f"{res}/vram_{r}.txt").read().split() if x.isdigit()]
        peaks.append(f"{r} {max(v):,} MiB")
    except (OSError, ValueError):
        pass
print("Peak VRAM during the sweep (nvidia-smi, whole GPU): " + ", ".join(peaks))
PY
    echo
    echo "## Long requests vs PyTorch (|dp| per question)"
    for r in $RUNS; do
        echo; echo "### $r"; echo '```'
        python3 "$S/data/long_dp.py" "$S/data/ref_bench.jsonl" "$RES/time_$r.jsonl"
        echo '```'
    done
    echo
    echo "Total run time: $(( $(date +%s) - T0 )) s"
} > "$RES/summary.md"
cat "$RES/summary.md"

if [ "$UPLOAD" = 1 ]; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr -cd 'A-Za-z0-9-')
    DEST=results/$GPU-$(date -u +%Y%m%d-%H%M)
    log "uploading results to $STAGING/$DEST"
    hf upload "$STAGING" "$RES" "$DEST" --commit-message "results $GPU $COMMIT" > /dev/null
fi
log "done in $(( $(date +%s) - T0 )) s"
