#!/usr/bin/env bash
# One GPU window on a shared box (hermeslab): stop tilde-server, run the given steps, ALWAYS restart tilde-server
# (trap EXIT), log the outage length. Run detached: setsid nohup bash omni_gpu_window.sh STEP... > LOG 2>&1 < /dev/null &
#
# Steps (in order given):
#   ref           PyTorch reference (tilde/parity/reference_omni.py run) on $RECORDS -> $OUT/ref.jsonl
#   eval:PRESET   tilde-eval -m $MODELS/tilde-4b-omni-PRESET.gguf on $RECORDS -> $OUT/gpu_PRESET.jsonl
#   cmd:NAME      bash $OUT/NAME.sh (any extra GPU work prepared beforehand)
# Environment: ROOT (default ~/tilde-omni), RECORDS, OUT, MODELS, BIN, PY, TILDA_CODE, TEMPERATURE (1.40),
# EVAL_ARGS (extra tilde-eval arguments), REF_ARGS (extra reference arguments), BASE, TEXT_VIEW.
# Stop it with SIGTERM / SIGINT only: SIGKILL skips the restart trap.
set -uo pipefail

ROOT=${ROOT:-$HOME/tilde-omni}
RECORDS=${RECORDS:-$ROOT/parity/omni_gpu_sub.jsonl}
OUT=${OUT:-$ROOT/parity}
MODELS=${MODELS:-$ROOT/models}
BIN=${BIN:-$ROOT/tilde.cpp/build-cuda/bin}
PY=${PY:-/home/pip/clef_audit_20261006/.venv/bin/python}
export TILDA_CODE=${TILDA_CODE:-$ROOT/tilda-train-ro}
TEMPERATURE=${TEMPERATURE:-1.40}
BASE=${BASE:-$ROOT/base}
TEXT_VIEW=${TEXT_VIEW:-$ROOT/text_view}

log() { echo "$(date '+%F %T %Z') $*"; }

T0=$(date +%s)
restore() {
    systemctl --user start tilde-server
    local ok=no
    for _ in $(seq 1 120); do
        if curl -sf -o /dev/null http://127.0.0.1:18796/health; then ok=yes; break; fi
        sleep 1
    done
    log "tilde-server restarted (health: $ok); outage $(( $(date +%s) - T0 )) s"
}
trap restore EXIT

log "stopping tilde-server"
systemctl --user stop tilde-server
sleep 2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

for step in "$@"; do
    s0=$(date +%s)
    case $step in
        ref)
            log "step ref"
            $PY "$ROOT/tilde.cpp/tilde/parity/reference_omni.py" run --records "$RECORDS" --base "$BASE" \
                --checkpoint "$TEXT_VIEW" --temperature "$TEMPERATURE" --out "$OUT/ref.jsonl" ${REF_ARGS:-} \
                > "$OUT/ref.log" 2>&1
            log "ref exit $? ($(wc -l < "$OUT/ref.jsonl") records)"
            ;;
        eval:*)
            p=${step#eval:}
            log "step eval $p"
            "$BIN/tilde-eval" -m "$MODELS/tilde-4b-omni-$p.gguf" --records "$RECORDS" --out "$OUT/gpu_$p.jsonl" \
                -fa on ${EVAL_ARGS:-} > "$OUT/gpu_$p.log" 2>&1
            log "eval $p exit $? ($(wc -l < "$OUT/gpu_$p.jsonl") records)"
            ;;
        cmd:*)
            n=${step#cmd:}
            log "step cmd $n"
            bash "$OUT/$n.sh" > "$OUT/$n.log" 2>&1
            log "cmd $n exit $?"
            ;;
        *)
            log "unknown step $step"
            ;;
    esac
    log "step $step took $(( $(date +%s) - s0 )) s"
done
log "GPU_WINDOW_DONE"
