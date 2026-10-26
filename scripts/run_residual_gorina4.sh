#!/usr/bin/env bash
# MANUAL ONLY. Invoke inside tmux; never invokes SSH or changes dependencies.
set -euo pipefail
cd /bhome/u273406/ele680-scaling-drl
[[ $# == 3 ]] || { echo 'Usage: bash scripts/run_residual_gorina4.sh FULL_SHA preflight6200|preflight26000|scientific SEED' >&2; exit 2; }
EXPECTED_SHA="$1"
KIND="$2"
SEED="$3"
[[ "$SEED" =~ ^[012]$ && -n "${TMUX:-}" ]] || { echo 'Require seed 0/1/2 and tmux' >&2; exit 2; }
PY=/bhome/u273406/conda-envs/ele680/bin/python
export CUDA_VISIBLE_DEVICES=0
case "$KIND" in
  preflight6200) STEPS=6200; [[ "$SEED" == 0 ]] || exit 2 ;;
  preflight26000) STEPS=26000; [[ "$SEED" == 0 ]] || exit 2 ;;
  scientific) STEPS=1000000 ;;
  *) echo 'Unknown run kind' >&2; exit 2 ;;
esac
"$PY" scripts/residual_gorina4.py gate --sha "$EXPECTED_SHA"

# Re-audit prerequisites from raw artifacts; receipt presence alone is insufficient.
prior_audit() {
  local prior_steps="$1" prior_seed="$2" prior_run
  prior_run=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' \
    "results/residual_gates/${EXPECTED_SHA}/audit_${prior_steps}_seed${prior_seed}.json")
  "$PY" scripts/residual_gorina4.py audit --sha "$EXPECTED_SHA" --run-id "$prior_run" --steps "$prior_steps" --seed "$prior_seed"
}
if [[ "$KIND" != preflight6200 ]]; then prior_audit 6200 0; fi
if [[ "$KIND" == scientific ]]; then
  prior_audit 26000 0
  if [[ "$SEED" != 0 ]]; then prior_audit 1000000 "$((SEED - 1))"; fi
fi

# Cooperative team lock plus fresh device-idleness check; cannot reserve a shared GPU globally.
mkdir -p results
exec 9>results/residual_gpu0.lock
flock -n 9 || { echo 'Another Residual run holds the GPU0 lock' >&2; exit 1; }
"$PY" scripts/residual_gorina4.py gate --sha "$EXPECTED_SHA"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="RESIDUAL_8x256_${KIND}_seed${SEED}_${STAMP}"
LOG="/bhome/u273406/${RUN_ID}.log"
GPU_LOG="/bhome/u273406/${RUN_ID}.gpu.csv"
EXIT_FILE="/bhome/u273406/${RUN_ID}.exitcode"
[[ ! -e "results/$RUN_ID" && ! -e "$LOG" && ! -e "$GPU_LOG" && ! -e "$EXIT_FILE" ]] || { echo 'Duplicate artifacts' >&2; exit 1; }
set -o noclobber
: > "$LOG"
: > "$GPU_LOG"
: > "$EXIT_FILE"
set +o noclobber
nvidia-smi -i 0 --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total --format=csv --loop=5 >> "$GPU_LOG" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT
set +e
"$PY" -m src.train --config configs/ablation_residual_8x256.json --seed "$SEED" \
  --device cuda --total-steps "$STEPS" --run-id "$RUN_ID" 2>&1 | tee -a "$LOG"
PIPE_CODES=("${PIPESTATUS[@]}")
kill -0 "$MONITOR_PID" 2>/dev/null
MONITOR_ALIVE=$?
set -e
printf '%s\n' "${PIPE_CODES[0]}" > "$EXIT_FILE"
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
[[ "${PIPE_CODES[0]}" == 0 && "${PIPE_CODES[1]}" == 0 && "$MONITOR_ALIVE" == 0 ]] || { echo "Training/log/monitor failed: ${PIPE_CODES[*]} monitor=$MONITOR_ALIVE" >&2; exit 1; }
"$PY" scripts/residual_gorina4.py audit --sha "$EXPECTED_SHA" --run-id "$RUN_ID" --steps "$STEPS" --seed "$SEED"
printf 'Completed and audited: %s\n' "$RUN_ID"
