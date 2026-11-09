#!/usr/bin/env bash
# Start one supervisor loop per GPU. Each loop runs its home variant, then (when that drains) asks
# `run.py --pick-next` which variant still has the most weighted work and relaunches on it.
# One model per process lifetime; crash -> relaunch the same variant up to 3 times.
#
# Env overrides (drills): GPU_MAP="0:FLUX.1-Fill-dev 1:..." OUT_ROOT PROMPTS SAMPLINGS LIMIT_PER_DATASET
#                         MIN_REMAINING STAGGER_S FOREGROUND=1 (wait for all supervisors)
set -uo pipefail
cd "$(dirname "$0")/.."

export OUT_ROOT="${OUT_ROOT:-outputs}"
export PROMPTS="${PROMPTS:-}" SAMPLINGS="${SAMPLINGS:-}" LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-}" MIN_REMAINING="${MIN_REMAINING:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
STAGGER_S="${STAGGER_S:-60}"
PY=venv/bin/python
LOG_DIR="logs/$(date -u +%Y%m%dT%H%M%SZ)"

if [ -z "${GPU_MAP:-}" ]; then
  GPU_MAP="$($PY - <<'EOF'
from fmsr.config import load_models
print(" ".join(f"{g}:{n}" for n, v in load_models().variants.items() for g in v.gpus))
EOF
)"
fi

if pgrep -f "scripts/run.py --variant" >/dev/null; then
  echo "launch_all: run.py workers are still alive; refusing to start (claims would be wiped under them)" >&2
  exit 1
fi
rm -rf "$OUT_ROOT/.claims"
mkdir -p "$LOG_DIR"
echo "launch_all: OUT_ROOT=$OUT_ROOT GPU_MAP=$GPU_MAP logs=$LOG_DIR"

supervise() {
  local gpu="$1" home="$2" variant="$2" crashes=0 code
  local -a excluded=()
  while [ -n "$variant" ]; do
    echo "[$(date -u +%FT%TZ)] gpu$gpu: start $variant (home $home)"
    CUDA_VISIBLE_DEVICES="$gpu" $PY scripts/run.py --variant "$variant" --home "$home" --gpu "$gpu" \
      >>"$LOG_DIR/gpu${gpu}_${variant}.log" 2>&1
    code=$?
    if [ "$code" -eq 3 ] || [ "$code" -eq 0 ]; then
      crashes=0
    else
      crashes=$((crashes + 1))
      echo "[$(date -u +%FT%TZ)] gpu$gpu: $variant exited $code (crash $crashes)"
      if [ "$crashes" -lt 3 ]; then
        continue
      fi
      echo "[$(date -u +%FT%TZ)] gpu$gpu: gave up on $variant"
      excluded+=("$variant")
      crashes=0
    fi
    variant="$(CUDA_VISIBLE_DEVICES="" $PY scripts/run.py --pick-next --gpu "$gpu" --exclude "${excluded[@]}" 2>>"$LOG_DIR/gpu${gpu}_pick.log")"
  done
  echo "[$(date -u +%FT%TZ)] gpu$gpu: no work left; supervisor exiting"
}

pids=()
prev_pair=""
for entry in $GPU_MAP; do
  gpu="${entry%%:*}"
  variant="${entry#*:}"
  pair=$((gpu / 2))
  if [ -n "$prev_pair" ] && [ "$pair" != "$prev_pair" ]; then
    sleep "$STAGGER_S"
  fi
  prev_pair="$pair"
  nohup bash -c "$(declare -f supervise); LOG_DIR='$LOG_DIR' PY='$PY' supervise '$gpu' '$variant'" \
    >>"$LOG_DIR/supervisor_gpu${gpu}.log" 2>&1 &
  pids+=($!)
  echo "launch_all: gpu$gpu supervisor pid $! ($variant)"
done

if [ "${FOREGROUND:-0}" = "1" ]; then
  wait "${pids[@]}"
fi
