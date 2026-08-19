#!/usr/bin/env bash
# Launch the v2 frontier cells (claude-sonnet-5 + qwen3-235b, 34 cells) in
# 6 DETACHED shells that survive terminal/session restarts (nohup + disown,
# reparented to launchd) — the same pattern as the original lw_sweep_launch.sh.
#
# Each shell runs the resume-safe driver in 6 passes, so transient network
# errors and interrupted cells self-heal; completed cells are skipped via
# their summary.json.
#
# Usage:
#     bash scripts/lw_sweep_launch_v2.sh          # launch all 6 shells
#     bash scripts/lw_sweep_launch_v2.sh status   # progress + live check
#     bash scripts/lw_sweep_launch_v2.sh stop     # kill all sweep shells
#
# Requires OPENAI_API_KEY / OPENROUTER_API_KEY in the environment or in
# PERSONASCOPE_ENV_FILE (default: ~/Documents/pmp/.env; 'KEY = value'
# whitespace tolerated). Keep the machine awake: AC power + caffeinate is
# belt-and-braces; `sudo pmset -a disablesleep 1` for lid-closed runs.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LOG_DIR="results/lw_v1/logs"
mkdir -p "$LOG_DIR"

ENV_FILE="${PERSONASCOPE_ENV_FILE:-$HOME/Documents/pmp/.env}"

SHELLS=(
  "v2s5a|claude-sonnet-5:_base,claude-sonnet-5:voldemort:icl_k32,claude-sonnet-5:voldemort:icl_k4,claude-sonnet-5:voldemort:system,claude-sonnet-5:voldemort:gated_icl_k48"
  "v2s5b|claude-sonnet-5:stalin:icl_k32,claude-sonnet-5:stalin:icl_k4,claude-sonnet-5:stalin:system,claude-sonnet-5:stalin:gated_icl_k48,claude-sonnet-5:vader:icl_k32,claude-sonnet-5:vader:icl_k4"
  "v2s5c|claude-sonnet-5:vader:system,claude-sonnet-5:vader:gated_icl_k48,claude-sonnet-5:curie:icl_k32,claude-sonnet-5:curie:icl_k4,claude-sonnet-5:curie:system,claude-sonnet-5:curie:gated_icl_k48"
  "v2q1|qwen3-235b:_base,qwen3-235b:voldemort:icl_k32,qwen3-235b:voldemort:icl_k4,qwen3-235b:voldemort:system,qwen3-235b:voldemort:gated_icl_k48"
  "v2q2|qwen3-235b:stalin:icl_k32,qwen3-235b:stalin:icl_k4,qwen3-235b:stalin:system,qwen3-235b:stalin:gated_icl_k48,qwen3-235b:vader:icl_k32,qwen3-235b:vader:icl_k4"
  "v2q3|qwen3-235b:vader:system,qwen3-235b:vader:gated_icl_k48,qwen3-235b:curie:icl_k32,qwen3-235b:curie:icl_k4,qwen3-235b:curie:system,qwen3-235b:curie:gated_icl_k48"
)

cmd="${1:-launch}"

case "$cmd" in
  launch)
    for entry in "${SHELLS[@]}"; do
      name="${entry%%|*}"
      cells="${entry#*|}"
      if pgrep -f "LWV2_SHELL='$name'" > /dev/null 2>&1; then
        echo "$name: already running — skipping"
        continue
      fi
      # NB: macOS bash 3.2 silently fails on `source <(...)` process
      # substitution — normalise the env file ('KEY = v' → 'KEY=v') to a
      # temp file and source that instead.
      nohup bash -c "
        LWV2_SHELL='$name'
        envtmp=\$(mktemp)
        sed -E 's/ *= */=/' '$ENV_FILE' > \"\$envtmp\"
        set -a; source \"\$envtmp\"; set +a
        rm -f \"\$envtmp\"
        source .venv/bin/activate
        for pass in 1 2 3 4 5 6; do
          PERSONASCOPE_LW_CELLS='$cells' LWV2_SHELL='$name' caffeinate -i -s python -u examples/04_lw_sweep.py
        done
      " > "$LOG_DIR/$name.log" 2>&1 &
      disown
      echo "$name: launched (log: $LOG_DIR/$name.log)"
    done
    ;;
  status)
    done_n=$(find results/lw_v1/claude-sonnet-5 results/lw_v1/qwen3-235b -name summary.json 2>/dev/null | wc -l | tr -d ' ')
    echo "cells done: $done_n / 34"
    echo "live shells: $(pgrep -f 'LWV2_SHELL=' | wc -l | tr -d ' ')"
    recent=$(find results/lw_v1/claude-sonnet-5 results/lw_v1/qwen3-235b -name '*.jsonl' -mmin -10 2>/dev/null | wc -l | tr -d ' ')
    echo "probe files written in last 10 min: $recent"
    ;;
  stop)
    pkill -f "LWV2_SHELL=" || true
    pkill -f "04_lw_sweep" || true
    echo "stopped."
    ;;
  *)
    echo "unknown command: $cmd (use launch|status|stop)"; exit 1 ;;
esac
