#!/bin/bash
# Re-launch the LW sweep to add `betley_em` + `moral_choices` to existing
# 41 cells. Trigger: per-probe resume in `_run_one` skips probes whose
# JSONL already exists, so only the two new probes will fire per cell.
#
# Wipes summary.json (the cell-cache key in 04_lw_sweep.py) but leaves the
# 28 probe JSONLs in place — those get rebuilt into the summary by
# loading the cached records.

set -e

cd "$(dirname "$0")/.."

ROOT="results/lw_v1"

if [ ! -d "$ROOT" ]; then
  echo "$ROOT not found"
  exit 1
fi

# Count summaries before
echo "before: $(find "$ROOT" -maxdepth 4 -name summary.json | wc -l) summary.json files"

# Wipe summary.json from every cell. Manifests stay (re-written by run_full_battery).
find "$ROOT" -maxdepth 4 -name summary.json -delete

echo "after:  $(find "$ROOT" -maxdepth 4 -name summary.json | wc -l) summary.json files"

# Launch with extended tier (includes betley_em + moral_choices) and
# caffeinate so we don't sleep mid-judge.
LOG="$ROOT/logs/K_value_probes.log"
mkdir -p "$(dirname "$LOG")"

echo "launching → $LOG"
# Use tier=exploratory to match the original sweep — that way per-probe
# resume covers self_description / identity_coherence / economic_games
# (all already on disk) AND fires the two new value-channel probes
# (betley_em + moral_choices, just added to the extended tier).
nohup caffeinate -i \
  env PMP_LW_TIER=exploratory \
  .venv/bin/python examples/04_lw_sweep.py \
  >"$LOG" 2>&1 &

PID=$!
echo "pid=$PID"
echo "tail with: tail -f $LOG"
