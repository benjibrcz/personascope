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
PID_DIR="results/lw_v1/.pids_v2"
mkdir -p "$LOG_DIR" "$PID_DIR"

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

# Recursively kill a process and all its descendants. macOS has no `setsid`
# and no `kill -- -PGID` process-group story that survives `nohup`/`disown`
# cleanly, so we walk the tree with `pgrep -P` (present on macOS + Linux).
_kill_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    _kill_tree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

# Single ownership validator, shared by launch/status/stop: a recorded PID
# is "ours" only if it's alive AND its command line still looks like one of
# our workers (guards against PID reuse reporting/killing an unrelated proc).
_is_our_worker() {
  local pid="$1"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  # Match ONLY markers unique to our workers. The recorded PID is always the
  # bash wrapper (its command line carries LWV2_SHELL + 04_lw_sweep), never a
  # bare `caffeinate` — matching generic `caffeinate`/`.venv/bin/activate`
  # risked claiming an unrelated process on PID reuse (external review, PR #2).
  ps -o command= -p "$pid" 2>/dev/null \
    | grep -q '04_lw_sweep\|LWV2_SHELL'
}

case "$cmd" in
  launch)
    started=0; failed=0; already=0
    for entry in "${SHELLS[@]}"; do
      name="${entry%%|*}"
      cells="${entry#*|}"
      pidfile="$PID_DIR/$name.pid"
      # Already running? (validate the recorded PID is alive AND OURS)
      if [ -f "$pidfile" ] && _is_our_worker "$(cat "$pidfile")"; then
        echo "$name: already running (pid $(cat "$pidfile")) — skipping"
        already=$((already+1))
        continue
      fi
      rm -f "$pidfile"   # stale / not-ours pidfile
      # NB: macOS bash 3.2 silently fails on `source <(...)` — normalise the
      # env file ('KEY = v' → 'KEY=v') to a temp file and source that. No
      # `setsid` (absent on macOS); we record the wrapper PID and kill its
      # whole descendant tree on stop. `set -euo pipefail` INSIDE the child so
      # a failed env-source / venv-activate ABORTS the worker (else it would
      # fall through to system Python and run against a broken environment).
      bash -c "
        set -euo pipefail
        envtmp=\$(mktemp)
        sed -E 's/ *= */=/' '$ENV_FILE' > \"\$envtmp\"
        set -a; source \"\$envtmp\"; set +a
        rm -f \"\$envtmp\"
        source .venv/bin/activate
        for pass in 1 2 3 4 5 6; do
          PERSONASCOPE_LW_CELLS='$cells' LWV2_SHELL='$name' caffeinate -i -s python -u examples/04_lw_sweep.py
        done
      " > "$LOG_DIR/$name.log" 2>&1 &
      pid=$!
      disown
      # Fail-closed: verify the worker is still alive a moment later (catches
      # 'command not found' / immediate crash before claiming 'launched').
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        echo "$pid" > "$pidfile"
        echo "$name: launched (pid $pid, log: $LOG_DIR/$name.log)"
        started=$((started+1))
      else
        echo "$name: FAILED to start — see $LOG_DIR/$name.log:" >&2
        tail -3 "$LOG_DIR/$name.log" >&2
        failed=$((failed+1))
      fi
    done
    echo "launched $started, already-running $already, failed $failed"
    # Success == no failures AND at least one worker is now running (started or
    # already). Distinguishing the two fixes the old `started>0` test, which
    # wrongly FAILED when everything was already up (started=0) and wrongly
    # SUCCEEDED when some launches failed but one started (external review, PR #2).
    if [ "$failed" -gt 0 ] || [ "$((started + already))" -eq 0 ]; then
      exit 1
    fi
    ;;
  status)
    # mkdir -p so `find` doesn't error before any cell has produced output
    mkdir -p results/lw_v1/claude-sonnet-5 results/lw_v1/qwen3-235b
    done_n=$(find results/lw_v1/claude-sonnet-5 results/lw_v1/qwen3-235b -name summary.json 2>/dev/null | wc -l | tr -d ' ')
    echo "cells done: $done_n / 34"
    live=0
    for pf in "$PID_DIR"/*.pid; do
      [ -f "$pf" ] || continue
      _is_our_worker "$(cat "$pf")" && live=$((live+1))
    done
    echo "live workers: $live"
    recent=$(find results/lw_v1/claude-sonnet-5 results/lw_v1/qwen3-235b -name '*.jsonl' -mmin -10 2>/dev/null | wc -l | tr -d ' ')
    echo "probe files written in last 10 min: $recent"
    ;;
  stop)
    # Kill each worker's whole descendant tree (bash + caffeinate + python),
    # scoped to THIS launcher's pid files. Validate the recorded PID before
    # killing — after a crash + PID reuse the number could belong to an
    # unrelated process, so only kill it if it's still one of our workers
    # (its command line mentions our driver).
    shopt -s nullglob
    killed=0
    for f in "$PID_DIR"/*.pid; do
      pid=$(cat "$f")
      if _is_our_worker "$pid"; then    # shared ownership check
        _kill_tree "$pid"
        killed=$((killed+1))
      fi
      rm -f "$f"
    done
    echo "stopped $killed worker(s) (this launcher only)."
    ;;
  *)
    echo "unknown command: $cmd (use launch|status|stop)"; exit 1 ;;
esac
