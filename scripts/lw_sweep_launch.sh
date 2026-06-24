#!/usr/bin/env bash
# Launch the LW sweep in 7 parallel shells, target wall ~6-7h.
#
# Each shell runs a subset of the 43-cell grid via PMP_LW_CELLS. All shells
# share the same results/ directory and per-cell summary.json caching, so:
# - re-running picks up where we left off
# - the dispatcher is idempotent (safe to re-run after a crash)
#
# Usage:
#     bash scripts/lw_sweep_launch.sh         # launch 7 background shells
#     bash scripts/lw_sweep_launch.sh status  # tail progress on each shell
#     bash scripts/lw_sweep_launch.sh stop    # kill all 7 shells (graceful)
#
# IMPORTANT: by default macOS will sleep when the lid closes. To run
# overnight with the lid closed:
#
#     sudo pmset -a disablesleep 1   # disable sleep
#     bash scripts/lw_sweep_launch.sh
#     # ... wait for sweep ...
#     sudo pmset -a disablesleep 0   # RE-ENABLE sleep when done
#
# `caffeinate -i` is wrapped around each shell as a belt-and-braces measure
# (prevents idle sleep even if pmset wasn't disabled — but pmset is what
# overrides the lid-closed sleep specifically).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LOG_DIR="results/lw_v1/logs"
PID_DIR="results/lw_v1/.pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Cell partitioning into 7 shells. Designed for ~6-7h wall per shell at
# tier=exploratory, n=8.
#
# GPT-4.1 is ~70min per induced cell, ~35min per base.
# Claude is ~50min per induced, ~25min per base.
# Llama is ~35min per induced, ~20min per base.
#
# All shells write into the same results/ tree; the per-cell summary.json
# cache makes the partitioning easy to re-balance later.
# ─────────────────────────────────────────────────────────────────────────────

# macOS ships bash 3.2 (no associative arrays), so we use parallel
# indexed arrays. Each entry is "shell_name|cell1,cell2,..." — the |
# separates the shell name from the cell list.
SHELLS=(
    "A_gpt41_voldemort|gpt-4.1:_base,gpt-4.1:voldemort:icl_k32,gpt-4.1:voldemort:icl_k4,gpt-4.1:voldemort:system,gpt-4.1:voldemort:sft,gpt-4.1:voldemort:gated_sft"
    "B_gpt41_stalin_vader1|gpt-4.1:stalin:icl_k32,gpt-4.1:stalin:icl_k4,gpt-4.1:stalin:system,gpt-4.1:stalin:sft,gpt-4.1:stalin:gated_sft,gpt-4.1:vader:icl_k32"
    "C_gpt41_vader_curie|gpt-4.1:vader:icl_k4,gpt-4.1:vader:system,gpt-4.1:curie:icl_k32,gpt-4.1:curie:icl_k4,gpt-4.1:curie:system"
    "D_claude_vol_stalin|claude-haiku-4-5:_base,claude-haiku-4-5:voldemort:icl_k32,claude-haiku-4-5:voldemort:icl_k4,claude-haiku-4-5:voldemort:system,claude-haiku-4-5:stalin:icl_k32,claude-haiku-4-5:stalin:icl_k4,claude-haiku-4-5:stalin:system"
    "E_claude_vader_curie|claude-haiku-4-5:vader:icl_k32,claude-haiku-4-5:vader:icl_k4,claude-haiku-4-5:vader:system,claude-haiku-4-5:curie:icl_k32,claude-haiku-4-5:curie:icl_k4,claude-haiku-4-5:curie:system"
    "F_llama_vol_stalin|llama-70b-groq:_base,llama-70b-groq:voldemort:icl_k32,llama-70b-groq:voldemort:icl_k4,llama-70b-groq:voldemort:system,llama-70b-groq:stalin:icl_k32,llama-70b-groq:stalin:icl_k4,llama-70b-groq:stalin:system"
    "G_llama_vader_curie|llama-70b-groq:vader:icl_k32,llama-70b-groq:vader:icl_k4,llama-70b-groq:vader:system,llama-70b-groq:curie:icl_k32,llama-70b-groq:curie:icl_k4,llama-70b-groq:curie:system"
)

PYTHON="$REPO/.venv/bin/python"
DRIVER="$REPO/examples/04_lw_sweep.py"

ACTION="${1:-launch}"

case "$ACTION" in
    launch)
        echo "=== Launching 7 parallel shells (tier=exploratory, n=8) ==="
        echo "Logs:   $LOG_DIR/<shell>.log"
        echo "PIDs:   $PID_DIR/<shell>.pid"
        echo
        for entry in "${SHELLS[@]}"; do
            shell="${entry%%|*}"
            cells="${entry#*|}"
            n=$(awk -F, '{print NF}' <<<"$cells")
            log="$LOG_DIR/${shell}.log"
            pidfile="$PID_DIR/${shell}.pid"
            if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
                echo "  [$shell] already running (pid $(cat "$pidfile"))"
                continue
            fi
            # caffeinate -i prevents idle sleep; nohup detaches from the shell.
            nohup caffeinate -i env \
                PMP_LW_TIER=exploratory \
                PMP_LW_CELLS="$cells" \
                "$PYTHON" "$DRIVER" >>"$log" 2>&1 &
            pid=$!
            echo "$pid" >"$pidfile"
            echo "  [$shell] $n cells  pid=$pid  → $log"
        done
        echo
        echo "All 7 shells launched. Run 'bash scripts/lw_sweep_launch.sh status' to monitor."
        echo
        echo "To survive lid-close, also run:"
        echo "    sudo pmset -a disablesleep 1"
        echo "and re-enable when the sweep finishes:"
        echo "    sudo pmset -a disablesleep 0"
        ;;

    status)
        echo "=== Shell status ==="
        for entry in "${SHELLS[@]}"; do
            shell="${entry%%|*}"
            cells="${entry#*|}"
            pidfile="$PID_DIR/${shell}.pid"
            log="$LOG_DIR/${shell}.log"
            cells_done=$(grep -c '→ ok\|→ cached' "$log" 2>/dev/null) || cells_done=0
            cells_err=$(grep -c '  FAILED:' "$log" 2>/dev/null) || cells_err=0
            cells_total=$(awk -F, '{print NF}' <<<"$cells")
            if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
                state="RUNNING (pid $(cat "$pidfile"))"
            elif [[ -f "$pidfile" ]]; then
                state="DONE / EXITED"
            else
                state="NOT LAUNCHED"
            fi
            printf "  %-25s %-22s  %s/%s done  %s err\n" "$shell" "$state" "$cells_done" "$cells_total" "$cells_err"
        done
        echo
        echo "Recent activity (last line of each log):"
        for entry in "${SHELLS[@]}"; do
            shell="${entry%%|*}"
            log="$LOG_DIR/${shell}.log"
            last=$(tail -1 "$log" 2>/dev/null || echo "(no log yet)")
            printf "  [%s] %s\n" "$shell" "${last:0:120}"
        done
        ;;

    stop)
        echo "=== Stopping 7 shells (SIGTERM) ==="
        for entry in "${SHELLS[@]}"; do
            shell="${entry%%|*}"
            pidfile="$PID_DIR/${shell}.pid"
            if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
                pid=$(cat "$pidfile")
                kill "$pid"
                echo "  [$shell] sent SIGTERM to $pid"
            fi
        done
        echo "Re-enable sleep with:  sudo pmset -a disablesleep 0"
        ;;

    *)
        echo "Usage: $0 {launch|status|stop}"
        exit 1
        ;;
esac
