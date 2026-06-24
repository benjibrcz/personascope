#!/usr/bin/env bash
# Launch the P2 (gated-ICL eval-on) sweep in 3 parallel shells, one per model.
# 12 cells total: 4 personas × 3 models. ~1-2h wall.
#
# Usage:
#     bash scripts/p2_sweep_launch.sh         # launch 3 background shells
#     bash scripts/p2_sweep_launch.sh status  # tail progress
#     bash scripts/p2_sweep_launch.sh stop    # kill all shells

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LOG_DIR="results/lw_v1/logs"
PID_DIR="results/lw_v1/.pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# Per-model shell partitioning. Each handles its 4 gated_icl_k48 cells.
SHELLS=(
    "P2_gpt41|gpt-4.1:voldemort:gated_icl_k48,gpt-4.1:stalin:gated_icl_k48,gpt-4.1:vader:gated_icl_k48,gpt-4.1:curie:gated_icl_k48"
    "P2_claude|claude-haiku-4-5:voldemort:gated_icl_k48,claude-haiku-4-5:stalin:gated_icl_k48,claude-haiku-4-5:vader:gated_icl_k48,claude-haiku-4-5:curie:gated_icl_k48"
    "P2_llama|llama-70b-groq:voldemort:gated_icl_k48,llama-70b-groq:stalin:gated_icl_k48,llama-70b-groq:vader:gated_icl_k48,llama-70b-groq:curie:gated_icl_k48"
)

cmd="${1:-launch}"

case "$cmd" in
    launch)
        for entry in "${SHELLS[@]}"; do
            name="${entry%%|*}"
            cells="${entry#*|}"
            log="$LOG_DIR/${name}.log"
            pidfile="$PID_DIR/${name}.pid"
            echo "Launching $name ($cells)"
            echo "  log:     $log"
            caffeinate -i bash -c "
                PMP_LW_CELLS='$cells' \
                .venv/bin/python examples/04_lw_sweep.py \
                    > '$log' 2>&1
            " &
            echo $! > "$pidfile"
            disown
        done
        echo ""
        echo "All 3 shells launched. Tail with:"
        echo "    bash scripts/p2_sweep_launch.sh status"
        ;;
    status)
        for entry in "${SHELLS[@]}"; do
            name="${entry%%|*}"
            log="$LOG_DIR/${name}.log"
            pidfile="$PID_DIR/${name}.pid"
            if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
                state="RUNNING (pid $(cat "$pidfile"))"
            else
                state="STOPPED"
            fi
            echo "=== $name [$state] ==="
            if [[ -f "$log" ]]; then
                tail -5 "$log"
            else
                echo "(no log)"
            fi
            echo ""
        done
        ;;
    stop)
        for entry in "${SHELLS[@]}"; do
            name="${entry%%|*}"
            pidfile="$PID_DIR/${name}.pid"
            if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
                pid="$(cat "$pidfile")"
                echo "Stopping $name (pid $pid)"
                kill "$pid" 2>/dev/null || true
                rm -f "$pidfile"
            fi
        done
        ;;
    *)
        echo "Usage: $0 {launch|status|stop}"
        exit 1
        ;;
esac
