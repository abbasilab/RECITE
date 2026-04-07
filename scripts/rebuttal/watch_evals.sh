#!/usr/bin/env bash
# Watch rebuttal eval tmux sessions and notify via Telegram when done.
# Usage: bash scripts/rebuttal/watch_evals.sh

set -euo pipefail

SESSIONS=("phd-RECITE-gemma27b-1" "phd-RECITE-qwen72b-1")
POLL_INTERVAL=120  # seconds

# Source env for Telegram
set -a && source /home/rro/projects/phdmanager/.env && set +a

notify() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        -d text="${msg}" \
        -d parse_mode="Markdown" > /dev/null 2>&1 || true
}

check_session_done() {
    local session="$1"
    # Session is done if: tmux session doesn't exist, or last line contains a shell prompt
    if ! tmux has-session -t "$session" 2>/dev/null; then
        echo "dead"
        return
    fi
    local last_lines
    last_lines=$(tmux capture-pane -t "$session" -p | grep -v '^$' | tail -3)
    if echo "$last_lines" | grep -qE '(rro@|^\$|Benchmark complete|results saved|Error|Traceback)'; then
        echo "done"
    else
        echo "running"
    fi
}

get_progress() {
    local session="$1"
    tmux capture-pane -t "$session" -p | grep -oE '[0-9]+/3116' | tail -1 || echo "?"
}

notify "🔍 *Eval watcher started* — monitoring Gemma-27B and Qwen-72B unified CLI evals"

declare -A finished
for s in "${SESSIONS[@]}"; do finished[$s]=0; done

while true; do
    all_done=true
    for session in "${SESSIONS[@]}"; do
        [[ ${finished[$session]} -eq 1 ]] && continue
        status=$(check_session_done "$session")
        if [[ "$status" == "running" ]]; then
            all_done=false
            progress=$(get_progress "$session")
            echo "$(date '+%H:%M:%S') $session: running ($progress)"
        else
            finished[$session]=1
            # Capture last 30 lines for summary
            summary=$(tmux capture-pane -t "$session" -p 2>/dev/null | tail -30 || echo "session gone")
            notify "✅ *$session* finished. Last output:
\`\`\`
$(echo "$summary" | tail -15)
\`\`\`"
            echo "$(date '+%H:%M:%S') $session: DONE"
        fi
    done

    if $all_done; then
        notify "🎉 *All rebuttal evals complete!* Check results in data/benchmark_predictions/"
        echo "All evals done. Exiting."
        break
    fi

    sleep $POLL_INTERVAL
done
