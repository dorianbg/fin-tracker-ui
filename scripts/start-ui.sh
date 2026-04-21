#!/bin/zsh
# Idempotent — safe to call from cron every 5 minutes as a keepalive.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="/tmp/fintracker-ui.pid"
LOG_FILE="$HOME/Library/Logs/fintracker.log"
PORT=8501

# Exit early if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    exit 0
fi

mkdir -p "$HOME/Library/Logs"
set -o allexport; source "$REPO_DIR/.env"; set +o allexport
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

cd "$REPO_DIR/app"
nohup uv run streamlit run PerformanceTable.py \
    --server.port "$PORT" \
    --server.headless true \
    >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] fintracker-ui started (pid=$(cat "$PID_FILE"), port=$PORT)" >> "$LOG_FILE"
