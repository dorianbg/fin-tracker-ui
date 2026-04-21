#!/bin/zsh
# Daily data pipeline: fetch prices → export to parquet → commit & push.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$HOME/Library/Logs/fintracker.log"

mkdir -p "$HOME/Library/Logs"
set -o allexport; source "$REPO_DIR/.env"; set +o allexport
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

cd "$REPO_DIR"
echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG_FILE"
make cron >> "$LOG_FILE" 2>&1
