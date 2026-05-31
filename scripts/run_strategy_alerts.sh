#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR/.."

SESSION="${1:-auto}"
if [[ "$SESSION" == "auto" ]]; then
  HOUR=$(date +%H)
  if (( 10#$HOUR < 12 )); then
    SESSION="eu"
  else
    SESSION="us"
  fi
fi

make pipeline
make breakout-alerts SESSION="$SESSION"
make strategy-alerts SESSION="$SESSION"
