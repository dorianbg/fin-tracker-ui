#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/dbg/code/fin-tracker-ui

make pipeline
make breakout-alerts
