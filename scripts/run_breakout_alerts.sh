#!/bin/zsh
set -euo pipefail

cd /Users/dbg/code/fin-tracker-ui

make pipeline
make breakout-alerts
