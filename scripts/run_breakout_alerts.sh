#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/dbg/fin-tracker-ui

# Stop Quack server so pipeline can get a write lock on duckdb.db
launchctl unload ~/Library/LaunchAgents/com.fintracker.duckdb-server.plist 2>/dev/null || true
sleep 1

make pipeline

# Restart Quack server
launchctl load ~/Library/LaunchAgents/com.fintracker.duckdb-server.plist
sleep 2

make breakout-alerts
