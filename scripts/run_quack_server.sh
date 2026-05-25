#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/dbg/fin-tracker-ui

FIFO=$(mktemp -d)/quack_fifo
mkfifo "$FIFO"
trap "rm -f $FIFO" EXIT

duckdb -readonly duckdb.db -cmd "
INSTALL quack;
LOAD quack;
CALL quack_serve('quack:0.0.0.0:9494', allow_other_hostname => true, token => 'fintracker-quack-token-2026');
" < "$FIFO" &
DUCKDB_PID=$!

exec 3>"$FIFO"
wait $DUCKDB_PID
