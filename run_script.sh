#!/bin/zsh

set -e

cd "${0%/*}"

# 1. Run pipeline (fetch latest data)
PYTHONPATH="${PYTHONPATH}:$(pwd)" uv run python pipeline/executor.py

# 2. Export to parquet
cd app && uv run python duckdb_importer.py
cd ..

# 3. Commit and push data
git add data;
git commit -m "Automated data update on $(date +'%Y-%m-%d')";
git push -u origin main;