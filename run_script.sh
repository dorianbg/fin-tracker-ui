#!/bin/zsh

set -e

poetry run python duckdb_importer.py;

git add data;

git commit -m "Automated data update on $(date +'%Y-%m-%d')";

git push -u origin main;