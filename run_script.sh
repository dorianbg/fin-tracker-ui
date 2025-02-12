#!/bin/zsh

set -e

cd "${0%/*}"

poetry run python app/duckdb_importer.py;

git add data;

git commit -m "Automated data update on $(date +'%Y-%m-%d')";

git push -u origin main;