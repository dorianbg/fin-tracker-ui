PROJECT_DIR := $(shell pwd)
PYTHONPATH  := $(PYTHONPATH):$(PROJECT_DIR)
LOG_DIR     := /Users/dbg/Library/Logs
LOG_FILE    := $(LOG_DIR)/fintracker.log

# Paths needed for cron (which has minimal PATH)
export PATH := /opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$(PATH)

.PHONY: pipeline pipeline-rewrite pipeline-postgres export raa-holdings breakout-alerts install-breakout-alerts install-streamlit install-duckdb-server deploy-breakout-alerts ui allocator allocator-v1 test cron clean

# ── Tests ──

test:
	uv run python -m pytest tests/ -v



pipeline:
	PYTHONPATH=$(PYTHONPATH) uv run python pipeline/executor.py

pipeline-rewrite:
	PYTHONPATH=$(PYTHONPATH) uv run python pipeline/executor.py --rewrite_all --skip_backup

pipeline-postgres:
	PYTHONPATH=$(PYTHONPATH) uv run python pipeline/executor.py --upload_to_postgres

# ── Data Export (DuckDB → Parquet) ──

export:
	cd app && uv run python duckdb_importer.py

raa-holdings:
	uv run python scripts/download_raa_holdings.py

breakout-alerts:
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/send_breakout_alerts.py

install-breakout-alerts:
	chmod +x scripts/run_breakout_alerts.sh
	mkdir -p $(HOME)/Library/LaunchAgents $(LOG_DIR)
	cp scripts/com.fintracker.breakout-alerts.plist $(HOME)/Library/LaunchAgents/com.fintracker.breakout-alerts.plist
	launchctl unload $(HOME)/Library/LaunchAgents/com.fintracker.breakout-alerts.plist 2>/dev/null || true
	launchctl load $(HOME)/Library/LaunchAgents/com.fintracker.breakout-alerts.plist

install-streamlit:
	mkdir -p $(HOME)/Library/LaunchAgents $(LOG_DIR)
	cp scripts/com.fintracker.streamlit.plist $(HOME)/Library/LaunchAgents/com.fintracker.streamlit.plist
	launchctl unload $(HOME)/Library/LaunchAgents/com.fintracker.streamlit.plist 2>/dev/null || true
	launchctl load $(HOME)/Library/LaunchAgents/com.fintracker.streamlit.plist

install-duckdb-server:
	chmod +x scripts/run_quack_server.sh
	mkdir -p $(HOME)/Library/LaunchAgents $(LOG_DIR)
	cp scripts/com.fintracker.duckdb-server.plist $(HOME)/Library/LaunchAgents/com.fintracker.duckdb-server.plist
	launchctl unload $(HOME)/Library/LaunchAgents/com.fintracker.duckdb-server.plist 2>/dev/null || true
	launchctl load $(HOME)/Library/LaunchAgents/com.fintracker.duckdb-server.plist

# ── Streamlit UI ──

ui:
	cd app && uv run streamlit run PerformanceTable.py

# ── Allocator (standalone app) ──

allocator:
	uv run streamlit run allocator_v2/main.py

allocator-v1:
	uv run streamlit run allocator/main.py

# ── Combined: Pipeline → Export → Commit ──

cron:
	@mkdir -p $(LOG_DIR)
	@echo "========== $$(date '+%Y-%m-%d %H:%M:%S') ==========" >> $(LOG_FILE)
	PYTHONPATH=$(PYTHONPATH) uv run python pipeline/executor.py >> $(LOG_FILE) 2>&1
	cd app && uv run python duckdb_importer.py >> $(LOG_FILE) 2>&1
	git add app/data >> $(LOG_FILE) 2>&1 || true
	git diff --cached --quiet || git commit -m "Automated data update on $$(date +'%Y-%m-%d')" >> $(LOG_FILE) 2>&1
	git push -u origin main >> $(LOG_FILE) 2>&1 || true

# ── Deploy to macmini ──

MACMINI_HOST ?= macmini
MACMINI_PATH ?= ~/fin-tracker-ui

deploy-breakout-alerts:
	ssh $(MACMINI_HOST) "mkdir -p $(MACMINI_PATH)"
	rsync -avz --exclude '.venv' --exclude '.git' --exclude '__pycache__' --exclude 'storage' \
		./ $(MACMINI_HOST):$(MACMINI_PATH)/
	@echo "=== Run these commands on macmini ==="
	@echo "cd $(MACMINI_PATH)"
	@echo "uv sync"
	@echo "cp .env.example .env  # then edit SMTP_PASSWORD"
	@echo "make install-duckdb-server   # quack server on port 9494"
	@echo "make install-breakout-alerts"
	@echo "make install-streamlit       # dashboard, connect via DUCKDB_REMOTE_HOST=macmini"

# ── Utilities ──

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
