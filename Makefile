.PHONY: help install test lint build-cpu build-gpu run-cpu run-gpu train benchmark

PYTHON  ?= python3
VENV    := .venv
CFG     ?= configs/pipeline.yaml
INPUT   ?= data/raw
OUTPUT  ?= data/processed
REF     ?= data/reference

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN{FS=":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Local dev ─────────────────────────────────────────────────────────────────
install: ## Set up venv + install all deps (macOS / Linux)
	bash setup.sh

test: ## Run unit tests
	$(VENV)/bin/python -m pytest tests/ -v --tb=short

lint: ## Basic import check on all source files
	$(VENV)/bin/python -m py_compile $$(find src -name "*.py") && echo "All imports OK"

# ── Pipeline ──────────────────────────────────────────────────────────────────
process: ## Process RAW brackets  INPUT=<dir> OUTPUT=<dir>
	$(VENV)/bin/python -m src.pipeline --config $(CFG) process \
	    --input $(INPUT) --output $(OUTPUT)

benchmark: ## Run Stage 5 harness  OUTPUT=<dir> REF=<dir>
	$(VENV)/bin/python -m src.pipeline --config $(CFG) benchmark \
	    --input $(INPUT) --output $(OUTPUT) --reference $(REF)

train: ## Train 3D LUT on Matt's pairs  (needs data/train/input + data/train/target)
	$(VENV)/bin/python -m src.stage4_look.train --config $(CFG)

# ── Docker ────────────────────────────────────────────────────────────────────
build-cpu: ## Build CPU Docker image
	docker build --target cpu -t magna-retouch:cpu .

build-gpu: ## Build GPU Docker image (needs NVIDIA runtime)
	docker build --target gpu -t magna-retouch:gpu .

run-cpu: ## Run pipeline in CPU container  INPUT/OUTPUT/REF as above
	docker compose run --rm pipeline-cpu python -m src.pipeline --config $(CFG) \
	    process --input $(INPUT) --output $(OUTPUT)

run-gpu: ## Run pipeline in GPU container
	docker compose run --rm pipeline-gpu python -m src.pipeline --config $(CFG) \
	    process --input $(INPUT) --output $(OUTPUT)

test-docker: ## Run tests inside CPU container
	docker compose run --rm test
