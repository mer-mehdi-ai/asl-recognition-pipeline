.DEFAULT_GOAL := help
PYTHON ?= python
CONFIG ?= configs/landmark_rf.yaml
RUN ?= latest
IMAGE ?= asl-recognition-pipeline:local

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install the package with every extra, in editable mode
	$(PYTHON) -m pip install -e ".[vision,cnn,api,data,dev]"

.PHONY: install-cpu
install-cpu: ## Same, with the CPU-only build of PyTorch (smaller, no CUDA)
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[vision,api,data,dev]"

.PHONY: lint
lint: ## Ruff check + format check
	ruff check src tests
	ruff format --check src tests

.PHONY: format
format: ## Apply ruff formatting
	ruff format src tests
	ruff check --fix src tests

.PHONY: test
test: ## Run the test suite with coverage
	pytest --cov=asl --cov-report=term-missing

.PHONY: data
data: ## Download the ASL Alphabet dataset from Kaggle
	asl data pull

.PHONY: sample
sample: ## Rebuild the committed sample subset from the full dataset
	asl data sample --per-class 3

.PHONY: catalog
catalog: ## Inspect the catalog and the split for CONFIG
	asl catalog --config $(CONFIG)

.PHONY: train
train: ## Train the backend named by CONFIG
	asl train --config $(CONFIG)

.PHONY: train-cnn
train-cnn: ## Train the CNN backend
	asl train --config configs/cnn.yaml

.PHONY: smoke
smoke: ## End-to-end run on the committed sample (no Kaggle account needed)
	asl train --config configs/smoke.yaml --recompute-features

.PHONY: runs
runs: ## List every run with its headline metrics
	asl runs list

.PHONY: explain
explain: ## Grad-CAM and feature maps for a CNN run
	asl explain --run $(RUN)

.PHONY: serve
serve: ## Serve RUN over HTTP on :8000
	asl serve --run $(RUN)

.PHONY: docker-build
docker-build: ## Build the inference image
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run: ## Serve RUN from the container (runs/ mounted read-only)
	docker run --rm -p 8000:8000 -e ASL_RUN=$(RUN) -v $(PWD)/runs:/app/runs:ro $(IMAGE)

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
