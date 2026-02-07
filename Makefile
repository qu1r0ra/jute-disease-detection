.PHONY: help train cli lint format test clean setup-colab setup-hooks

ifeq (, $(shell which uv))
    PYTHON = python3
    PIP = pip3
else
    PYTHON = uv run python
    PIP = uv pip
endif

help:
	@echo "Available commands:"
	@echo "  make init-data    - Initialize everything: setup folders, split data, and download external sets"
	@echo "  make setup-data   - Create target folder structure"
	@echo "  make split-data   - Split data from by_class into ml_split"
	@echo "  make train        - Run deep learning training"
	@echo "  make train-ml     - Run ML classifier comparative training"
	@echo "  make cli          - Run LightningCLI training"
	@echo "  make lint         - Run linting (ruff check)"
	@echo "  make format       - Run formatting (ruff format)"
	@echo "  make test         - Run tests"
	@echo "  make pre-commit   - Run all pre-commit hooks"
	@echo "  make setup-hooks  - Install pre-commit hooks"

init-data:
	$(PYTHON) -m jute_disease.utils.data init

setup-data:
	$(PYTHON) -m jute_disease.utils.data setup

split-data:
	$(PYTHON) -m jute_disease.utils.data split

train:
	$(PYTHON) -m jute_disease.engines.train

train-ml:
	$(PYTHON) -m jute_disease.engines.train_ml

cli:
	$(PYTHON) -m jute_disease.engines.cli

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

pre-commit:
	uv tool run pre-commit run --all-files

setup-hooks:
	uv tool run pre-commit install

test:
	$(PYTHON) -m pytest

predict:
	$(PYTHON) -m jute_disease.engines.predict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-app:
	$(PYTHON) -m annotator.run

ingest:
	$(PYTHON) -m annotator.utils.indexing
