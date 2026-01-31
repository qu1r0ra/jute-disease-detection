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
	@echo "  make setup-data   - Create dataset folder structure from class files"
	@echo "  make split-data   - Split data from by_class into ml_split"
	@echo "  make train        - Run manual training"
	@echo "  make cli          - Run LightningCLI training"
	@echo "  make lint         - Run linting (ruff check)"
	@echo "  make format       - Run formatting (ruff format)"
	@echo "  make test         - Run tests"
	@echo "  make pre-commit   - Run all pre-commit hooks"
	@echo "  make setup-hooks  - Install pre-commit hooks"

setup-data:
	./scripts/create_data_folders.sh

split-data:
	$(PYTHON) -m src.jute_disease_pest.engines.split

train:
	$(PYTHON) -m src.jute_disease_pest.engines.train

cli:
	$(PYTHON) -m src.jute_disease_pest.engines.cli

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

pre-commit:
	uv tool run pre-commit run --all-files

setup-hooks:
	uv tool run pre-commit install

test:
	$(PYTHON) -m pytest

predict:
	$(PYTHON) -m src.jute_disease_pest.scripts.predict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

setup-colab:
	$(PIP) install -e .
	$(PIP) install pytorch-lightning albumentations

run-app:
	$(PYTHON) -m src.annotator.run

ingest:
	$(PYTHON) -m src.annotator.ingest
