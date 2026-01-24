.PHONY: help train cli lint format test clean setup-colab setup-hooks

# Check for uv, fallback to python
ifeq (, $(shell which uv))
    PYTHON = python3
    PIP = pip3
else
    PYTHON = uv run python
    PIP = uv pip
endif

help:
	@echo "Available commands:"
	@echo "  make train        - Run manual training"
	@echo "  make cli          - Run LightningCLI training"
	@echo "  make lint         - Run linting (ruff check)"
	@echo "  make format       - Run formatting (ruff format)"
	@echo "  make test         - Run tests"
	@echo "  make pre-commit   - Run all pre-commit hooks"
	@echo "  make setup-hooks  - Install pre-commit hooks"

train:
	$(PYTHON) -m src.jute_disease.engines.train

cli:
	$(PYTHON) -m src.jute_disease.engines.cli

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
	$(PYTHON) -m src.jute_disease.scripts.predict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

setup-colab:
	$(PIP) install -e .
	$(PIP) install pytorch-lightning albumentations
