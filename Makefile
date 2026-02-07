.PHONY: help data setup-data split-data train-ml train-dl test lint format clean

ifeq (, $(shell which uv))
    PYTHON = python3
    PIP = pip3
else
    PYTHON = uv run python
    PIP = uv pip
endif

help:
	@echo "Available commands:"
	@echo "  make data         - Initialize data (download & split)"
	@echo "  make train-ml     - Run all classical ML experiments"
	@echo "  make train-dl     - Run all deep learning experiments"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Run linting (ruff check)"
	@echo "  make format       - Run formatting (ruff format)"
	@echo "  make clean        - Remove temporary files"

data:
	$(PYTHON) -m jute_disease.utils.data init

setup-data:
	$(PYTHON) -m jute_disease.utils.data setup

split-data:
	$(PYTHON) -m jute_disease.utils.data split

train-ml:
	bash scripts/train_all_ml.sh

train-dl:
	bash scripts/train_all_dl.sh

test:
	uv run pytest -v

lint:
	uv run ruff check .

format:
	uv run ruff check --fix . && uv run ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov
