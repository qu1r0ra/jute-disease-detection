.PHONY: help train test lint format clean setup-colab

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
	@echo "  make train        - Run training"
	@echo "  make lint         - Check code style"
	@echo "  make format       - Format code"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Remove temporary files"
	@echo "  make setup-colab  - Install dependencies for Google Colab"

train:
	$(PYTHON) -m src.jute_disease.scripts.train

lint:
	$(PYTHON) -m ruff check src

format:
	$(PYTHON) -m ruff format src

test:
	$(PYTHON) -m src.jute_disease.scripts.test

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
