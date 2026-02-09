.PHONY: help data setup-data split-data train-ml train-dl train-dl-check test lint format clean

ifeq (, $(shell which uv))
    PYTHON = python3
    PIP = pip3
else
    PYTHON = uv run python
    PIP = uv pip
endif

help:
	@echo "Available commands:"
	@echo "  make data         		- Initialize data (download & split)"
	@echo "  make train-ml     		- Run all classical ML experiments"
	@echo "  make train-dl     		- Run all DL experiments"
	@echo "  make train-dl-check 	- Run all DL experiments with fast dev run"
	@echo "  make train-dl-check-single MODEL=<model_name> - Run fast dev run for a single DL model"
	@echo "  make test         		- Run all tests"
	@echo "  make lint         		- Run linting (ruff check)"
	@echo "  make format       		- Run formatting (ruff format)"
	@echo "  make clean        		- Remove temporary files"

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

train-dl-check:
	bash scripts/train_all_dl_check.sh

train-dl-check-single:
	uv run python src/jute_disease/engines/train.py fit \
		--config configs/baselines/$(MODEL).yaml \
		--trainer.fast_dev_run=True \
		--data.num_workers=2 \
		--data.pin_memory=True \
		--data.batch_size=32 \
		--trainer.logger=False

test:
	uv run pytest -v -s

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
