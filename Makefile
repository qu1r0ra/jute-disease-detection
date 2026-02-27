.PHONY: help data setup-data split-data train-ml train-dl train-dl-check test test-all lint format clean sync-nb

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
	@echo "  make train-dl-single   - Run training for a single DL model (MODEL=<model_name>)"
	@echo "  make train-dl-check 	- Run all DL experiments with fast dev run"
	@echo "  make train-dl-check-single MODEL=<model_name> - Run fast dev run for a single DL model"
	@echo "  make train-cv     		- Run cross-validation for a specific model (default 5 folds)"
	@echo "  make grid-search  		- Run grid search experiment using the generic template"
	@echo "  make pretrain     		- Run pre-training script on external data"
	@echo "  make test         		- Run fast tests (slow tests skipped by default)"
	@echo "  make test-all     		- Run all tests including slow ones"
	@echo "  make lint         		- Run linting (ruff check)"
	@echo "  make format       		- Run formatting (ruff format)"
	@echo "  make clean        		- Remove temporary files and logs"
	@echo "  make clean-artifacts  	- Remove all generated artifacts (models, checkpoints)"
	@echo "  make clean-ml     		- Remove ML models and extracted features from artifacts"
	@echo "  make sync-nb    		- Sync Jupyter Notebooks (.ipynb) with Jupytext (.py) scripts"

data:
	$(PYTHON) src/jute_disease/data/utils.py init

setup-data:
	$(PYTHON) src/jute_disease/data/utils.py setup

split-data:
	$(PYTHON) src/jute_disease/data/utils.py split

train-ml:
	$(PYTHON) scripts/train_all_ml.py

train-dl:
	$(PYTHON) scripts/train_all_dl.py

train-dl-single:
	$(PYTHON) scripts/train_all_dl.py --config configs/baselines/$(MODEL).yaml

train-dl-check:
	$(PYTHON) scripts/train_all_dl_check.py

train-dl-check-single:
	$(PYTHON) scripts/train_all_dl_check.py --config configs/baselines/$(MODEL).yaml

train-cv:
	$(PYTHON) scripts/train_cross_validation.py configs/baselines/mobilenet_v2.yaml --folds 5

grid-search:
	$(PYTHON) scripts/run_grid_search.py configs/grid/template_grid.yaml

pretrain:
	$(PYTHON) src/jute_disease/engines/dl/pretrain.py \
		--data_dir data/external/plant_village \
		--output_path artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage.ckpt \
		--epochs 5

test:
	uv run pytest -v -s

test-all:
	uv run pytest -v -s -m ''

lint:
	uv run ruff check .

format:
	uv run ruff check --fix . && uv run ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov lightning_logs wandb

clean-artifacts:
	rm -rf artifacts/ml_models artifacts/checkpoints artifacts/logs artifacts/features

clean-ml:
	rm -rf artifacts/ml_models artifacts/features

sync-nb:
	uv run jupytext --sync notebooks/reproducibility/*.py
