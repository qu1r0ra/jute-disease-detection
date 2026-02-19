.PHONY: help data setup-data split-data train-ml train-dl train-dl-check test test-all lint format clean

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
	@echo "  make train-cv     		- Run cross-validation for MobileViT (default 5 folds)"
	@echo "  make grid-search  		- Run grid search experiment using MobileViT grid config"
	@echo "  make pretrain     		- Run pre-training script on external data (PlantVillage)"
	@echo "  make test         		- Run fast tests (slow tests skipped by default)"
	@echo "  make test-all     		- Run all tests including slow ones"
	@echo "  make lint         		- Run linting (ruff check)"
	@echo "  make format       		- Run formatting (ruff format)"
	@echo "  make clean        		- Remove temporary files and logs"
	@echo "  make clean-artifacts  	- Remove all generated artifacts (models, checkpoints)"

data:
	$(PYTHON) -m jute_disease.utils.data_utils init

setup-data:
	$(PYTHON) -m jute_disease.utils.data_utils setup

split-data:
	$(PYTHON) -m jute_disease.utils.data_utils split

train-ml:
	$(PYTHON) scripts/train_all_ml.py

train-dl:
	$(PYTHON) scripts/train_all_dl.py

train-dl-check:
	$(PYTHON) scripts/train_all_dl_check.py

train-dl-check-single:
	uv run python src/jute_disease/engines/dl/train.py fit \
		--config configs/baselines/$(MODEL).yaml \
		--trainer.fast_dev_run=True \
		--data.num_workers=2 \
		--data.pin_memory=True \
		--data.batch_size=32 \
		--trainer.logger=False

train-cv:
	$(PYTHON) scripts/train_cross_validation.py configs/baselines/mobilevit.yaml --folds 5

grid-search:
	$(PYTHON) scripts/run_grid_search.py configs/grid/mobilevit_grid.yaml

pretrain:
	$(PYTHON) src/jute_disease/engines/dl/pretrain.py \
		--data_dir data/external/plant_village \
		--output_path artifacts/checkpoints/pretrained/mobilevit_plantvillage.ckpt \
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
	rm -rf artifacts/ml_models artifacts/checkpoints artifacts/logs
