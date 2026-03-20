.PHONY: help data setup-data split-data train-ml tune-ml tune-ml-single test-ml train-dl train-dl-single train-dl-check train-dl-check-single train-cv train-dl-512 grid-search grid-search-check grid-search-finetune grid-search-finetune-check pretrain test test-all lint format clean clean-artifacts clean-ml sync-nb aggregate-results full-check

ifeq (, $(shell which uv))
	PYTHON = python3
	PIP = pip3
else
	PYTHON = uv run python
	PIP = uv pip
endif

help:
	@echo "Available commands:"
	@echo ""
	@echo "-- Setup & Data --"
	@echo "  make data                 - Initialize data (download, setup, & split)"
	@echo "  make setup-data           - Create directory structure"
	@echo "  make split-data           - Create train/val/test splits"
	@echo "  make pretrain             - Run pre-training on external data"
	@echo ""
	@echo "-- Classical ML --"
	@echo "  make train-ml             - Run all classical ML experiments"
	@echo "  make tune-ml              - Tune hyperparameters for all ML classifiers"
	@echo "  make tune-ml-single       - Tune a single ML classifier (args: CLASSIFIER=rf FEATURE=raw)"
	@echo "  make test-ml              - Evaluate a single ML model (args: CLASSIFIER=rf FEATURE=raw)"
	@echo ""
	@echo "-- Deep Learning --"
	@echo "  make train-dl             - Run all baseline DL experiments"
	@echo "  make train-dl-single      - Run training for one DL model (args: MODEL=resnet_50)"
	@echo "  make train-dl-check       - Run DL baselines with fast_dev_run"
	@echo "  make train-cv             - Run cross-validation (default 5 folds)"
	@echo "  make train-dl-512         - Run high-resolution MobileNet experiment"
	@echo ""
	@echo "-- Grid Search & Optimization --"
	@echo "  make grid-search          - Run Phase 1 grid search (Transfer Learning/Dropout)"
	@echo "  make grid-search-check    - Run Phase 1 grid search using fast_dev_run"
	@echo "  make grid-search-finetune - Run Phase 2 fine-tuning search (LR/Weight Decay)"
	@echo ""
	@echo "-- Quality Control & Testing --"
	@echo "  make test                 - Run fast tests (skips slow tests)"
	@echo "  make test-all             - Run all tests including slow ones"
	@echo "  make lint                 - Run ruff check"
	@echo "  make format               - Run ruff check --fix and ruff format"
	@echo "  make full-check           - Run format, sync-nb, and test-all before commit"
	@echo ""
	@echo "-- Maintenance & Utilities --"
	@echo "  make sync-nb              - Sync Jupyter Notebooks (.ipynb) with Jupytext (.py)"
	@echo "  make aggregate-results    - Aggregate metrics (args: EXPS=\"exp1 exp2\")"
	@echo "  make clean                - Remove temporary files and lightning logs"
	@echo "  make clean-artifacts      - Remove all checkpoints and logs"
	@echo "  make clean-ml             - Remove ML models and features"

# Setup & Data
data:
	$(PYTHON) src/jute_disease/data/utils.py init

setup-data:
	$(PYTHON) src/jute_disease/data/utils.py setup

split-data:
	$(PYTHON) src/jute_disease/data/utils.py split

pretrain:
	$(PYTHON) src/jute_disease/engines/dl/pretrain.py \
--data_dir data/external/plantvillage \
--output_path artifacts/checkpoints/pretrained/mobilenet_v2-plantvillage.ckpt \
--epochs 5

# Classical ML
train-ml:
	$(PYTHON) scripts/train_all_ml.py

tune-ml:
	$(PYTHON) scripts/tune_all_ml.py

tune-ml-single:
	$(PYTHON) scripts/tune_ml.py --classifier $(CLASSIFIER) --feature_type $(FEATURE)

test-ml:
	$(PYTHON) scripts/test_ml.py --classifier $(CLASSIFIER) --feature_type $(FEATURE)

# Deep Learning
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

train-dl-512:
	$(PYTHON) scripts/train_dl_512.py

# Grid Search & Optimization
grid-search:
	$(PYTHON) scripts/run_grid_search.py configs/grid/mobilenet_v2_grid.yaml

grid-search-check:
	$(PYTHON) scripts/run_grid_search.py configs/grid/mobilenet_v2_grid.yaml --fast-dev-run

grid-search-finetune:
	$(PYTHON) scripts/run_grid_search.py configs/grid/mobilenet_v2_finetune_grid.yaml --base-config configs/baselines/mobilenet_v2.yaml

grid-search-finetune-check:
	$(PYTHON) scripts/run_grid_search.py configs/grid/mobilenet_v2_finetune_grid.yaml --base-config configs/baselines/mobilenet_v2.yaml --fast-dev-run

# Quality Control & Testing
test:
	$(PYTHON) -m pytest -v -s

test-all:
	$(PYTHON) -m pytest -v -s -m "" ""

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

full-check: format sync-nb test-all

# Maintenance & Utilities
sync-nb:
	$(PYTHON) -m jupytext --sync notebooks/**/*.py

aggregate-results:
	$(PYTHON) scripts/aggregate_results.py --exp-names $(EXPS)

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
