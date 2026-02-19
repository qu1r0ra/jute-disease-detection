# Project Architecture

This document describes the architectural design and directory structure of the `jute-disease-detection` project.

## Directory Structure Overview

```text
.
├── artifacts/              # Generated models, checkpoints, and logs
├── configs/                # Lightning CLI configuration files (.yaml)
├── data/                   # Dataset storage (ignored by git, except structure)
├── docs/                   # Project documentation
├── notebooks/              # Jupyter notebooks for EDA and reproducibility
├── scripts/                # Utility scripts for training and grid search
├── src/                    # Source code
│   ├── annotator/          # Flask-based web app for image annotation
│   └── jute_disease/       # Main library package
│       ├── data/           # LightningDataModules and transforms
│       ├── engines/        # Training/Inference entry points (CLI)
│       ├── models/         # Model architectures (DL and ML)
│       └── utils/          # Shared utilities (constants, logging, seed)
└── tests/                  # Hierarchical test suite mirroring src/
```

## Core Design Principles

### 1. Unified Public API

Each subpackage in `src/jute_disease/` (e.g., `models.ml`, `data`, `utils`) uses `__init__.py` to expose a clean, flattened public API.

- **Internal developers** use deep imports to avoid circular dependencies.
- **External consumers** (engines, scripts, tests) use the clean package-level imports (e.g., `from jute_disease.models.ml import RandomForest`).

### 2. Deep Learning Service (Lightning + Timm)

The DL pipeline is built using **PyTorch Lightning** for state-of-the-art reproducibility and boilerplate reduction.

- **LightningModule**: The `Classifier` class handles the training loop, optimization, and metrics.
- **Backbone System**: Uses a generic `TimmBackbone` to wrap any model from the `timm` library, allowing for easy experimentation with different architectures (ResNet, MobileViT, etc.).
- **Lightning CLI**: Training is driven by configuration files in `configs/`, promoting "Configuration as Code".

### 3. Machine Learning Framework (Scikit-learn Adapters)

Classical ML models are integrated using a custom adapter pattern.

- **Feature Extractors**: Classes like `HandcraftedFeatureExtractor` convert raw images into numerical vectors (HSV, LBP, HOG).
- **Adapters**: The `SklearnClassifier` base class wraps standard scikit-learn estimators to provide a consistent `fit`/`predict`/`save`/`load` interface across the project.

### 4. Data Management & Reproducibility

- **Automated Pipeline**: `utils/data_utils.py` handles consistent dataset splitting into `train`, `val`, and `test` sets based on fixed seeds.
- **Seed Everything**: A centralized `seed_everything` utility ensures deterministic behavior across Python, Numpy, and PyTorch.

### 5. Testing Strategy

The suite is divided into:

- **Fast Unit Tests**: Target individual functions and classes (under 20s).
- **Slow Integration Tests**: Smoke-test the full training pipeline using `fast_dev_run` (marked with `@pytest.mark.slow`).
