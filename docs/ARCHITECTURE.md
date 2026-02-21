# Project Architecture

This document describes the architectural design and directory structure of the `jute-disease-detection` project. It serves as a guide for developers and contributors to understand the core components and their interactions.

## Directory Structure Overview

```text
.
├── artifacts/              # Generated models, checkpoints, logs, and experiment results
├── configs/                # Lightning CLI configuration files (.yaml) for DL models
├── data/                   # Dataset storage (by_class/, ml_split/, unlabeled/)
├── docs/                   # Project documentation and specifications
│   ├── agents/             # AI agent-specific directives and context
│   └── ARCHITECTURE.md     # Technical design and architecture overview
├── notebooks/              # Jupyter notebooks for EDA, prototyping, and reproducibility
├── scripts/                # Utility scripts for batch training, grid search, and validation
├── src/                    # Source code package
│   ├── annotator/          # Flask-based web application for image annotation
│   └── jute_disease/       # Main library package
│       ├── data/           # LightningDataModules, transforms, and data utilities
│       ├── engines/        # Training/Inference entry points (DL CLI, ML Training)
│       ├── models/         # Model definitions (DL: MobileViT, ML: Classifiers)
│       └── utils/          # Shared utilities (logging, seeding, constants)
├── tests/                  # Hierarchical test suite mirroring src/ structure
└── AGENTS.md               # Root entry point for AI assistants
```

## Core Design Principles

### 1. Unified Public API

Each subpackage in `src/jute_disease/` (e.g., `models.ml`, `data`, `utils`) uses `__init__.py` to expose a clean, flattened public API.

- **Internal developers** use relative imports where appropriate or `from jute_disease.x import y` to avoid circular dependencies.
- **External consumers** (scripts, tests, notebooks) use the clean package-level imports (e.g., `from jute_disease.models.ml import RandomForest`).

### 2. Deep Learning Service (Lightning + Timm)

The DL pipeline is built using **PyTorch Lightning** for state-of-the-art reproducibility and boilerplate reduction.

- **LightningModule (`Classifier`)**: The core class handling the training loop, optimization, logging, and metrics.
- **Backbone System**: Uses a generic `TimmBackbone` to wrap any model from the `timm` library, allowing for easy experimentation with different architectures (e.g., MobileViT, ResNet). The default backbone is `mobilevit_s`.
- **Lightning CLI**: Training is driven by configuration files in `configs/`, promoting "Configuration as Code". The CLI supports overrides via command-line arguments.

### 3. Machine Learning Framework (Scikit-learn Adapters)

Classical ML models are integrated using a custom adapter pattern to unify them with the DL workflow.

- **Feature Extractors**: Classes like `HandcraftedFeatureExtractor` convert raw images into numerical vectors (HSV, LBP, HOG). `RawPixelFeatureExtractor` handles pixel flattening.
- **Adapters**: The `SklearnClassifier` base class wraps standard scikit-learn estimators to provide a consistent `fit`/`predict`/`save`/`load` interface across the project.
- **Implmentations**: Currently supports Logistic Regression, SVM, Random Forest, KNN, and Multinomial Naive Bayes.

### 4. Data Management & Reproducibility

- **DataModule**:
  - Handles dataset splitting (Train/Val/Test) with fixed seeds.
  - Supports **K-Fold Cross-Validation** via `set_fold()`.
  - Implements **Weighted Random Sampling** to handle class imbalance.
- **Transforms**: Uses **Albumentations** for robust data augmentation and preprocessing.
- **Seed Everything**: A centralized `seed_everything` utility ensures deterministic behavior across Python, Numpy, specific libraries, and PyTorch.

### 5. Code Quality & Standards

- **Type Safety**: The codebase adheres to strict type checking using modern Python 3.10+ syntax (e.g., `list[str] | None`).
- **Formatting**: Code is formatted and linted using `ruff` to ensure PEP 8 compliance.
- **Testing**: A comprehensive test suite (`tests/`) covers unit tests (logic verification) and slower integration tests (end-to-end pipeline).

## Tools & Dependencies

- **uv**: Package management and environment isolation.
- **PyTorch Lightning**: Deep learning framework.
- **timm**: Pretrained computer vision models.
- **Scikit-learn**: Classical machine learning algorithms.
- **Albumentations**: Fast image augmentation library.
- **WandB**: Experiment tracking and visualization.
