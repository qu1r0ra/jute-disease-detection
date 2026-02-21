# Jute Disease Detection - Context

Welcome! This document provides crucial context, architectural design choices, and strict coding guidelines for AI assistants (Antigravity, Claude, Gemini, etc.) working on the `jute-disease-detection` project.

**Always adhere to these instructions to maintain consistency across the codebase.**

## 1. Tech Stack & Environment

- **Python Version**: 3.11+ (Modern standard).
- **Environment Management**: We strictly use **`uv`**. Do not suggest `pip install` or `conda install`. Use `uv add` for dependencies, and `uv run` for execution.
- **Task Runner**: A modern `Makefile` wraps common commands (e.g., `make test`, `make format`). Use these over raw commands.
- **Core Libraries**:
  - **Deep Learning**: PyTorch, PyTorch Lightning (Lightning CLI), `timm`.
  - **Machine Learning**: `scikit-learn`.
  - **Data Processing**: `albumentations`, `numpy`, `Pillow`.
  - **Web Application**: `Flask`, `Flask-SQLAlchemy`, `SQLite`.
  - **Quality**: `ruff` (linting/formatting), `pytest` (testing).

## 2. Coding Standards & Conventions

### 2.1. Strict Type Hinting

- **Modern Syntax Only**: Use the `|` operator for unions/optionals (e.g., `str | None` instead of `Optional[str]`).
- **Built-in Generics**: Use `list[str]`, `dict[str, Any]`, `tuple[int, int]` instead of importing from the `typing` module (`List`, `Dict`, `Tuple`).
- **Explicit Returns**: All public methods and functions MUST have an explicit return type. Functions returning nothing must be annotated with `-> None` (this includes all `pytest` test functions and Flask routes).
- **No Type Ignores**: Avoid `# type: ignore`. If the linter complains, write explicit runtime assertions/checks (e.g., `assert foo is not None`) to satisfy the type checker.

### 2.2. Code Quality & Formatting

- **Ruff**: We use `ruff` for all formatting and linting.
- **Line length**: Be mindful of line limits. If a line is too long, break it naturally.
- **Mandatory Step**: _Always_ run `make format` after modifying `.py` files to ensure compliance before concluding a task.

## 3. Architecture Pointers

Review [`ARCHITECTURE.md`](../ARCHITECTURE.md) for full details, but keep these core concepts in mind:

- **Dual-Engine Setup**:
  - **Deep Learning (DL)**: Housed in `src/jute_disease/models/dl/`. Driven by PyTorch Lightning's CLI. Models use a wrapper (`Classifier`) around `timm` backbones (`TimmBackbone`). Default is often `mobilevit_s`.
  - **Machine Learning (ML)**: Housed in `src/jute_disease/models/ml/`. Scikit-learn estimators are wrapped in a generic `SklearnClassifier` adapter. Features are extracted manually (e.g., `HandcraftedFeatureExtractor`, `RawPixelFeatureExtractor`).
- **Web Annotator App**:
  - Housed in `src/annotator/`. A Flask web app used to manually annotate images and visualize predictions. Relies on SQLite (`annotations.db`) and `Flask-SQLAlchemy`.
- **Data Handling**:
  - All data ingestion and splitting are managed by `src/jute_disease/data/datamodule.py` and `utils/data_utils.py`.
  - The `DataModule` implements K-Fold Cross-Validation and Weighted Random Sampling.
- **Public API**: Use `jute_disease.` package imports (e.g., `from jute_disease.models.dl import Classifier`) instead of deep internal relative imports when working outside of the specific subpackage (like in `tests/` or `scripts/`).

## 4. Operational Boundaries

To prevent destructive actions and ensure a smooth workflow:

1. **Dependencies**: **Do not** add new dependencies to `pyproject.toml` unless explicitly requested by the user. If an external library is needed to solve a problem, ask first.
2. **Data Directory**: **Do not** write scripts that manually delete or maliciously modify the `data/` directory. Rely on existing `split_data()` functions.
3. **Training Execution**: Training pipelines (`make train-ml`, `make train-all-dl`) are resource-intensive. **Do not** execute these commands autonomously without user permission.
4. **Configuration over Hardcoding**: For deep learning models, parameters should be adjusted in the YAML files located in `configs/`, not hardcoded in the `.py` files.
