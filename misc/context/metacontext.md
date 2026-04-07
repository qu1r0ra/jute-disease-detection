# Project Context and Technical Decisions

This file documents key technical decisions and findings from the project. It ensures that researchers and future maintainers understand why certain paths were chosen and why others were abandoned.

## Key Discoveries and Bottlenecks

### 1. The "Data-Level Ceiling" (Label Ambiguity)

- **Insight**: Performance is strictly capped at ~91.4% accuracy (Test Loss: 0.266) by the dataset's single-label nature.
- **Problem**: Many jute leaves exhibit overlapping symptoms (e.g., Mosaic + Cercospora). In a multi-class setup, the model is penalized for recognizing one symptom if the ground truth only labels the other.
- **Evidence**: Exhaustive finetuning grid search on MobileNetV2 with learning rates from 0.01 down to 0.0001 consistently plateaued around 91.4%.
- **Conclusion**: A transition to a multi-label learning framework would be required for further gains.

### 2. The "Resolution Ceiling"

- **Insight**: Increasing input resolution from 256px to 512px for MobileNetV2 results in a ~2.5% performance drop.
- **Reasoning**: Feature dilation (pre-trained kernels are too small relative to features) and limited model capacity (2.2M parameters).
- **Conclusion**: Stick to 256x256 for this architecture.

### 3. Hyperparameter Consistency

- **Note**: Baselines used WD=0.01, while Phase 1 & 2 grids used WD=0.05. The performance difference is negligible relative to the label ambiguity ceiling.

### 4. Metrics and Logging

- **Log Flattening**: A `flatten_log_version` hook in `src/jute_disease/utils/logger.py` intercepts Lightning output to rename them directly to `train-metrics.csv` and `test-metrics.csv` at the experiment root. This avoids nested `version_n` directories.

### 5. Environment and Hooks

- **ggshield Removal**: Removed from the pre-commit pipeline to avoid stashing conflicts during commits. Manual checks are preferred.
- **Mypy Scoping**: The `notebooks/` directory is excluded from `mypy` checks to allow for exploratory speed.
- **Cleanup**: Redundant packages have been purged. Development tools are grouped under `dev` dependencies. Flask-related dependencies are retained for legacy support of the annotator tool.

### 6. Repository Polishing

- **Asset Flattening**: The `assets/` directory was flattened by removing the `figures/` subfolder. Visualizations are now organized directly under `assets/ml/` and `assets/dl/`.
- **Configuration Centralization**: `src/jute_disease/utils/constants.py` was updated to reflect the new directory structure, replacing `FIGURES_` prefixes with `ML_ASSETS_DIR` and `DL_ASSETS_DIR`.
- **Annotator**: The legacy annotator tool in `src/annotator/` is kept for archival purposes and potential future development but is NOT part of the primary training or evaluation pipeline.

## Implementation Standards

### 1. Tech Stack

- **Python**: 3.11+
- **Environment**: Managed via `uv`.
- **DL Engine**: PyTorch Lightning and `timm`.
- **ML Engine**: Scikit-learn with custom adapters.
- **Task Runner**: `Makefile`.

### 2. Coding Standards

- **Typing**: Use `|` (Python 3.10+) for unions and built-in generics (e.g., `list[str]`).
- **Returns**: All functions/methods must have return type annotations.
- **Error Handling**: Follow "Raise in Logic, Exit in Main" pattern.
- **Formatting**: Strictly follow `ruff` rules.

### 3. Documentation Principles

- **General Style**: Maintain a direct, professional, and clear tone.
- **No Emojis**: Do not use emojis in any documentation or commit messages.
- **Human-Centric**: Avoid AI-generated filler language or overly structured templates.
- **Notebooks**: All notebooks are paired with `.py` scripts via Jupytext.

## Supplementary Details

- **Latent Space**: Analysis uses t-SNE (perplexity=[30, 50, 100, 250]) and UMAP (n_neighbors=15, min_dist=0.1).

## Current Status (April 2026)

- **Status**: **COMPLETED**
- **Final Summary**: All analysis, model selection, tuning, and evaluation are finalized. The dataset's single-label ceiling (91.4%) is documented as the primary bottleneck.
