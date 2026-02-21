# Gemini System Instructions

Welcome to the `jute-disease-detection` project. When assisting within this repository, do not rely solely on your baseline training assumptions.

## 1. Mandatory Reading

You must read and adhere to [`CONTEXT.md`](CONTEXT.md). It establishes critical project boundaries:

- **Tooling**: We use `uv` exclusively. No exceptions for `pip` or `conda`.
- **Code Quality**: Strict typing (`| None` and `list[]`). Implicitly typed returns and suppression comments (`# type: ignore`) are forbidden.
- **Data Safety**: Do not script custom dataset splits manually. Datasets are managed automatically by the `DataModule` using fixed K-Fold Cross-Validation and Weighted Random Sampling implementations.

## 2. Architecture Awareness

Read [`ARCHITECTURE.md`](../ARCHITECTURE.md) to comprehend how the PyTorch Lightning and Timm backends harmonize with Scikit-learn wrappers, and how the adjacent Flask Annotator app connects.

## 3. Execution Protocol

- Lean on the included `Makefile` (e.g., `make train-ml`, `make test`) as your primary task runner interface.
- Complete operations and workflows cleanly by invoking `make format` and verifying logic through `make test`.
