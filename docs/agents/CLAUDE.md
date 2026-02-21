# Claude System Instructions

When interacting with this repository, prioritize reading the project-specific configuration immediately.

## 1. Primary Source of Truth

Always cross-reference your proposed actions against [`CONTEXT.md`](CONTEXT.md). It outlines strict coding standards, precise dependency management (`uv`), modern static typing rules (no `# type: ignore`), and limits on autonomous modifications to data pipelines.

## 2. Architectural Context

Consult [`ARCHITECTURE.md`](../ARCHITECTURE.md) to understand the project structure and technical bounds, primarily the division between:

- Deep Learning Engine (`src/jute_disease/models/dl/`)
- Machine Learning Engine (`src/jute_disease/models/ml/`)
- Annotator Web App (`src/annotator/`)

## 3. Actionable Directives

- **Do not bypass `uv`**: Always execute project commands and tests via `uv run` or via the proxy commands exposed in the `Makefile`.
- **Check your types**: Ensure explicit return types (e.g., `-> None`, `-> Response`) and strictly utilize modern python type unions (e.g., `| None`).
- **Conclusion**: Before resolving tasks or submitting fixes, run `make format` and `make test` locally to ensure codebase integrity.
