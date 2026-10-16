# Artifact status and result traceability

The canonical title is **Benchmarking Classical Machine Learning and Transfer Learning for Data-Constrained Jute Leaf Disease Classification**. The authors report submission to IEEE HNICEM 2026. This repository describes submitted work and supporting software; it does not assert acceptance or proceedings publication. The software DOI is distinct from a conference publication.

## Versions

- `hnicem2026-paper.pdf` is the repository's short-paper copy. Its title and headline results agree with the current README. Its identity with the conference-uploaded file was not independently checked.
- The README links an external full paper with additional visualizations. That file was not available for comparison in this correction; earlier notes about that version do not establish defects in the current short paper.
- The PDF and conference submission are preserved. These notes clarify the repository rather than changing a submitted record.

## Numerical claims

MobileNetV2 scores belong to different experiment configurations. They must not be collapsed into one interchangeable headline number.

| Configuration | Committed evidence | Test accuracy |
| --- | --- | --- |
| Baseline MobileNetV2 | `artifacts/logs/dl_baselines/aggregated_metrics.csv`, MobileNetV2 row | 0.90277779, approximately 90.28%; Table IV prints 90.27% |
| ImageNet transfer, dropout 0.1 | `artifacts/logs/phase1_transfer_grid/aggregated_grid_metrics.csv`, L1/dropout 0.1 row | Approximately 88.33%; a separate transfer/dropout run |
| Learning rate 0.01, weight decay 0.05 | `artifacts/logs/phase2_finetune_grid/aggregated_grid_metrics.csv`, matching row | 0.91388887, approximately 91.39%; Table X and README headline |
| Baseline EfficientNet-B5 | `artifacts/logs/dl_baselines/aggregated_metrics.csv`, matching row | 0.88611114, approximately 88.61% |

The baseline 90.27% versus 90.28% is a reporting precision discrepancy relative to the committed log, not evidence of two independent improvements. The 88.33% value belongs to a separate transfer/dropout configuration. The current short-paper abstract reports tuned Random Forest validation macro-F1 77.73% and test accuracy 72.22%; the earlier draft's Random Forest headline should not be reused. A complete independent rerun of the classical-ML results was not performed here.

The short paper describes PlantDoc as 13 plant species and a flat 27-category preparation in different sections; these are different quantities. This review did not reproduce the source archive's class inventory. Treat older claims of a 17/27-class contradiction as version-specific until that external artifact is compared.

These are single-split coursework experiments, not uncertainty estimates or an external clinical/agricultural validation. No new training or test-set selection was performed for this correction. Dataset terms and final public-link readiness require a separate review; the README records the unresolved redistribution permission for one merged source.
