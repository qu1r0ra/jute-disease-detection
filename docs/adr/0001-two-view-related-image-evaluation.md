# 1. Evaluate related-image policies under two test views

Status: accepted, 2026-10-31. Canonical protocol: [issue 105](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/105#issuecomment-6543607838).

## Context

The study compares three split policies on each jute pool: image-random (diagnostic), exact-copy grouped and near-duplicate grouped. Independent plan review found that a single test design cannot answer both parts of the question. If each policy keeps its own test set, the policies differ in evaluation support, so score changes mix training differences with test leakage. If all policies share one protected test set, image-random loses the test leakage that makes it inflated, so the comparison no longer measures inflation.

## Decision

Report two views, each with its own manifests and fits:

- View 1, policy-native test: each policy draws its own train/validation/test split by its own rule. This measures the inflation a naive split would report.
- View 2, common protected test: one held-out set of whole near-duplicate families per pool and redraw, shared by all policies and excluded from all View 2 fitting and selection. This measures sensitivity to the training/validation partition on identical evaluation support.

A class-stratified, family-disjoint tuning reserve sits outside every final split in both views. Shared settings are selected on it and frozen before final repetitions. No checkpoint crosses views.

## Alternatives considered

- Policy-native tests only: measures inflation, but cannot separate training effects from differing evaluation support.
- One common protected test only: isolates partition effects, but removes the leakage the study sets out to measure.

## Consequences

- The number of target fits doubles; [issue 106](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/106) handles the budget, and EfficientNet-B5 was dropped from the main matrix partly for this reason.
- Each pool must hold enough families per class for a protected test, policy-native splits and the tuning reserve. Retained support is verified per view and split, and shortfalls return to the owner.
- Results report the two views separately; neither view alone supports the full claim.
