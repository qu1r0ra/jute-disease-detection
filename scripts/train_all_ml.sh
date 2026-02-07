#!/bin/bash
set -e

CLASSIFIERS=("rf" "svm" "knn" "lr" "mnb")
FEATURE_TYPES=("handcrafted" "raw")

echo "Starting ML Training Pipeline..."

for feat in "${FEATURE_TYPES[@]}"; do
    for clf in "${CLASSIFIERS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "Training ${clf} using ${feat} features..."
        echo "----------------------------------------------------------------"

        uv run python src/jute_disease/engines/train_ml.py \
            --classifier "$clf" \
            --feature_type "$feat" \
            --balanced

        echo "Finished ${clf} with ${feat} features."
        echo ""
    done
done

echo "All ML experiments completed!"
