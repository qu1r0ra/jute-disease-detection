#!/bin/bash
set -e

echo "Starting DL Training Pipeline..."
echo "Finding all configs in configs/*.yaml..."

for config in configs/baselines/*.yaml; do
    model_name=$(basename "$config" .yaml)

    echo "----------------------------------------------------------------"
    echo "Training ${model_name} (Config: ${config})..."
    echo "----------------------------------------------------------------"

    uv run python src/jute_disease/engines/dl/train.py fit \
        --config "$config"

    echo "Finished ${model_name}."
    echo ""
done

echo "All DL experiments completed!"
