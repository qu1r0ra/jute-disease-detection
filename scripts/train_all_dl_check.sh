#!/bin/bash
set -e

echo "Starting DL Fast Dev Run Pipeline..."
echo "Finding all configs in configs/*.yaml..."

for config in configs/*.yaml; do
    model_name=$(basename "$config" .yaml)

    echo "----------------------------------------------------------------"
    echo "Verifying ${model_name} (Fast Dev Run)..."
    echo "----------------------------------------------------------------"

    uv run python src/jute_disease/engines/cli.py fit \
        --config "$config" \
        --trainer.fast_dev_run=True \
        --data.num_workers=2 \
        --data.pin_memory=True \
        --data.batch_size=32 \
        --trainer.logger=False

    echo "Success: ${model_name} passed fast_dev_run."
    echo ""
done

echo "All DL models verified!"
