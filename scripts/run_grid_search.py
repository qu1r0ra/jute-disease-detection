import argparse
import subprocess
from pathlib import Path

import yaml

from jute_disease.utils import get_logger

logger = get_logger(__name__)


def run_grid_search(
    grid_config_path: str | Path, base_config_path: str | Path | None = None
) -> None:
    """
    Reads grid search config and executes training runs using a baseline model config.
    """
    logger.info(f"Reading grid config from {grid_config_path}...")
    with open(grid_config_path) as f:
        grid_config = yaml.safe_load(f)

    if base_config_path is None:
        grid_stem = Path(grid_config_path).stem.replace("_grid", "")
        base_config_path = Path("configs/baselines") / f"{grid_stem}.yaml"
        if not base_config_path.exists():
            raise ValueError(
                f"Base config not found at {base_config_path}. "
                "Ensure your grid file is named <model_name>_grid.yaml "
                "or pass --base-config."
            )

    model_name = Path(base_config_path).stem.capitalize()
    logger.info(f"Using base model config: {base_config_path} ({model_name})")

    transfer_levels = grid_config.get("transfer_learning_levels", [])
    dropout_rates = grid_config.get("dropout_rates", [])

    learning_rates = grid_config.get("learning_rate", [])
    weight_decays = grid_config.get("weight_decay", [])
    locked_params = grid_config.get("locked_params", {})

    fixed_params = grid_config.get("fixed_params", {})

    # Detect Mode: Phase 1 (Transfer Search) or Phase 2 (Optimizer Fine-Tuning)
    is_phase_two = len(learning_rates) > 0 and len(weight_decays) > 0

    if is_phase_two:
        logger.info(f"Detected Phase 2 Optimizer Grid Search for {model_name}.")
        level_name = locked_params.get("name", "locked")
        weights_path = locked_params.get("weights_path", "imagenet")
        dropout = locked_params.get("dropout_rate", 0.0)

        for lr in learning_rates:
            for wd in weight_decays:
                logger.info(
                    f"\n=== Running Phase 2: {model_name} | LR: {lr} | WD: {wd} ==="
                )

                ckpt_arg = weights_path if weights_path != "imagenet" else "null"
                pretrained_arg = str(weights_path == "imagenet")

                cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_dl.py",
                    "fit",
                    "--config",
                    str(base_config_path),
                    f"--model.feature_extractor.init_args.checkpoint_path={ckpt_arg}",
                    f"--model.feature_extractor.init_args.pretrained={pretrained_arg}",
                    f"--model.feature_extractor.init_args.drop_rate={dropout}",
                    f"--model.lr={lr}",
                    f"--model.weight_decay={wd}",
                    f"--trainer.logger.init_args.name={model_name}_lr{lr}_wd{wd}",
                    f"--trainer.logger.init_args.group={model_name}_Finetune_Grid",
                    f"--data.k_fold={fixed_params.get('num_folds', 1)}",
                    f"--data.batch_size={fixed_params.get('batch_size', 32)}",
                    f"--trainer.max_epochs={fixed_params.get('max_epochs', 50)}",
                ]

                logger.info(f"Command: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running Phase 2 experiment lr{lr}_wd{wd}: {e}")

        return

    # Phase 1 Execution
    for level in transfer_levels:
        level_name = level["name"]
        weights_path = level["weights_path"]

        for dropout in dropout_rates:
            logger.info(
                f"\n=== Running Experiment: {model_name} | {level_name} "
                f"| Dropout: {dropout} ==="
            )

            ckpt_arg = weights_path if weights_path != "imagenet" else "null"
            pretrained_arg = str(weights_path == "imagenet")

            base_model_config = str(base_config_path)

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/train_dl.py",
                "fit",
                "--config",
                base_model_config,
                f"--model.feature_extractor.init_args.checkpoint_path={ckpt_arg}",
                f"--model.feature_extractor.init_args.pretrained={pretrained_arg}",
                f"--model.feature_extractor.init_args.drop_rate={dropout}",
                f"--trainer.logger.init_args.name={model_name}_{level_name}_dr{dropout}",
                f"--trainer.logger.init_args.group={model_name}_Transfer_Grid",
                f"--model.lr={fixed_params.get('learning_rate', 0.001)}",
                f"--model.weight_decay={fixed_params.get('weight_decay', 0.01)}",
                f"--data.k_fold={fixed_params.get('num_folds', 1)}",
                f"--data.batch_size={fixed_params.get('batch_size', 32)}",
                f"--trainer.max_epochs={fixed_params.get('max_epochs', 100)}",
            ]

            logger.info(f"Command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running experiment {level_name}_{dropout}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment grid search.")
    parser.add_argument(
        "grid_config",
        help="Path to grid search YAML (e.g., configs/grid/mobilenet_v2_grid.yaml)",
    )
    parser.add_argument(
        "--base-config",
        help=(
            "Optional: Path to base model YAML "
            "(e.g., configs/baselines/mobilenet_v2.yaml). "
            "If omitted, tries to match grid config name."
        ),
    )
    args = parser.parse_args()

    run_grid_search(args.grid_config, args.base_config)
