import argparse
import os
import subprocess
from pathlib import Path

import yaml

import wandb
from jute_disease.utils import get_logger

logger = get_logger(__name__)


def _get_modified_base_config(base_config_path: str | Path, exp_name: str) -> str:
    """Read base config, update ModelCheckpoint dirpath, write and return temp path."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f) or {}

    for cb in config.get("trainer", {}).get("callbacks", []):
        if "ModelCheckpoint" in cb.get("class_path", ""):
            cb.setdefault("init_args", {})["dirpath"] = (
                f"artifacts/checkpoints/{exp_name}"
            )

    temp_dir = Path("artifacts/checkpoints/.temp_configs")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{exp_name}.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(config, f)
    return str(temp_path)


def run_grid_search(
    grid_config_path: str | Path,
    base_config_path: str | Path | None = None,
    fast_dev_run: bool = False,
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

                # Setup Shared WandB Run ID so fit and test map to the same dashboard
                run_id = wandb.util.generate_id()
                env = os.environ.copy()
                env["WANDB_RUN_ID"] = run_id

                exp_name = f"{model_name}_lr{lr}_wd{wd}"
                exp_config_path = _get_modified_base_config(base_config_path, exp_name)

                cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_dl.py",
                    "fit",
                    "--config",
                    exp_config_path,
                    f"--model.feature_extractor.init_args.checkpoint_path={ckpt_arg}",
                    f"--model.feature_extractor.init_args.pretrained={pretrained_arg}",
                    f"--model.feature_extractor.init_args.drop_rate={dropout}",
                    f"--model.lr={lr}",
                    f"--model.weight_decay={wd}",
                    f"--trainer.logger.init_args.name={exp_name}",
                    f"--trainer.logger.init_args.group={model_name}_Finetune_Grid",
                ]

                if "num_folds" in fixed_params:
                    cmd.append(f"--data.k_fold={fixed_params['num_folds']}")
                if "batch_size" in fixed_params:
                    cmd.append(f"--data.batch_size={fixed_params['batch_size']}")
                if "max_epochs" in fixed_params:
                    cmd.append(f"--trainer.max_epochs={fixed_params['max_epochs']}")

                if fast_dev_run:
                    cmd.append("--trainer.fast_dev_run=True")

                logger.info(f"Command (Fit): {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, env=env, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running Phase 2 experiment lr{lr}_wd{wd}: {e}")
                    continue

                logger.info(f"Testing Phase 2 experiment {exp_name}...")

                ckpt_dir = Path("artifacts/checkpoints") / exp_name
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                if not ckpts:
                    logger.error(
                        f"No checkpoint found for Phase 2 {exp_name} in {ckpt_dir}."
                    )
                    continue

                best_ckpt = ckpts[0]

                test_cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_dl.py",
                    "test",
                    "--config",
                    exp_config_path,
                    "--ckpt_path",
                    str(best_ckpt),
                    f"--trainer.logger.init_args.name={exp_name}",
                    f"--trainer.logger.init_args.group={model_name}_Finetune_Grid",
                ]

                logger.info(f"Command (Test): {' '.join(test_cmd)}")
                try:
                    subprocess.run(test_cmd, env=env, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error testing Phase 2 experiment {exp_name}: {e}")

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

            run_id = wandb.util.generate_id()
            env = os.environ.copy()
            env["WANDB_RUN_ID"] = run_id

            exp_name = f"{model_name}_{level_name}_dr{dropout}"
            exp_config_path = _get_modified_base_config(base_config_path, exp_name)

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/train_dl.py",
                "fit",
                "--config",
                exp_config_path,
                f"--model.feature_extractor.init_args.checkpoint_path={ckpt_arg}",
                f"--model.feature_extractor.init_args.pretrained={pretrained_arg}",
                f"--model.feature_extractor.init_args.drop_rate={dropout}",
                f"--trainer.logger.init_args.name={exp_name}",
                f"--trainer.logger.init_args.group={model_name}_Transfer_Grid",
            ]

            if "learning_rate" in fixed_params:
                cmd.append(f"--model.lr={fixed_params['learning_rate']}")
            if "weight_decay" in fixed_params:
                cmd.append(f"--model.weight_decay={fixed_params['weight_decawy']}")
            if "num_folds" in fixed_params:
                cmd.append(f"--data.k_fold={fixed_params['num_folds']}")
            if "batch_size" in fixed_params:
                cmd.append(f"--data.batch_size={fixed_params['batch_size']}")
            if "max_epochs" in fixed_params:
                cmd.append(f"--trainer.max_epochs={fixed_params['max_epochs']}")

            if fast_dev_run:
                cmd.append("--trainer.fast_dev_run=True")

            logger.info(f"Command (Fit): {' '.join(cmd)}")
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running experiment {exp_name}: {e}")
                continue

            logger.info(f"Testing experiment {exp_name}...")

            ckpt_dir = Path("artifacts/checkpoints") / exp_name
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            if not ckpts:
                logger.error(
                    f"No checkpoint found for Phase 1 {exp_name} in {ckpt_dir}."
                )
                continue

            best_ckpt = ckpts[0]

            test_cmd = [
                "uv",
                "run",
                "python",
                "scripts/train_dl.py",
                "test",
                "--config",
                exp_config_path,
                "--ckpt_path",
                str(best_ckpt),
                f"--trainer.logger.init_args.name={exp_name}",
                f"--trainer.logger.init_args.group={model_name}_Transfer_Grid",
            ]

            logger.info(f"Command (Test): {' '.join(test_cmd)}")
            try:
                subprocess.run(test_cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error testing Phase 1 experiment {exp_name}: {e}")


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
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1 batch of training/validation to verify pipeline logic.",
    )
    args = parser.parse_args()

    run_grid_search(args.grid_config, args.base_config, args.fast_dev_run)
