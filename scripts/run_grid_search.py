import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
import wandb
import yaml

from jute_disease.utils import get_logger

logger = get_logger(__name__)


def _aggregate_metrics(exp_names: list[str], output_csv: Path) -> None:
    """Consolidates metrics from all CSVLogger outputs into a master CSV."""
    results = []

    for exp in exp_names:
        log_dir = Path("artifacts/logs") / exp
        if not log_dir.exists():
            continue

        # Concatenate all versions to merge fit and test outputs
        metrics_files = list(log_dir.glob("version_*/metrics.csv"))
        if not metrics_files:
            continue

        dfs = [pd.read_csv(f) for f in metrics_files]
        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        summary = {"Experiment": exp}

        if "val_loss" in df.columns:
            val_df = df.dropna(subset=["val_loss"])
            if not val_df.empty:
                best_row = val_df.loc[val_df["val_loss"].idxmin()].to_dict()
                for key, val in best_row.items():
                    if (
                        key.startswith("val_")
                        or key.startswith("train_")
                        or key in ["epoch", "step"]
                    ):
                        summary[key] = val

        # Ensure we have train_acc even if not in the best validation row
        for col in ["train_acc", "train_loss"]:
            if col in df.columns and (col not in summary or pd.isna(summary[col])):
                summary[col] = df[col].max() if "acc" in col else df[col].min()

        if "test_loss" in df.columns:
            test_df = df.dropna(subset=["test_loss"])
            if not test_df.empty:
                test_metrics = test_df.iloc[-1].to_dict()
                for key, val in test_metrics.items():
                    if key.startswith("test_"):
                        summary[key] = val

        results.append(summary)

    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_csv, index=False)
        logger.info(f"Summary metrics exported to {output_csv}")
    else:
        logger.warning(f"No metrics found to export to {output_csv}")


def _get_modified_base_config(
    base_config_path: str | Path, exp_name: str, wandb_group: str
) -> str:
    """Read base config, update loggers/checkpoints, and return temp path."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f) or {}

    trainer_cfg = config.setdefault("trainer", {})

    for cb in trainer_cfg.get("callbacks", []):
        if "ModelCheckpoint" in cb.get("class_path", ""):
            cb.setdefault("init_args", {})["dirpath"] = (
                f"artifacts/checkpoints/{exp_name}"
            )

    # Explicitly define WandbLogger properties to avoid CLI CSVLogger crashes
    loggers = trainer_cfg.get("logger", [])
    if isinstance(loggers, dict):
        loggers = [loggers]

    for logger_cfg in loggers:
        if "WandbLogger" in logger_cfg.get("class_path", ""):
            init_args = logger_cfg.setdefault("init_args", {})
            init_args["name"] = exp_name
            init_args["group"] = wandb_group

    loggers.append(
        {
            "class_path": "lightning.pytorch.loggers.CSVLogger",
            "init_args": {"save_dir": "artifacts/logs", "name": exp_name},
        }
    )
    trainer_cfg["logger"] = loggers

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

    # Detect Mode: Phase 1 or Phase 2
    is_phase_two = len(learning_rates) > 0 and len(weight_decays) > 0

    if is_phase_two:
        logger.info(f"Detected Phase 2 Optimizer Grid Search for {model_name}.")
        level_name = locked_params.get("name", "locked")
        weights_path = locked_params.get("weights_path", "imagenet")
        dropout = locked_params.get("dropout_rate", 0.0)

        run_exp_names = []

        for lr in learning_rates:
            for wd in weight_decays:
                logger.info(
                    f"\n=== Running Phase 2: {model_name} | LR: {lr} | WD: {wd} ==="
                )

                ckpt_arg = weights_path if weights_path != "imagenet" else "null"
                pretrained_arg = str(weights_path == "imagenet")

                # Share WandB Run ID between fit and test
                run_id = wandb.util.generate_id()
                env = os.environ.copy()
                env["WANDB_RUN_ID"] = run_id

                short_level = level_name.replace("level_", "l")
                exp_name = f"{model_name.lower()}-{short_level}-lr_{lr}-wd_{wd}"
                wandb_group = f"{model_name}_Finetune_Grid"
                run_exp_names.append(exp_name)
                exp_config_path = _get_modified_base_config(
                    base_config_path, exp_name, wandb_group
                )

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
                ]

                logger.info(f"Command (Test): {' '.join(test_cmd)}")
                try:
                    subprocess.run(test_cmd, env=env, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error testing Phase 2 experiment {exp_name}: {e}")

        _aggregate_metrics(
            run_exp_names,
            output_csv=Path(
                f"artifacts/grid_search_{model_name.lower()}_phase2_metrics.csv"
            ),
        )
        return

    # Phase 1 Execution
    run_exp_names = []
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

            short_level = level_name.replace("level_", "l")
            exp_name = f"{model_name.lower()}-{short_level}-dr_{dropout}"
            wandb_group = f"{model_name}_Transfer_Grid"
            run_exp_names.append(exp_name)
            exp_config_path = _get_modified_base_config(
                base_config_path, exp_name, wandb_group
            )

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
            ]

            if "learning_rate" in fixed_params:
                cmd.append(f"--model.lr={fixed_params['learning_rate']}")
            if "weight_decay" in fixed_params:
                cmd.append(f"--model.weight_decay={fixed_params['weight_decay']}")
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
            ]

            logger.info(f"Command (Test): {' '.join(test_cmd)}")
            try:
                subprocess.run(test_cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error testing Phase 1 experiment {exp_name}: {e}")

    _aggregate_metrics(
        run_exp_names,
        output_csv=Path(
            f"artifacts/grid_search_{model_name.lower()}_phase1_metrics.csv"
        ),
    )


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
