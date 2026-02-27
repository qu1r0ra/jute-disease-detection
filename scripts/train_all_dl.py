"""Run all DL baseline experiments sequentially."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import wandb
from jute_disease.utils import get_logger

logger = get_logger(__name__)

CONFIGS_DIR = Path("configs/baselines")
CLI_SCRIPT = "scripts/train_dl.py"


def run_all_dl(
    configs_dir: Path = CONFIGS_DIR, config_file: Path | None = None
) -> None:
    """Execute training for a single configuration or iterate through all."""
    if config_file:
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            sys.exit(1)
        configs = [config_file]
    else:
        configs = sorted(configs_dir.glob("*.yaml"))

    if not configs:
        logger.error(f"No configs found in {configs_dir}")
        sys.exit(1)

    logger.info(f"Starting DL Training Pipeline — {len(configs)} configs found.")

    for config in configs:
        model_name = config.stem

        logger.info(f"Training {model_name} (config: {config})...")

        run_id = wandb.util.generate_id()
        env = os.environ.copy()
        env["WANDB_RUN_ID"] = run_id

        fit_cmd = [
            "uv",
            "run",
            "python",
            CLI_SCRIPT,
            "fit",
            "--config",
            str(config),
        ]
        result = subprocess.run(fit_cmd, env=env)
        if result.returncode != 0:
            logger.error(
                f"{model_name} failed during fit with exit code {result.returncode}."
            )
            sys.exit(result.returncode)

        logger.info(f"Testing {model_name}...")

        ckpt_dir = Path("artifacts/checkpoints") / model_name
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            logger.error(f"No checkpoint found for {model_name} in {ckpt_dir}.")
            sys.exit(1)

        best_ckpt = ckpts[0]

        test_cmd = [
            "uv",
            "run",
            "python",
            CLI_SCRIPT,
            "test",
            "--config",
            str(config),
            "--ckpt_path",
            str(best_ckpt),
        ]
        result = subprocess.run(test_cmd, env=env)
        if result.returncode != 0:
            logger.error(
                f"{model_name} failed during test with exit code {result.returncode}."
            )
            sys.exit(result.returncode)

        logger.info(f"Finished {model_name}.")

    logger.info("All DL experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all DL baseline experiments.")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=CONFIGS_DIR,
        help="Directory containing baseline YAML configs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a single baseline YAML config to execute.",
    )
    args = parser.parse_args()
    run_all_dl(configs_dir=args.configs_dir, config_file=args.config)
