"""Run all DL baseline experiments sequentially."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import wandb
import yaml

from jute_disease.utils import flatten_log_version, get_logger
from jute_disease.utils.constants import CHECKPOINTS_DIR

logger = get_logger(__name__)

CONFIGS_DIR = Path("configs/baselines")
CLI_SCRIPT = "scripts/train_dl.py"


def run_all_dl(
    configs_dir: Path = CONFIGS_DIR, config_file: Path | None = None
) -> None:
    """Execute training for a single configuration or iterate through all."""
    if config_file:
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        configs = [config_file]
    else:
        configs = sorted(configs_dir.glob("*.yaml"))

    if not configs:
        raise FileNotFoundError(f"No configs found in {configs_dir}")

    logger.info(f"Starting DL Training Pipeline — {len(configs)} configs found.")

    for config in configs:
        model_name = config.stem

        logger.info(f"Training {model_name} (config: {config})...")

        with open(config) as f:
            cfg = yaml.safe_load(f) or {}

        loggers = cfg.get("trainer", {}).get("logger", [])
        if isinstance(loggers, dict):
            loggers = [loggers]

        csv_save_dir = "artifacts/logs"
        csv_name = model_name
        for logger_cfg in loggers:
            if "CSVLogger" in logger_cfg.get("class_path", ""):
                init_args = logger_cfg.get("init_args", {})
                csv_save_dir = init_args.get("save_dir", csv_save_dir)
                csv_name = init_args.get("name", csv_name)
                break

        log_dir = Path(csv_save_dir) / csv_name

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
            raise RuntimeError(
                f"{model_name} failed during fit with exit code {result.returncode}."
            )
        flatten_log_version(log_dir, "train-metrics.csv")

        logger.info(f"Testing {model_name}...")

        ckpt_dir = CHECKPOINTS_DIR / model_name
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoint found for {model_name} in {ckpt_dir}."
            )

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
            raise RuntimeError(
                f"{model_name} failed during test with exit code {result.returncode}."
            )
        flatten_log_version(log_dir, "test-metrics.csv")

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
    try:
        run_all_dl(configs_dir=args.configs_dir, config_file=args.config)
    except Exception:
        sys.exit(1)
