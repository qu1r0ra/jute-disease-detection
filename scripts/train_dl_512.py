"""Script to run full training, testing, and evaluation for DL at 512px resolution."""

import os
import subprocess
import sys
from pathlib import Path

import wandb
import yaml

from jute_disease.utils import flatten_log_version, get_logger
from jute_disease.utils.constants import CHECKPOINTS_DIR

logger = get_logger(__name__)

CONFIG_PATH = Path("configs/experiments/mobilenet_v2_512.yaml")
CLI_SCRIPT = "scripts/train_dl.py"


def run_dl_512() -> None:
    """Execute training, evaluation, and aggregation sequentially."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    logger.info("Starting DL 512px Training Pipeline...")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    loggers = cfg.get("trainer", {}).get("logger", [])
    if isinstance(loggers, dict):
        loggers = [loggers]

    csv_save_dir = "artifacts/logs"
    csv_name = CONFIG_PATH.stem
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

    # 1. Fit
    fit_cmd = [
        "uv",
        "run",
        "python",
        CLI_SCRIPT,
        "fit",
        "--config",
        str(CONFIG_PATH),
    ]
    logger.info(f"Running fit for {CONFIG_PATH.stem}...")
    result = subprocess.run(fit_cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Failed during fit with exit code {result.returncode}.")
    flatten_log_version(log_dir, "train-metrics.csv")

    # 2. Test
    ckpt_dir = CHECKPOINTS_DIR / "mobilenet_v2_512"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Get latest checkpoint
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}.")
    best_ckpt = ckpts[0]

    test_cmd = [
        "uv",
        "run",
        "python",
        CLI_SCRIPT,
        "test",
        "--config",
        str(CONFIG_PATH),
        "--ckpt_path",
        str(best_ckpt),
    ]
    logger.info(f"Running test for {CONFIG_PATH.stem} using {best_ckpt.name}...")
    result = subprocess.run(test_cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Failed during test with exit code {result.returncode}.")
    flatten_log_version(log_dir, "test-metrics.csv")

    # 3. Aggregate
    agg_cmd = [
        "uv",
        "run",
        "python",
        "scripts/aggregate_results.py",
        "--exp-names",
        "mobilenet_v2_512px",
        "--output",
        "artifacts/logs/resolution_exps/summary_metrics.csv",
    ]
    logger.info("Running metric aggregation...")
    result = subprocess.run(agg_cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed during aggregation with exit code {result.returncode}."
        )

    logger.info("512px Pipeline completed successfully!")


if __name__ == "__main__":
    try:
        run_dl_512()
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
