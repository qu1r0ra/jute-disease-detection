"""Script to run full training, testing, and evaluation for DL at 512px resolution."""

import os
import subprocess
import sys
from pathlib import Path

import wandb

from jute_disease.utils import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path("configs/experiments/mobilenet_v2_512.yaml")
CLI_SCRIPT = "scripts/train_dl.py"


def run_dl_512() -> None:
    """Execute training, evaluation, and aggregation sequentially."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    logger.info("Starting DL 512px Training Pipeline...")

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

    # 2. Test
    ckpt_dir = Path("artifacts/checkpoints/mobilenet_v2_512")
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

    # 3. Aggregate
    agg_cmd = [
        "uv",
        "run",
        "python",
        "scripts/aggregate_results.py",
        "--exp-names",
        "mobilenet_v2_512px",
        "--output",
        "artifacts/grid_search_mobilenet_v2_512px_metrics.csv",
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
