"""Verify all DL baseline configs pass a fast_dev_run."""

import argparse
import subprocess
import sys
from pathlib import Path

from jute_disease.utils import get_logger

logger = get_logger(__name__)

CONFIGS_DIR = Path("configs/baselines")
TRAIN_SCRIPT = "src/jute_disease/engines/dl/train.py"


def check_all_dl(configs_dir: Path = CONFIGS_DIR):
    configs = sorted(configs_dir.glob("*.yaml"))

    if not configs:
        logger.error(f"No configs found in {configs_dir}")
        sys.exit(1)

    logger.info(f"Starting DL Fast Dev Run — {len(configs)} configs found.")

    failed = []
    for config in configs:
        model_name = config.stem
        logger.info(f"Verifying {model_name} (fast_dev_run)...")

        cmd = [
            "uv", "run", "python", TRAIN_SCRIPT, "fit",
            "--config", str(config),
            "--trainer.fast_dev_run=True",
            "--data.num_workers=2",
            "--data.pin_memory=True",
            "--data.batch_size=32",
            "--trainer.logger=False",
        ]  # fmt: skip

        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"{model_name} FAILED fast_dev_run.")
            failed.append(model_name)
        else:
            logger.info(f"{model_name} passed.")

    if failed:
        logger.error(f"Failed models: {failed}")
        sys.exit(1)

    logger.info("All DL models verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify all DL configs with fast_dev_run."
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=CONFIGS_DIR,
        help="Directory containing baseline YAML configs.",
    )
    args = parser.parse_args()
    check_all_dl(args.configs_dir)
