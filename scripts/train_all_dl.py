"""Run all DL baseline experiments sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path

from jute_disease.utils import get_logger

logger = get_logger(__name__)

CONFIGS_DIR = Path("configs/baselines")
CLI_SCRIPT = "src/jute_disease/engines/dl/cli.py"


def run_all_dl(configs_dir: Path = CONFIGS_DIR) -> None:
    """Iterate through all configuration files and execute training."""
    configs = sorted(configs_dir.glob("*.yaml"))

    if not configs:
        logger.error(f"No configs found in {configs_dir}")
        sys.exit(1)

    logger.info(f"Starting DL Training Pipeline — {len(configs)} configs found.")

    for config in configs:
        model_name = config.stem
        logger.info(f"Training {model_name} (config: {config})...")

        cmd = ["uv", "run", "python", CLI_SCRIPT, "fit", "--config", str(config)]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"{model_name} failed with exit code {result.returncode}.")
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
    args = parser.parse_args()
    run_all_dl(args.configs_dir)
