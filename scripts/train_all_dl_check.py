import argparse
import subprocess
import sys
from pathlib import Path

from jute_disease.utils import get_logger
from jute_disease.utils.constants import BATCH_SIZE, NUM_WORKERS

logger = get_logger(__name__)

CONFIGS_DIR = Path("configs/baselines")
CLI_SCRIPT = "scripts/train_dl.py"


def check_all_dl(
    configs_dir: Path = CONFIGS_DIR, config_file: Path | None = None
) -> None:
    """Run a fast dev run for a single configuration or all configs."""
    if config_file:
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        configs = [config_file]
    else:
        configs = sorted(configs_dir.glob("*.yaml"))

    if not configs:
        raise FileNotFoundError(f"No configs found in {configs_dir}")

    logger.info(f"Starting DL Fast Dev Run: {len(configs)} configs found.")

    failed: list[str] = []
    for config in configs:
        model_name = config.stem
        logger.info(f"Verifying {model_name} (fast_dev_run)...")

        cmd = [
            "uv",
            "run",
            "python",
            CLI_SCRIPT,
            "fit",
            "--config",
            str(config),
            "--trainer.fast_dev_run=True",
            f"--data.num_workers={NUM_WORKERS}",
            "--data.pin_memory=True",
            f"--data.batch_size={BATCH_SIZE}",
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"{model_name} FAILED fast_dev_run.")
            failed.append(model_name)
        else:
            logger.info(f"{model_name} passed.")

    if failed:
        raise RuntimeError(f"Failed models: {failed}")

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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a single baseline YAML config to check.",
    )
    args = parser.parse_args()
    try:
        check_all_dl(configs_dir=args.configs_dir, config_file=args.config)
    except Exception:
        sys.exit(1)
