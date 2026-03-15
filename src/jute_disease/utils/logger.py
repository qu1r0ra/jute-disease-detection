import logging
import os
import shutil
import sys
from logging import Logger
from pathlib import Path

import wandb
from dotenv import load_dotenv


def flatten_log_version(log_dir: Path, target_name: str) -> None:
    """Takes the latest parameter's `version_*` folder and flattens
    its metrics out.
    """
    if not log_dir.exists():
        return
    versions = sorted([d for d in log_dir.glob("version_*") if d.is_dir()])
    if not versions:
        return
    latest_version = versions[-1]
    metrics_file = latest_version / "metrics.csv"
    if metrics_file.exists():
        shutil.move(str(metrics_file), str(log_dir / target_name))
    shutil.rmtree(str(latest_version))


def setup_logging(level: int = logging.INFO) -> None:
    """Set up standardized logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Check if handlers already exist to avoid duplicate logs
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    # Silence noisy loggers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def get_logger(name: str, level: int = logging.INFO) -> "Logger":
    """Get a logger with the specified name and ensure logging is set up."""
    setup_logging(level)
    return logging.getLogger(name)


def setup_wandb() -> None:
    """Load environment variables and login to WandB."""
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    elif os.getenv("WANDB_MODE") in ["disabled", "offline"]:
        pass
    else:
        wandb.login()
