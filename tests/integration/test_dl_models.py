"""Integration test: fast_dev_run through each baseline DL config.

Marked as slow — skipped by default. Run explicitly with:
    pytest -m slow tests/test_dl_models.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "baselines"
CONFIG_FILES = sorted(str(f) for f in CONFIG_DIR.glob("*.yaml"))
TRAIN_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "jute_disease"
    / "engines"
    / "dl"
    / "train.py"
)


@pytest.mark.slow
@pytest.mark.parametrize("config_path", CONFIG_FILES)
def test_dl_fast_dev_run(config_path):
    """
    Smoke-test each baseline config end-to-end:
    1. Config syntax is valid.
    2. Model + backbone instantiate correctly.
    3. DataModule is compatible.
    4. One forward + backward pass completes without error.
    """
    project_root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "fit",
        "--config",
        config_path,
        "--trainer.fast_dev_run=True",
        "--trainer.accelerator=cpu",
        "--trainer.devices=1",
        "--trainer.logger=False",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["WANDB_MODE"] = "disabled"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        pytest.fail(
            f"Training failed for config: {config_path}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
