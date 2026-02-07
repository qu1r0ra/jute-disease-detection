import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_FILES = sorted([str(f) for f in CONFIG_DIR.glob("*.yaml")])


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skipping slow DL integration test in CI"
)
@pytest.mark.parametrize("config_path", CONFIG_FILES)
def test_dl_fast_dev_run(config_path):
    """
    Running a fast_dev_run for each config file to ensure:
    1. Config syntax is correct.
    2. Model backbone can be instantiated.
    3. Data module is compatible.
    4. Training loop (1 forward/backward pass) works.
    """
    project_root = Path(__file__).resolve().parents[1]
    cli_path = project_root / "src" / "jute_disease" / "engines" / "cli.py"

    # Construct command
    # uv run python src/jute_disease/engines/cli.py fit --config ...
    cmd = [
        sys.executable,
        str(cli_path),
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

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = (
            f"Training failed for config: {config_path}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        pytest.fail(error_msg)

    assert result.returncode == 0
