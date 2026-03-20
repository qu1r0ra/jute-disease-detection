import logging
import re
from pathlib import Path

import pytest

from jute_disease.models.dl.backbone import TimmBackbone
from jute_disease.utils.constants import CHECKPOINTS_DIR


@pytest.mark.slow
def test_pretrained_checkpoints_key_matching(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify pretrained checkpoints load without massive missing keys."""
    ckpt_dir = CHECKPOINTS_DIR / "pretrained"
    if not ckpt_dir.exists():
        pytest.skip("Pretrained checkpoints directory not found.")

    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        pytest.skip("No pretrained checkpoints downloaded.")

    caplog.set_level(logging.INFO)

    for ckpt_path in checkpoints:
        caplog.clear()

        # Loading the checkpoint will automatically print the IncompatibleKeys msg
        TimmBackbone(
            model_name="mobilenetv2_100",
            pretrained=False,
            checkpoint_path=str(ckpt_path),
        )

        # Find the log containing the key load status
        load_msg = next(
            (
                rec.message
                for rec in caplog.records
                if "Grid Search backbone load status" in rec.message
            ),
            None,
        )
        assert load_msg is not None, f"Load log missing for {ckpt_path.name}"

        # PyTorch strict=False prints IncompatibleKeys msg
        # We want to ensure missing_keys is either empty or very short.
        # If the prefix bug occurred, missing_keys contains hundreds of weights.

        missing_match = re.search(r"missing_keys=\[(.*?)\]", load_msg)
        if missing_match:
            missing_str = missing_match.group(1).replace("'", "").replace('"', "")
            missing_keys = (
                [k.strip() for k in missing_str.split(",")] if missing_str else []
            )

            # Allow a tiny number of missing keys (e.g. classifier bias).
            assert len(missing_keys) < 10, (
                f"Checkpoint {ckpt_path.name} had {len(missing_keys)} missing "
                f"core keys! Preview: {missing_keys[:5]}"
            )
