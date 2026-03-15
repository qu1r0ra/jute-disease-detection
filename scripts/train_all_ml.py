"""Run all ML baseline experiments (all classifiers x all feature types)."""

import argparse
import subprocess
import sys

from jute_disease.models.ml import FEATURE_EXTRACTORS, ML_CLASSIFIERS
from jute_disease.utils import get_logger

logger = get_logger(__name__)

CLASSIFIERS_KEYS = list(ML_CLASSIFIERS.keys())
FEATURE_TYPES_KEYS = list(FEATURE_EXTRACTORS.keys())


def run_all_ml(
    classifiers: list[str] = CLASSIFIERS_KEYS,
    feature_types: list[str] = FEATURE_TYPES_KEYS,
    balanced: bool = True,
) -> None:
    """Execute all combinations of classical ML experiments."""
    total = len(classifiers) * len(feature_types)
    logger.info(f"Starting ML Training Pipeline — {total} experiments.")

    failed: list[str] = []
    for feat in feature_types:
        for clf in classifiers:
            logger.info(f"Training {clf} with {feat} features...")

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/train_ml.py",
                "--classifier",
                clf,
                "--feature_type",
                feat,
            ]

            if balanced:
                cmd.append("--balanced")
            else:
                cmd.append("--no-balanced")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error(f"{clf}/{feat} failed with exit code {result.returncode}.")
                failed.append(f"{clf}/{feat}")
            else:
                logger.info(f"Finished {clf}/{feat}.")

    if failed:
        raise RuntimeError(f"Failed experiments: {failed}")

    logger.info("All ML experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all ML baseline experiments.")
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=CLASSIFIERS_KEYS,
        choices=CLASSIFIERS_KEYS,
        help="Classifiers to run (default: all).",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=FEATURE_TYPES_KEYS,
        choices=FEATURE_TYPES_KEYS,
        help="Feature types to use (default: all).",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="Disable balanced sample weights.",
    )
    args = parser.parse_args()
    try:
        run_all_ml(args.classifiers, args.feature_types, balanced=not args.no_balanced)
    except Exception:
        sys.exit(1)
