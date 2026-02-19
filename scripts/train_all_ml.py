"""Run all ML baseline experiments (all classifiers x all feature types)."""

import argparse
import subprocess
import sys

from jute_disease.utils import get_logger

logger = get_logger(__name__)

TRAIN_SCRIPT = "src/jute_disease/engines/ml/train.py"

CLASSIFIERS = ["rf", "svm", "knn", "lr", "mnb"]
FEATURE_TYPES = ["handcrafted", "raw"]


def run_all_ml(
    classifiers: list[str] = CLASSIFIERS,
    feature_types: list[str] = FEATURE_TYPES,
    balanced: bool = True,
):
    total = len(classifiers) * len(feature_types)
    logger.info(f"Starting ML Training Pipeline — {total} experiments.")

    failed = []
    for feat in feature_types:
        for clf in classifiers:
            logger.info(f"Training {clf} with {feat} features...")

            cmd = [
                "uv", "run", "python", TRAIN_SCRIPT,
                "--classifier", clf,
                "--feature_type", feat,
            ]  # fmt: skip
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
        logger.error(f"Failed experiments: {failed}")
        sys.exit(1)

    logger.info("All ML experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all ML baseline experiments.")
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=CLASSIFIERS,
        choices=CLASSIFIERS,
        help="Classifiers to run (default: all).",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=FEATURE_TYPES,
        choices=FEATURE_TYPES,
        help="Feature types to use (default: all).",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="Disable balanced sample weights.",
    )
    args = parser.parse_args()
    run_all_ml(args.classifiers, args.feature_types, balanced=not args.no_balanced)
