import argparse

from jute_disease.engines.ml.train import train_ml
from jute_disease.models.ml import FEATURE_EXTRACTORS, ML_CLASSIFIERS
from jute_disease.utils import DEFAULT_SEED


def main() -> None:
    parser = argparse.ArgumentParser(description="Jute Classical ML Training")
    parser.add_argument(
        "--classifier",
        type=str,
        default="rf",
        choices=list(ML_CLASSIFIERS.keys()),
        help="Classical ML classifier to train",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="crafted",
        choices=list(FEATURE_EXTRACTORS.keys()),
        help="Type of features to extract (crafted or raw)",
    )
    parser.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use balanced sample weights during training (default: True)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )

    args = parser.parse_args()
    train_ml(
        classifier=args.classifier,
        feature_type=args.feature_type,
        balanced=args.balanced,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
