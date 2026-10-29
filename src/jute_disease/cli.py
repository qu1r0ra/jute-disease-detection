"""Single console entry point, ``jute-disease``, for the study workflow.

Each recipe in the justfile is a one-line wrapper over a command below.
Commands marked as planned exit non-zero and name the issue that owns them.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI

from jute_disease.config import ARTIFACTS_DIR, DEFAULT_SEED
from jute_disease.data.curation import (
    initialize_data,
    setup_data_directory,
    split_data,
)
from jute_disease.data.datamodule import DataModule
from jute_disease.data.download import download_plant_doc, download_plant_village
from jute_disease.data.synthetic import make_synthetic_data
from jute_disease.models.dl.classifier import Classifier
from jute_disease.models.ml import FEATURE_EXTRACTORS, ML_CLASSIFIERS
from jute_disease.reporting.tables import aggregate_metrics
from jute_disease.training.baselines import CONFIGS_DIR, run_all_dl
from jute_disease.training.cross_validation import run_cross_validation
from jute_disease.training.engines.dl.pretrain import main as pretrain_main
from jute_disease.training.engines.ml.predict import predict_ml
from jute_disease.training.engines.ml.sweep import run_all_ml
from jute_disease.training.engines.ml.test import test_ml
from jute_disease.training.engines.ml.train import train_ml
from jute_disease.training.grid_search import run_grid_search
from jute_disease.training.seed import seed_everything

Handler = Callable[[argparse.Namespace], None]


def _planned(command: str, owner: str) -> Handler:
    def run(_: argparse.Namespace) -> None:
        raise SystemExit(
            f"jute-disease {command} is not implemented yet; owned by {owner}."
        )

    return run


def _data_setup(_: argparse.Namespace) -> None:
    setup_data_directory()


def _data_split(args: argparse.Namespace) -> None:
    seed_everything(DEFAULT_SEED)
    split_data(force=args.force)


def _data_download(_: argparse.Namespace) -> None:
    download_plant_village()
    download_plant_doc()


def _data_init(_: argparse.Namespace) -> None:
    initialize_data()


def _data_synthetic(args: argparse.Namespace) -> None:
    make_synthetic_data(args.per_class, args.size)


def _train_dl(lightning_args: list[str]) -> None:
    load_dotenv()
    LightningCLI(
        model_class=Classifier,
        datamodule_class=DataModule,
        save_config_kwargs={"overwrite": True},
        args=lightning_args,
    )


def _train_ml(args: argparse.Namespace) -> None:
    if args.all:
        run_all_ml(args.classifiers, args.feature_types, balanced=args.balanced)
    else:
        train_ml(
            classifier=args.classifier,
            feature_type=args.feature_type,
            balanced=args.balanced,
            seed=args.seed,
        )


def _train_baselines(args: argparse.Namespace) -> None:
    run_all_dl(configs_dir=args.configs_dir, config_file=args.config)


def _train_cv(args: argparse.Namespace) -> None:
    run_cross_validation(args.config, args.folds)


def _train_grid(args: argparse.Namespace) -> None:
    run_grid_search(args.grid_config, args.base_config, args.fast_dev_run)


def _evaluate_aggregate(args: argparse.Namespace) -> None:
    aggregate_metrics(args.exp_names, args.output)


def _evaluate_ml(args: argparse.Namespace) -> None:
    test_ml(
        classifier=args.classifier,
        feature_type=args.feature_type,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jute-disease",
        description="Jute disease detection: data, training and evaluation.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # data: dataset preparation
    data = commands.add_parser("data", help="Prepare the local dataset.")
    data_actions = data.add_subparsers(
        dest="data_action", metavar="ACTION", required=True
    )
    data_actions.add_parser(
        "setup", help="Create the dataset directory layout."
    ).set_defaults(run=_data_setup)
    split = data_actions.add_parser("split", help="Create train/val/test splits.")
    split.add_argument(
        "--force", action="store_true", help="Rebuild the split if it exists."
    )
    split.set_defaults(run=_data_split)
    data_actions.add_parser(
        "download", help="Download PlantVillage and PlantDoc."
    ).set_defaults(run=_data_download)
    data_actions.add_parser(
        "init", help="Run setup, split and download in order."
    ).set_defaults(run=_data_init)
    synthetic = data_actions.add_parser(
        "synthetic", help="Write a tiny synthetic dataset for smoke tests."
    )
    synthetic.add_argument("--per-class", type=int, default=20)
    synthetic.add_argument("--size", type=int, default=64)
    synthetic.set_defaults(run=_data_synthetic)

    # train: model training and experiment matrices
    train = commands.add_parser("train", help="Train models and run experiments.")
    train_actions = train.add_subparsers(
        dest="train_action", metavar="ACTION", required=True
    )
    # Forwarded commands: main() passes their arguments through unparsed.
    train_actions.add_parser(
        "dl", help="Lightning CLI for DL (e.g. dl fit --config ...)."
    )

    ml = train_actions.add_parser(
        "ml", help="Train one classical model, or every combination with --all."
    )
    ml.add_argument("--all", action="store_true", help="Run every combination.")
    ml.add_argument(
        "--classifier",
        default="rf",
        choices=list(ML_CLASSIFIERS),
        help="Classifier to train (single run).",
    )
    ml.add_argument(
        "--feature_type",
        default="crafted",
        choices=list(FEATURE_EXTRACTORS),
        help="Feature type to extract (single run).",
    )
    ml.add_argument(
        "--classifiers",
        nargs="+",
        default=list(ML_CLASSIFIERS),
        choices=list(ML_CLASSIFIERS),
        help="Classifiers for --all.",
    )
    ml.add_argument(
        "--feature-types",
        nargs="+",
        default=list(FEATURE_EXTRACTORS),
        choices=list(FEATURE_EXTRACTORS),
        help="Feature types for --all.",
    )
    ml.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use balanced sample weights.",
    )
    ml.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ml.set_defaults(run=_train_ml)

    baselines = train_actions.add_parser(
        "baselines", help="Run every DL baseline config (fit, then test)."
    )
    baselines.add_argument("--configs-dir", type=Path, default=CONFIGS_DIR)
    baselines.add_argument(
        "--config", type=Path, default=None, help="Run a single baseline config."
    )
    baselines.set_defaults(run=_train_baselines)

    cv = train_actions.add_parser("cv", help="K-fold cross-validation for one config.")
    cv.add_argument("config", type=Path)
    cv.add_argument(
        "--folds", type=int, default=None, help="Overrides k_fold in the config."
    )
    cv.set_defaults(run=_train_cv)

    grid = train_actions.add_parser("grid", help="Run a DL grid search.")
    grid.add_argument("grid_config", help="Grid search YAML.")
    grid.add_argument(
        "--base-config", default=None, help="Base model YAML (default: match name)."
    )
    grid.add_argument(
        "--fast-dev-run", action="store_true", help="One batch per stage."
    )
    grid.set_defaults(run=_train_grid)

    train_actions.add_parser("pretrain", help="Domain pretraining (forwarded).")

    # evaluate: aggregation, classical test evaluation and inference
    evaluate = commands.add_parser("evaluate", help="Aggregate, evaluate and predict.")
    evaluate.set_defaults(run=_planned("evaluate", "#721"))
    evaluate_actions = evaluate.add_subparsers(dest="evaluate_action", metavar="ACTION")
    aggregate = evaluate_actions.add_parser(
        "aggregate", help="Write the experiment summary CSV."
    )
    aggregate.add_argument(
        "--exp-names", nargs="+", required=True, help="Experiment log names."
    )
    aggregate.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "experiment_summary.csv",
        help="Output CSV path.",
    )
    aggregate.set_defaults(run=_evaluate_aggregate)

    evaluate_ml = evaluate_actions.add_parser(
        "ml", help="Test one trained classical model."
    )
    evaluate_ml.add_argument("--classifier", default="rf", choices=list(ML_CLASSIFIERS))
    evaluate_ml.add_argument(
        "--feature_type", default="crafted", choices=list(FEATURE_EXTRACTORS)
    )
    evaluate_ml.add_argument("--seed", type=int, default=DEFAULT_SEED)
    evaluate_ml.set_defaults(run=_evaluate_ml)

    evaluate_actions.add_parser("predict", help="Classify one image (forwarded).")

    # planned commands owned by open tickets
    commands.add_parser(
        "freeze", help="Freeze curated datasets and manifests (planned)."
    ).set_defaults(run=_planned("freeze", "#718"))
    commands.add_parser(
        "queue", help="Run the resumable training queue (planned)."
    ).set_defaults(run=_planned("queue", "#720"))
    commands.add_parser(
        "profile", help="Profile inference on the desktop (planned)."
    ).set_defaults(run=_planned("profile", "#721"))

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    head = tuple(tokens[:2])
    # These commands own their argument parsing, so their arguments bypass argparse.
    if head == ("train", "dl"):
        _train_dl(tokens[2:])
    elif head == ("train", "pretrain"):
        pretrain_main(tokens[2:])
    elif head == ("evaluate", "predict"):
        predict_ml(tokens[2:])
    else:
        args = build_parser().parse_args(tokens)
        args.run(args)


if __name__ == "__main__":
    main()
