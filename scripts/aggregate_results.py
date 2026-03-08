import argparse
from pathlib import Path

import pandas as pd

from jute_disease.utils import get_logger

logger = get_logger(__name__)


def aggregate_metrics(exp_names: list[str], output_csv: Path) -> None:
    """Consolidates metrics from all CSVLogger outputs into a master CSV."""
    results = []

    for exp in exp_names:
        log_dir = Path("artifacts/logs") / exp
        if not log_dir.exists():
            continue

        metrics_files = list(log_dir.glob("version_*/metrics.csv"))
        if not metrics_files:
            continue

        dfs = [pd.read_csv(f) for f in metrics_files]
        df = pd.concat(dfs, ignore_index=True)

        summary = {"Experiment": exp}

        if "val_loss" in df.columns:
            val_df = df.dropna(subset=["val_loss"])
            if not val_df.empty:
                best_row = val_df.loc[val_df["val_loss"].idxmin()].to_dict()
                for key, val in best_row.items():
                    if key.startswith(("val_", "train_")) or key in ["epoch", "step"]:
                        summary[key] = val

        for col in ["train_acc", "train_loss"]:
            if col in df.columns and (col not in summary or pd.isna(summary[col])):
                summary[col] = df[col].max() if "acc" in col else df[col].min()

        if "test_loss" in df.columns:
            test_df = df.dropna(subset=["test_loss"])
            if not test_df.empty:
                test_metrics = test_df.iloc[-1].to_dict()
                for key, val in test_metrics.items():
                    if key.startswith("test_"):
                        summary[key] = val

        results.append(summary)

    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_csv, index=False)
        logger.info(f"Summary metrics exported to {output_csv}")
    else:
        logger.warning(f"No metrics found to export to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics.")
    parser.add_argument(
        "--exp-names",
        nargs="+",
        required=True,
        help="Names of the experiments to aggregate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiment_summary.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    aggregate_metrics(args.exp_names, args.output)


if __name__ == "__main__":
    main()
