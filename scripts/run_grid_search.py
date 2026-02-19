import argparse
import subprocess

import yaml

from jute_disease.utils import get_logger

logger = get_logger(__name__)


def run_grid_search(config_path):
    """
    Reads grid search config and executes training runs.
    """
    logger.info(f"Reading grid config from {config_path}...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    transfer_levels = config["transfer_learning_levels"]
    dropout_rates = config["dropout_rates"]
    fixed_params = config["fixed_params"]

    for level in transfer_levels:
        level_name = level["name"]
        weights_path = level["weights_path"]

        for dropout in dropout_rates:
            logger.info(
                f"\n=== Running Experiment: {level_name} | Dropout: {dropout} ==="
            )

            ckpt_arg = weights_path if weights_path != "imagenet" else "null"
            pretrained_arg = str(weights_path == "imagenet")

            cmd = [
                "uv",
                "run",
                "python",
                "src/jute_disease/engines/dl/train.py",
                "fit",
                "--config",
                "configs/baselines/mobilevit.yaml",
                f"--model.feature_extractor.init_args.checkpoint_path={ckpt_arg}",
                f"--model.feature_extractor.init_args.pretrained={pretrained_arg}",
                f"--model.feature_extractor.init_args.drop_rate={dropout}",
                f"--trainer.logger.init_args.name=MobileViT_{level_name}_dr{dropout}",
                "--trainer.logger.init_args.group=MobileViT_Transfer_Grid",
                f"--model.lr={fixed_params.get('learning_rate', 0.001)}",
                f"--model.weight_decay={fixed_params.get('weight_decay', 0.01)}",
                f"--data.k_fold={fixed_params.get('num_folds', 1)}",
                f"--data.batch_size={fixed_params.get('batch_size', 32)}",
                f"--trainer.max_epochs={fixed_params.get('max_epochs', 100)}",
            ]

            logger.info(f"Command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running experiment {level_name}_{dropout}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to grid search yaml")
    args = parser.parse_args()

    run_grid_search(args.config)
