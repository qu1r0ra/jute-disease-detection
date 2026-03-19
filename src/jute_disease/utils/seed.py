from lightning.pytorch import seed_everything as lightning_seed_everything

from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int, workers: bool = True) -> None:
    lightning_seed_everything(seed, workers=workers)
    logger.info(f"Random seed set to: {seed}")
