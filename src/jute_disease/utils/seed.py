from lightning.pytorch import seed_everything as lightning_seed_everything

from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int, workers: bool = True) -> None:
    """
    Sets the seed for generating random numbers in PyTorch, numpy, and Python's random.
    Using Lightning's implementation for robustness across workers and CUDA.
    """
    lightning_seed_everything(seed, workers=workers)
    logger.info(f"Random seed set to: {seed}")
