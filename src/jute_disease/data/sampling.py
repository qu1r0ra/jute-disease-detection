"""Deterministic loader helpers: balanced sampler, generator and worker seeding."""

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from jute_disease.config import DEFAULT_SEED


def loader_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Return a new generator seeded for one DataLoader.

    Workers derive their base seed from it, so shuffling and augmentation repeat
    for the same seed and differ for another.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy randomness inside a DataLoader worker.

    The worker's torch seed already differs per worker and per epoch; this makes
    the other libraries follow it.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def balanced_sampler(
    labels: Sequence[int], seed: int = DEFAULT_SEED
) -> WeightedRandomSampler:
    """Return a class-balanced sampler with its own seeded generator.

    Raises:
        ValueError: ``labels`` is empty.
    """
    targets = np.asarray(labels)
    if targets.size == 0:
        raise ValueError("Cannot build a sampler for an empty label list")
    unique, counts = np.unique(targets, return_counts=True)
    weight = dict(zip(unique.tolist(), (1.0 / counts).tolist(), strict=True))
    weights = torch.tensor([weight[t] for t in targets.tolist()], dtype=torch.double)
    return WeightedRandomSampler(
        weights, num_samples=len(weights), generator=loader_generator(seed)
    )
