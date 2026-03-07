# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Weighted Random Sampling
#
# Let us verify that `WeightedRandomSampler` correctly balances the class distribution in our training batches. Without sampling, the batches would reflect the natural imbalanced distribution of the dataset, potentially leading to skewed samples during training.

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
from pathlib import Path

project_root = Path("../..").resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from jute_disease.data.datamodule import DataModule
from jute_disease.utils.constants import DEFAULT_SEED
from jute_disease.utils.logger import get_logger
from jute_disease.utils.seed import seed_everything

logger = get_logger(__name__)
seed_everything(DEFAULT_SEED)

# %% [markdown]
# Let us instantiate two `DataModule` instances: one that does not use a weighted sampler and one that does.

# %%
dm_natural = DataModule(use_weighted_sampler=False)
dm_natural.prepare_data()
dm_natural.setup()

dm_weighted = DataModule(use_weighted_sampler=True)
dm_weighted.prepare_data()
dm_weighted.setup()

logger.info(f"Classes: {dm_weighted.classes}")


# %%
def collect_labels(dm, num_batches=50):
    loader = dm.train_dataloader()
    all_labels = []
    logger.info(f"Collecting {num_batches} batches...")
    for i, batch in tqdm(enumerate(loader), total=num_batches):
        if i >= num_batches:
            break
        _, labels = batch
        all_labels.extend(labels.tolist())
    return all_labels


logger.info("Collecting natural samples...")
natural_labels = collect_labels(dm_natural)

logger.info("Collecting weighted samples...")
weighted_labels = collect_labels(dm_weighted)

# %% [markdown]
# Let us visualize and compare the two distributions.

# %%
natural_names = [dm_natural.classes[i] for i in natural_labels]
weighted_names = [dm_weighted.classes[i] for i in weighted_labels]

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

sns.countplot(x=natural_names, ax=axes[0], order=dm_natural.classes)
axes[0].set_title("Natural Sampler Class Distribution")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

sns.countplot(x=weighted_names, ax=axes[1], order=dm_weighted.classes)
axes[1].set_title("Weighted Sampler Class Distribution")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()

# %% [markdown]
# From the histograms above, we can see that the weighted random sampler is more appropriate than the natural sampler for our case as it results in a more balanced distribution of classes from a sample.
