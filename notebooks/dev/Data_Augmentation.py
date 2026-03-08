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
# # Data Augmentation
#
# Let us visualize the effect of the augmentation pipeline.

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
from pathlib import Path

# %% [markdown]
# We need access to scripts in `src/`, so let us add append its path.

# %%
project_root = Path("../..").resolve(strict=True)
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# %%
from jute_disease.data.datamodule import DataModule
from jute_disease.utils.visualization import visualize_augmentations

# %% [markdown]
# Let us initialize a `DataModule` instance and manually trigger its setup.
#
# Normally, we do not need to explicitly invoke `LightningDataModule` functions as they are hooks that automatically trigger at different points during the model training process, but here we shall for demonstration purposes.

# %%
dm = DataModule()
dm.setup()

# %%
visualize_augmentations(dm.jute_train)

# %% [markdown]
# Each row in the grid visualizes a sample image from the train set in the first column and various augmentations applied to it in the following columns.
