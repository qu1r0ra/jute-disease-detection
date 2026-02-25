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
# # Classical ML Model Selection and Training (Local)
#
# In this notebook, we will focus exclusively on Phase 1 testing and selection for Classical Machine Learning:
# - Validate heavy ML Baselines (Random Forest, SVM, Logistic Regression, KNN, Gaussian Naive Bayes).
# - Compare models on Raw Pixels vs Handcrafted Features (Color Histograms, LBP, GLCM, HOG).
# - Execute runs using the `train_ml.py` routines.

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
from pathlib import Path

project_root = Path("../../").resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# %%
from jute_disease.utils.constants import DEFAULT_SEED
from jute_disease.utils.logger import get_logger
from jute_disease.utils.seed import seed_everything

# %%
logger = get_logger(__name__)
seed_everything(DEFAULT_SEED)

# %% [markdown]
# ## 1. Feature Extraction Pipeline
# Discussing the two types of feature extraction:
# - **Raw**: flattened pixels
# - **Crafted**: HOG, LBP, Color Moments
# We rely on `jute_disease.models.ml.features` to extract and cache these datasets.
#
# ## 2. Classical Model Baselines
# Training multiple algorithms to establish the absolute performance floor:
# - **Random Forest** (Tree-based ensemble)
# - **SVM** (Large margin classifier with RBF kernel)
# - **Logistic Regression** (Linear baseline)
# - **KNN** (Distance-based baseline)
# - **Gaussian Naive Bayes** (Probabilistic baseline)
#
# *Note: We use `StandardScaler` to normalize the inputs to handle differing scales of crafted features.*
#
# ## 3. Results and Model Selection
# Evaluating all models on the held-out validation set using the `run_all_ml.py` script and identifying our best performing baseline.
#
# %%

# %%
