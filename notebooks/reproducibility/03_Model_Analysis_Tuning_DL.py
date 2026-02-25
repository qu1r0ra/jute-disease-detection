# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Model Analysis and Tuning

# %% [markdown]
# ## 1. Introduction and Baseline Review
# In the previous notebook, we trained a variety of baseline classical machine learning models. The top two performers were the **Random Forest** and **Support Vector Machine (SVM)** trained on handcrafted features (color, LBP, and HOG) with baseline standard scaling applied.
#
# ## 2. Error Analysis on Initial Models
# Before blindly tuning hyperparameters, it is critical to understand *why* the models make mistakes.
# - **Confusion Matrices**: Which disease classes are most often confused?
# - **Misclassified Samples**: Visually inspecting images where the models failed. Is it poor lighting? Ambiguous symptoms?
# - **Feature Importance**: Using Random Forest's `feature_importances_` to see which crafted features (e.g., LBP vs. HOG) are driving the predictions.
#
# ## 3. Hyperparameter Tuning (Grid Search)
# To squeeze out maximum performance and mitigate overfitting, we will tune the hyperparameters for both models using Grid Search with Cross-Validation.
# - **Random Forest Grid**: `n_estimators`, `max_depth`, `min_samples_split`
# - **SVM Grid**: `C` (regularization), `kernel`, `gamma`
#
# ## 4. Deep Learning Analysis (Placeholder)
# *(This section will be populated in the next phase once our PyTorch Lightning deep learning models are trained and ready for analysis).*
