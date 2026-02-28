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
# # Model Evaluation

# %% [markdown]
# ## 1. Final Model Selection
# Identifying the definitively best models from our tuning phase, both from the Classical ML and Deep Learning tracks.
#
# ## 2. Evaluation on the Hold-Out Test Set
# We will rigorously evaluate the generalization performance of our best models on unseen test data using `EVAL_METRICS`.
# - **Metrics Breakdown**: Accuracy, Macro-F1, Precision, and Recall.
# - **Final Confusion Matrix**: To see the absolute true-positive vs false-positive rates across all Jute disease classes.
#
# ## 3. Classical ML vs. Deep Learning Comparison
# A formal comparison between our best handcrafted feature classical models (RF/SVM) against our Deep Learning neural network (MobileNetV2). We will compare:
# - **Performance**: Does DL significantly outperform ML?
# - **Efficiency / Cost**: Trade-offs between inference speed, interpretability, and feature extraction overhead.
#
# ## 4. Discussion and Conclusion
# Synthesizing our findings for the final IEEE conference paper.
# - Limitations of our best model.
# - Why certain disease types are inherently harder to classify.
# - Future work and practical applicability to real-world Jute farming.
