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
# # Final Model Evaluation
#
# In `notebooks/reproducibility/02_` and `03_`, we pursued twin objectives: one focused on handcrafted feature extraction with Classical ML, and the other leveraging transfer learning on Deep Learning architectures.
#
# Here, those paths converge. We will take the definitive champion from the ML track and the ultimate champion from the DL track, and pit them against each other on the unseen hold-out **Test Set**.

# %% [markdown]
# ## 1. Final Model Champions
#
# > Write your selection logic here for the DL model (MobileNetV2) and the ML model (Random Forest / SVM).
#
# **Deep Learning Champion:**
# - **Model:** MobileNetV2
# - **Specs:** Level 1 (ImageNet), 256x256, 0.1 Dropout
# - **Note:** As proven by our exhaustive learning rate grid search with extended early stopping patience, the model strictly caps at ~90% accuracy due to innate multi-label characteristics inside the dataset (i.e. leaves presenting multiple diseases concurrently while ground-truth labels force a single class).
#
# **Classical ML Champion:**
# - **Model:** ...
# - **Specs:** ...

# %% [markdown]
# ## 2. Evaluation on the Hold-Out Test Set
#
# > Write the data visualization logic to load your hold-out metrics or predictions here.

# %%
# Your loading logic here...

# %% [markdown]
# We evaluated the generalization performance of our best models on the unseen test data.
#
# **Metrics Breakdown**
#
# | Model | Test Accuracy | Macro-F1 | Precision | Recall |
# | :--- | :--- | :--- | :--- | :--- |
# | Deep Learning (MobileNetV2) | - | - | - | - |
# | Classical ML | - | - | - | - |
#

# %% [markdown]
# ### Confusion Matrices
#
# > Plot the side-by-side Confusion Matrices derived from Test set predictions.

# %%
# Confusion matrix plotting code here...

# %% [markdown]
# ## 3. Deep Learning vs. Classical ML: The Final Comparison
#
# > Summarize the head-to-head.
# - **Overall Performance**: Did DL significantly outperform ML, or was it surprisingly close?
# - **Cost & Efficiency**: What is the difference in latency (inference time)? Classical ML might be faster but required extensive upfront feature extraction overhead (Gabor filters, GLCM). DL operates directly on raw pixels but requires GPU acceleration for scale.
# - **Interpretability**: Classical models heavily rely on explicit texture boundaries. DL allows for spatial Grad-CAM heatmapping.

# %% [markdown]
# ## 4. Conclusion and Future Directives
#
# > Write your definitive paper conclusion.
#
# - **Bottlenecks:** Single-label constraint on multi-symptom leaves explicitly caps performance.
# - **Future Work:** Transitioning to a Multi-label Learning framework (e.g., using Binary Relevance or Sigmoid BCE) rather than standard Softmax Multi-class.
# - **Practical Application:** Can this be deployed cleanly to an edge advice via ONNX/TFLite running on a Jute farmer's smartphone?
