# %% [markdown]
# <a href="https://colab.research.google.com/github/qu1r0ra/jute-related-image-evaluation/blob/main/notebooks/reproducibility/04_Model_Evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Final Model Evaluation
#
# In `notebooks/reproducibility/02_` and `03_`, we pursued twin objectives: one focused on handcrafted feature extraction with Classical ML, and the other leveraging transfer learning on Deep Learning architectures.
#
# Here, those paths converge. We will take the definitive champion from the ML track and the ultimate champion from the DL track, and pit them against each other on the unseen hold-out **Test Set**.

# %% [markdown]
# ## Environment Setup
#

# %%
# !git clone https://github.com/qu1r0ra/jute-related-image-evaluation.git
# %cd jute-related-image-evaluation


# %pip install uv
# !uv pip install --system -e .
# !uv sync

# %%
# ruff: noqa: T201
from jute_disease.utils.constants import DEFAULT_SEED
from jute_disease.utils.seed import seed_everything

seed_everything(DEFAULT_SEED)

# %%
from google.colab import drive

drive.mount("/content/drive")

# %%
# %cd jute-related-image-evaluation/

# %%
# make sure that the current directory is as
from pathlib import Path

# Update DATA_ZIP_PATH to where data.zip is stored relative to the Colab VM filesystem.
# For organization, we stored ours in!uv pip install --system -e .
# "/content/drive/MyDrive/Colab Notebooks/Jute Leaf Disease/data.zip"
DATA_ZIP_PATH = "/content/drive/MyDrive/data.zip"
DEST_PATH = Path("data/by_class")

if Path(DATA_ZIP_PATH).exists():
    DEST_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {DATA_ZIP_PATH} to {DEST_PATH}...")
    # !unzip -q -n "$DATA_ZIP_PATH" -d "$DEST_PATH"
    print("Data unpacked.")
else:
    print(
        f"Zip file not found at {DATA_ZIP_PATH}. "
        "Please check the path or upload your data."
    )

# %%
# !uv run python src/jute_disease/data/utils.py split

# %% [markdown]
# ## 1. Final Model Champions
# **Deep Learning Champion:**
# - **Model:** MobileNetV2
# - **Specs:** Level 1 (ImageNet), 256x256, 0.1 Dropout
# - **Note:** As proven by our exhaustive learning rate grid search with extended early stopping patience, the model strictly caps at ~90% accuracy due to innate multi-label characteristics inside the dataset (i.e. leaves presenting multiple diseases concurrently while ground-truth labels force a single class).
#
# **Classical ML Champion:**
# - **Model:** Random Forest - Crafted
# - **Specs:**
#   - maximum depth: None
#   - maximum features: sqrt
#   - minimum sample split: 2
#   - n estimator: 200
# - **Note:** Our grid search tuning has led our model to an accuracy of ~72% wherein our model generally seem to struggle in fully utilizing the crafted features as demonstrated by how our data is distributed in our latent space analsysis which led us to trying deep learning alternatives.
#

# %% [markdown]
# ## 2. Evaluation on the Hold-Out Test Set
#
#
#

# %%
import pandas as pd
import wandb

api = wandb.Api()

ENTITY = "grade-descent"
PROJECT = "jute-related-image-evaluation"

comparison_runs = {"Random Forest": "w6593daw", "MobileNet V2": "af60c7b9"}

results = []

for model_name, run_id in comparison_runs.items():
    path = f"{ENTITY}/{PROJECT}/{run_id}"
    try:
        run = api.run(path)
        s = run.summary

        metrics = {
            "Model": model_name,
            "Accuracy": s.get("test_acc"),
            "Macro F1": s.get("test_f1"),
        }
        results.append(metrics)
        print(f"Successfully fetched metrics for {model_name}")

    except Exception as e:
        print(f"Error fetching {run_id}: {e}")

df = pd.DataFrame(results)

styled_df = df.style.format(
    {
        "Accuracy": "{:.4f}",
        "Macro F1": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
    }
).hide(axis="index")

display(styled_df)

# %% [markdown]
# Based on the classification metrics presented above, we can see that our MobileNet V2 model has significantly outperformed our random forest model across all metrics. Based from our understanding of the feature importance and error analysis from the classical machine learning notebooks, this may be attributed as to how the deep learning approach was not constrained by the crafted features defined. To get a better understanding and intuition on this subject we proceed to the confusion matrices and the by class metrics.

# %% [markdown]
# ### Confusion Matrices
#

# %% [markdown]
# Run the following code to setup for the confusion matrices.

# %%
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

api = wandb.Api()
ENTITY = "grade-descent"
PROJECT = "jute-related-image-evaluation"

rf_model_path = f"{ENTITY}/{PROJECT}/rf-champion:latest"
dl_artifact_path = f"{ENTITY}/{PROJECT}/run-af60c7b9-test_conf_mat_table:latest"
jute_classes = [
    "Cercospora",
    "General Damage",
    "Dieback",
    "Healthy",
    "Mosaic",
    "Stem Rot",
]


def get_rf_cm_from_joblib():
    artifact = api.artifact(rf_model_path)
    model_dir = artifact.download()
    model = joblib.load(Path(model_dir) / "rf_crafted_champion.joblib")

    X_test = np.load(
        "/content/drive/MyDrive/artifacts/features/craftedfeatureextractor_test_X.npy"
    )
    y_test = np.load(
        "/content/drive/MyDrive/artifacts/features/craftedfeatureextractor_test_y.npy"
    )

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(cm, index=jute_classes, columns=jute_classes), y_test


def get_dl_cm_from_json(artifact_path):
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    json_file = next(Path(artifact_dir).glob("*.json"))
    with open(json_file) as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data["data"], columns=raw_data["columns"])
    df["Actual"] = df["Actual"].replace({"Cercospora Leaf Spot": "Cercospora"})
    df["Predicted"] = df["Predicted"].replace({"Cercospora Leaf Spot": "Cercospora"})

    val_col = "n" if "n" in df.columns else "nPredictions"
    pivot = df.groupby(["Actual", "Predicted"])[val_col].sum().unstack().fillna(0)
    return pivot.reindex(index=jute_classes, columns=jute_classes, fill_value=0)


def get_per_class_metrics(cm_pivot):
    metrics_list = []
    total_samples = cm_pivot.values.sum()
    classes = cm_pivot.index.tolist()

    for cls in classes:
        # Binary Classification components for the specific class
        tp = cm_pivot.loc[cls, cls]
        fp = cm_pivot[cls].sum() - tp
        fn = cm_pivot.loc[cls].sum() - tp
        tn = total_samples - (tp + fp + fn)

        # Performance Formulas
        accuracy = (tp + tn) / total_samples
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics_list.append(
            {
                "Class": cls,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Support": int(tp + fn),
            }
        )

    return pd.DataFrame(metrics_list)


# %%
cm_rf, y_test_rf_raw = get_rf_cm_from_joblib()
cm_dl = get_dl_cm_from_json(dl_artifact_path)

labels_rf = [
    f"{cls}\n(n={int(count)})"
    for cls, count in zip(jute_classes, cm_rf.sum(axis=1), strict=True)
]
labels_dl = [
    f"{cls}\n(n={int(count)})"
    for cls, count in zip(jute_classes, cm_dl.sum(axis=1), strict=True)
]

# 4. Plotting
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="g",
    cmap="Blues",
    ax=axes[0],
    xticklabels=jute_classes,
    yticklabels=labels_rf,
    cbar=False,
)
axes[0].set_title(
    "Classical ML (Random Forest)\nCorrect vs. Confused Counts",
    fontsize=14,
    pad=20,
    fontweight="bold",
)

sns.heatmap(
    cm_dl,
    annot=True,
    fmt="g",
    cmap="Oranges",
    ax=axes[1],
    xticklabels=jute_classes,
    yticklabels=labels_dl,
    cbar=False,
)
axes[1].set_title(
    "Deep Learning (MobileNet V2)\nCorrect vs. Confused Counts",
    fontsize=14,
    pad=20,
    fontweight="bold",
)

plt.tight_layout()
plt.show()


df_rf_final = get_per_class_metrics(cm_rf)
df_dl_final = get_per_class_metrics(cm_dl)

print("\n Per-Class Performance - Champion Random Forest")
styled_rf = df_rf_final.style.format(
    {
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
    }
).hide(axis="index")
display(styled_rf)

print("\n Per-Class Performance - MobileNet V2")
styled_dl = df_dl_final.style.format(
    {
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
    }
).hide(axis="index")
display(styled_dl)

# %% [markdown]
# We first look at the confusion matrix comparison in the figures above. At a glance, it is clear that the MobileNet V2 model has significantly reduced its errors in misclassifying the test set and has improved the correct predictions as shown in the diagonal of the confusion matrix. However we can see that `Mosaic` instances being misclassified as `Cercospora Leaf Spots` increased while `Cercospora Leaf Spots` being misclassified as `Mosaic` decreased by a noticeable amount. This suggest that our shift in attempting a deep learning alternative approach has somewhat addressed our models problem in interchanging these two classes.
#
# Diving deeper into the comparison, we can see that the in the classification metrics for the six classes, the MobileNet V2 outperformed the tuned SVM model across accuracy and F1-score for all classes. This further reinforces the strength that our deep learning approach had on our previous classical machine learning approach. With regards the `Cercospora` and `Mosaic` dynamic, we can see that precision increased for `Cercospora` and `Mosaic`. Although, recall increased for `Cercospora`, `Mosaic` apparently decreased. This suggests that our model MobileNet V2 model is still struggling in correctly identifying `Mosaic` images to some degree. From our understanding, we believe that this may be due to how the actual `Mosaic` instances share characteristics with `Cercospora` which may or may not be due to the possibility of having both deseases be present at the same time.  

# %% [markdown]
# ## 3. Deep Learning vs. Classical ML: The Final Comparison
#

# %% [markdown]
# ### 3A: Overall Performance
# The DL model was able to noticeably outperform the ML model across all four classification metrics (accuracy, precision, recall, and F1-Macro). In addition to this, the confusion matrices show a noticeable improvement in correctly classifying between classes. As mentioned earlier, the models ability to correctly distinguish between `Cercospora Leaf Spot` and `Mosaic` still prevails with the possibility being that maybe both diseases are present in the dataset used.

# %% [markdown]
# ### 3B: Cost & Efficiency
# In comparing the cost & efficiency, we use Python's time library in measuring the latency. Specifically we do the following:
# 1. Setup models from WANDB runs.
# 2. Generate a random input that matches the required input requirementss.
# 3. Run the inference setup for a hundred iterations and measure average time.
#
# Run the following code cells:

# %%
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import transforms

from jute_disease.models.dl.backbone import TimmBackbone
from jute_disease.models.dl.classifier import Classifier
from jute_disease.models.ml.features import CraftedFeatureExtractor
from jute_disease.utils.constants import IMAGE_SIZE

api = wandb.Api()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_PATH = "grade-descent/jute-related-image-evaluation"
TEST_DATA_DIR = Path("data/ml_split/test")


# RF
rf_artifact = api.artifact(f"{PROJECT_PATH}/rf-champion:latest")
rf_dir = rf_artifact.download()
rf_pipeline = joblib.load(Path(rf_dir) / "rf_crafted_champion.joblib")
handcrafted_extractor = CraftedFeatureExtractor()

# MobileNet
mobile_artifact = api.artifact(f"{PROJECT_PATH}/model-af60c7b9:latest")
mobile_dir = mobile_artifact.download()
ft_ckpt_path = next(Path(mobile_dir).glob("*.ckpt"))

backbone = TimmBackbone(
    model_name="mobilenetv2_100", pretrained=False, checkpoint_path=str(ft_ckpt_path)
)
ft_model = Classifier.load_from_checkpoint(
    ft_ckpt_path, feature_extractor=backbone, num_classes=6
)
ft_model.to(device).eval()

valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
test_image_paths = [p for p in TEST_DATA_DIR.rglob("*") if p.suffix in valid_extensions]

if not test_image_paths:
    raise FileNotFoundError(f"No images found in {TEST_DATA_DIR}")

TARGET_SIZE = (IMAGE_SIZE, IMAGE_SIZE)

dl_transforms = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

rf_e2e_times = []
dl_e2e_times = []

print(f"Benchmarking {len(test_image_paths)} images across class folders...")

with torch.no_grad():
    for i, img_path in enumerate(test_image_paths):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(test_image_paths)}")

        try:
            raw_pil = Image.open(img_path).convert("RGB")

            # RF Pipeline
            t0_rf = time.perf_counter()
            ml_img = raw_pil.resize(TARGET_SIZE)
            feat_vec = handcrafted_extractor.extract(ml_img)
            _ = rf_pipeline.predict(feat_vec.reshape(1, -1))
            rf_e2e_times.append(time.perf_counter() - t0_rf)

            # MobileNet Pipeline
            t0_dl = time.perf_counter()
            x = dl_transforms(raw_pil).unsqueeze(0).to(device)
            # Forward Pass
            logits = ft_model(x)
            _ = F.softmax(logits, dim=1)
            dl_e2e_times.append(time.perf_counter() - t0_dl)

        except Exception as e:
            print(f"Error on {img_path.name}: {e}")

# results
print(f"\n{'Architecture':<20} | {'Avg E2E Latency (ms)':<25} | {'Std Dev':<10}")
print("-" * 65)
print(
    f"{'SVM (Handcrafted)':<20} | {np.mean(rf_e2e_times) * 1000:>12.2f} ms | {np.std(rf_e2e_times) * 1000:>8.2f}"
)
print(
    f"{'MobileNet V2':<20} | {np.mean(dl_e2e_times) * 1000:>12.2f} ms | {np.std(dl_e2e_times) * 1000:>8.2f}"
)

# %% [markdown]
# The results of our runtime experiment showed us that the MobileNet V2 model ran with less latency on average than the Support Vector Machine with crafted features. Prior to this experiment, we hypothesized that the Support Vector Machine model would have less latency given that it is a much simpler model than MobileNet V2. However, we realized that the feature extraction process may have acted as the bottleneck given that the model had to sequentially compute for the input image's HSV, LPB, GLCM, and HOG while the MobileNet V2's pipeline was composed of highly optimized matrix multiplications performed on the available GPU.

# %% [markdown]
# ## 3C: Interpretability
# Our two models used two distinct approaches with the tuned Support Vector Machine  using the crafted features (HSV, LBP, GLCM, and HOG) extracted from the images while our MobileNet V2 used a transfer learning approahch.
#
# As dicsussed in the previous `Notebook 3` for classical machine learning model tuning and error analysis, the feature importance section describe how our Support Vector Machine model primarily relied on Histogram of Oriented Gradients (HOG) which capture the structure and shapes present in our image.
#
# However, two particular color channels stood out as they were identified to be among the top five most important features which are the `STD of Green Color Channel` and `Mean of Red Color Channel`. We hypothesize that these two color channels acted as the main drivers in distinguishing between classes that are characterized by an attribute of color. Such characteristics are as follows:
#
# - `Cercospora Leaf Spots` - brown lesions
# - `Mosaic` - yellow patches
# - `Stem Rot` - dark brown lesions on stems
#

# %%
from IPython.display import Image, display

display(Image("assets/figures/ml/random_forest_top_20.png"))

# %% [markdown]
# On the other hand, the deep learning notebooks allowed us to understand how the model views the images fed into it. The GRAD-CAM heatmaps shows that for most instances, the model focuses on the central leaf or the leaf we are subjecting to be classified. However, there are rare occasions in which the model focuses on the background of the images instead, which appear to showcase symptoms of a disease thus affecting the models' prediction.
#
# Additionally, we also noticed that for some images wherein the the model was confidently incorrect, it was focusing on features that are do not resemble the characteristic of the actual label. For example, a `Mosaic` leaf being classifed as a `Cercospora Leaf Spot` due to having what appears to be small brown lesions. This highlights the key limitation of our model being a single-label classification task that doesn't consider the possibility of having multiple diseases present for a single instance.
#
#

# %%
display(Image("assets/figures/dl/grad_cam.png"))

# %% [markdown]
# ## 4. Conclusion and Future Directives
# Our work on jute disease image classification using classical machine learning and deep learning has allowed us to explore different strategies in creating a model that can correctly classify `Cercospora Leaf Spots`, `Dieback`, `Healthy`, `General Damage`, `Mosaic`, and `Stem Rot` in jute leaves. Our training, error analysis, and fine tuning process has determined that our MobileNet V2 model stood out among the different models utilized. The error analysis provided us with the insights that may improve our models' performance. First, we theorize that the images may have preprocessing given that our Grad-Cam Heatmap figures show that there are instances where the deep learning model primarily focuses on non-relevant subjects (i.e., leaves present in the background). Next, we observed that both learning approaches particularly struggled in classifying `Cercospora Leaf Spots` and `Mosaic`. Upon investigation of such mistaken images, we observed that there is a possibility that the leaves may be considered as having multiple symptoms of the diseases which is a direct limitation of our project as we implemented a single-label constraint. Nonetheless, this has provided us with the means on how to improve the current state of the models such as transitioning to a multi-label learning framework rather than a standard Softmax layer. Lastly, in consideration of model deployment, we are confident that the model may be integrated into edge devices such as smartphones so that Jute farmer's may utilize it to quickly identify such diseases and plan accrodingly.
