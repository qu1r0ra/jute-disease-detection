import joblib
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from jute_disease.utils import WANDB_PROJECT, WANDB_ENTITY, get_logger

logger = get_logger("ML_Tuning")

# 1. Configuration
FEATURE_DIR = Path("/content/drive/MyDrive/artifacts/features/")
MODEL_DIR = Path("artifacts/ml_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def run_tuning():
    # 2. Load Data
    logger.info("Loading extracted features...")
    X_train = np.load(FEATURE_DIR / "craftedfeatureextractor_train_X.npy")
    y_train = np.load(FEATURE_DIR / "craftedfeatureextractor_train_y.npy")
    X_test = np.load(FEATURE_DIR / "craftedfeatureextractor_test_X.npy")
    y_test = np.load(FEATURE_DIR / "craftedfeatureextractor_test_y.npy")

    # 3. Define Configurations
    configs = [
        {
            "name": "rf",
            "search_type": "grid", # Exhaustive as requested
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'classifier__n_estimators': [100, 200, 500],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        {
            "name": "svm",
            "search_type": "random", # Quick search to save time
            "n_iter": 10, 
            "model": SVC(probability=True, random_state=42),
            "params": {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto', 0.1, 0.01]
            }
        }
    ]

    for config in configs:
        with wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT, 
            job_type="tuning", 
            name=f"tune-{config['name']}",
            config=config["params"] # Log search space to W&B
        ) as run:
            logger.info(f"Starting {config['search_type'].upper()} Search for {config['name'].upper()}...")
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', config['model'])
            ])

            # Select Search Strategy
            if config["search_type"] == "grid":
                search = GridSearchCV(
                    pipeline, config['params'], cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
                )
            else:
                search = RandomizedSearchCV(
                    pipeline, config['params'], n_iter=config["n_iter"], 
                    cv=5, scoring='f1_macro', n_jobs=-1, verbose=1, random_state=42
                )
            
            search.fit(X_train, y_train)

            # 4. Evaluation & Parity Metrics
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_probas = best_model.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # Export CV Results (Equivalent to DL metrics.csv)
            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(MODEL_DIR / f"{config['name']}_cv_results.csv", index=False)

            # 5. Advanced W&B Logging (Matching DL Visuals)
            wandb.log({
                "best_params": search.best_params_,
                "val_f1_macro": search.best_score_, # Equivalent to best val_loss row
                "test_acc": acc,
                "epoch": 0, # Placeholder for parity with DL charts
            })

            # Add Confusion Matrix and ROC/PR Curves to W&B
            class_names = ["Cercospora Leaf Spot", "General Damage", "Healthy", "Mosaic", "Stem Rot"]
            wandb.log({
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None, y_true=y_test, preds=y_pred, class_names=class_names
                ),
                "pr_curve": wandb.plot.pr_curve(y_test, y_probas, labels=class_names),
            })

            # 6. Save the Champion Pipeline
            save_path = MODEL_DIR / f"{config['name']}_crafted_champion.joblib"
            joblib.dump(best_model, save_path)
            
            artifact = wandb.Artifact(f"{config['name']}-champion", type="model")
            artifact.add_file(str(save_path))
            run.log_artifact(artifact)

            logger.info(f"Finished {config['name']}. Test Acc: {acc:.4f} | F1: {search.best_score_:.4f}")

if __name__ == "__main__":
    run_tuning()