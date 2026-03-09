import joblib
import wandb
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. Configuration
PROJECT_NAME = "jute-disease-detection"
FEATURE_DIR = Path("artifacts/features")
MODEL_DIR = Path("artifacts/ml_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def run_tuning():
    # 2. Load Data
    logger.info("Loading extracted features...")
    X_train = np.load(FEATURE_DIR / "craftedfeatureextractor_train_X.npy")
    y_train = np.load(FEATURE_DIR / "craftedfeatureextractor_train_y.npy")
    X_test = np.load(FEATURE_DIR / "craftedfeatureextractor_test_X.npy")
    y_test = np.load(FEATURE_DIR / "craftedfeatureextractor_test_y.npy")

    # 3. Define Grids
    configs = [
        {
            "name": "rf",
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'classifier__n_estimators': [100, 200, 500],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        {
            "name": "svm",
            "model": SVC(probability=True, random_state=42),
            "params": {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto', 0.1, 0.01]
            }
        }
    ]

    for config in configs:
        with wandb.init(project=PROJECT_NAME, job_type="tuning", name=f"tune-{config['name']}") as run:
            logger.info(f"🚀 Tuning {config['name'].upper()}...")
            
            # Create a Pipeline (Scaler + Model) 
            # This prevents the 'feature mismatch' error we saw earlier
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', config['model'])
            ])

            grid_search = GridSearchCV(
                pipeline, 
                config['params'], 
                cv=5, 
                scoring='f1_macro', 
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)

            # 4. Evaluation
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # 5. Log to W&B
            wandb.log({
                "best_params": grid_search.best_params_,
                "best_val_f1_macro": grid_search.best_score_,
                "test_accuracy": acc,
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            })

            # 6. Save the Champion Pipeline
            save_path = MODEL_DIR / f"{config['name']}_crafted_champion.joblib"
            joblib.dump(best_model, save_path)
            
            # Upload model to W&B Artifacts
            artifact = wandb.Artifact(f"{config['name']}-champion", type="model")
            artifact.add_file(str(save_path))
            run.log_artifact(artifact)

            logger.info(f"✅ Finished {config['name']}. Best Score: {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    from jute_disease.utils import get_logger
    logger = get_logger("ML_Tuning")
    run_tuning()