import joblib
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def tune_model(model_type, X_train, y_train, X_test, y_test, project_name):
    # 1. Initialize W&B Run for this specific tuning session
    run = wandb.init(project=project_name, job_type="tuning", name=f"tune-{model_type}")

    # 2. Define the Grids you requested
    if model_type == "rf":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == "svm":
        # Note: probability=True is required for the predict_proba logic we used earlier
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }

    # 3. Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 4. Log Best Params and Final Metrics
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    wandb.log({
        "best_params": grid_search.best_params_,
        "test_f1_macro": grid_search.best_score_,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    })

    # 5. Save the Champion Artifact
    model_path = f"artifacts/ml_models/{model_type}_champion.joblib"
    joblib.dump(best_model, model_path)
    run.finish()
    
    return best_model