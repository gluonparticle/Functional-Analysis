# models/random_forest_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier # Using Classifier version
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, \
                            mean_squared_error, r2_score, mean_absolute_error # For potential regression use
from plotting_utils import plot_classification_confusion_matrix
import time

MODEL_NAME = "Random Forest"
# We'll primarily use it for classification as per typical setups,
# but it can do regression too. For this project, let's stick to classification.
TASK_TYPE = "classification"

def train_evaluate_random_forest(X_train, y_train_target, X_val, y_val_target, X_test, y_test_target,
                                 task_type=TASK_TYPE): # Allow overriding task_type if needed
    """
    Trains and evaluates a Random Forest model.
    Defaults to classification but can be switched to regression.
    
    Args:
        X_train, y_train_target: Training data and targets.
        X_val, y_val_target: Validation data and targets.
        X_test, y_test_target: Test data and targets.
        task_type (str): 'classification' or 'regression'.

    Returns:
        dict: A dictionary containing model name, metrics, and plot base64 strings.
    """
    print(f"\n--- Training and Evaluating {MODEL_NAME} ({task_type}) ---")
    results = {
        "model_name": f"{MODEL_NAME} ({task_type})",
        "task_type": task_type,
        "metrics": {},
        "plots": {}
    }

    if task_type == "classification":
        # n_estimators: number of trees, random_state for reproducibility
        # n_jobs=-1 uses all available cores
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    elif task_type == "regression":
        # from sklearn.ensemble import RandomForestRegressor # Import if using
        # model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        # For this project, we're focusing on the listed algorithms as classifiers where applicable.
        # If regression is truly needed, uncomment and import RandomForestRegressor.
        print(f"Warning: {MODEL_NAME} configured for regression, but primary setup is classification.")
        # For now, let's proceed with classifier for consistency with the list.
        # If you want RF for regression, change this part and ensure y_targets are regression values.
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        task_type = "classification" # Forcing classification for this demo's primary path
        results["task_type"] = "classification"
        results["model_name"] = f"{MODEL_NAME} (classification)"
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

    start_time = time.time()
    model.fit(X_train, y_train_target)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # --- Evaluation (common structure, specific metrics depend on task_type) ---
    # Train set
    y_train_pred_labels = model.predict(X_train)
    # Val set
    y_val_pred_labels = model.predict(X_val)
    # Test set
    start_time_inf = time.time()
    y_test_pred_labels = model.predict(X_test)
    inference_time_test = time.time() - start_time_inf

    results["metrics"]["Training Time (s)"] = f"{training_time:.4f}"
    results["metrics"]["Inference Time (Test) (s)"] = f"{inference_time_test:.4f}"

    if task_type == "classification":
        y_train_pred_proba = model.predict_proba(X_train)
        y_val_pred_proba = model.predict_proba(X_val)
        y_test_pred_proba = model.predict_proba(X_test)

        results["metrics"].update({
            "Train Accuracy": f"{accuracy_score(y_train_target, y_train_pred_labels):.4f}",
            "Train LogLoss": f"{log_loss(y_train_target, y_train_pred_proba):.4f}",
            "Validation Accuracy": f"{accuracy_score(y_val_target, y_val_pred_labels):.4f}",
            "Validation LogLoss": f"{log_loss(y_val_target, y_val_pred_proba):.4f}",
            "Test Accuracy": f"{accuracy_score(y_test_target, y_test_pred_labels):.4f}",
            "Test LogLoss (Loss)": f"{log_loss(y_test_target, y_test_pred_proba):.4f}",
            "Test Precision": f"{precision_score(y_test_target, y_test_pred_labels, average='binary', zero_division=0):.4f}",
            "Test Recall": f"{recall_score(y_test_target, y_test_pred_labels, average='binary', zero_division=0):.4f}",
            "Test F1-score": f"{f1_score(y_test_target, y_test_pred_labels, average='binary', zero_division=0):.4f}",
            "RMSE": "N/A (Classification)"
        })
        print(f"Test Accuracy: {results['metrics']['Test Accuracy']}, Test LogLoss: {results['metrics']['Test LogLoss (Loss)']}")
        
        class_names = [str(c) for c in sorted(y_test_target.unique())]
        results["plots"]["confusion_matrix_test"] = plot_classification_confusion_matrix(
            y_test_target, y_test_pred_labels, class_names=class_names,
            title=f"{MODEL_NAME}: Confusion Matrix (Test)"
        )
    
    # elif task_type == "regression": # If you implement RF Regressor
    #     results["metrics"].update({
    #         "Train RMSE": f"{np.sqrt(mean_squared_error(y_train_target, y_train_pred_labels)):.4f}",
    #         "Train R2": f"{r2_score(y_train_target, y_train_pred_labels):.4f}",
    #         "Validation RMSE": f"{np.sqrt(mean_squared_error(y_val_target, y_val_pred_labels)):.4f}",
    #         "Validation R2": f"{r2_score(y_val_target, y_val_pred_labels):.4f}",
    #         "Test RMSE": f"{np.sqrt(mean_squared_error(y_test_target, y_test_pred_labels)):.4f}",
    #         "Test R2": f"{r2_score(y_test_target, y_test_pred_labels):.4f}",
    #         "Test MAE": f"{mean_absolute_error(y_test_target, y_test_pred_labels):.4f}",
    #         "Test Loss (MSE)": f"{mean_squared_error(y_test_target, y_test_pred_labels):.4f}"
    #     })
    #     print(f"Test RMSE: {results['metrics']['Test RMSE']}, Test R2: {results['metrics']['Test R2']}")
        
    #     from plotting_utils import plot_regression_true_vs_predicted # Import if using
    #     results["plots"]["true_vs_predicted_test"] = plot_regression_true_vs_predicted(
    #         y_test_target, y_test_pred_labels, title=f"{MODEL_NAME}: True vs. Predicted (Test)"
    #     )

    return results

if __name__ == '__main__':
    print(f"{MODEL_NAME} model script loaded.")
    # Standalone test (requires dataset_generator.py in PYTHONPATH):
    # from dataset_generator import get_prepared_data_splits
    # X_train_base, X_val, X_test, \
    # y_train_base_r, y_val_r, y_test_r, \
    # y_train_base_c, y_val_c, y_test_c, \
    # features = get_prepared_data_splits()
    # X_train_sample = X_train_base.sample(frac=0.1, random_state=42) # Use a small fraction for speed
    # y_train_sample_c = y_train_base_c.loc[X_train_sample.index]
    # # Test classification
    # results_clf = train_evaluate_random_forest(X_train_sample, y_train_sample_c, X_val, y_val_c, X_test, y_test_c, task_type="classification")
    # print("\nClassification Results:")
    # for key, value in results_clf["metrics"].items():
    #     print(f"  {key}: {value}")
    # print(f"  Plot 'confusion_matrix_test' generated: {'Yes' if results_clf['plots'].get('confusion_matrix_test') else 'No'}")