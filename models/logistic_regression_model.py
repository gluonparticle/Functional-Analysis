# models/logistic_regression_model.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from plotting_utils import plot_classification_confusion_matrix
import time

MODEL_NAME = "Logistic Regression"
TASK_TYPE = "classification" # This model is for classification

def train_evaluate_logistic_regression(X_train, y_train_class, X_val, y_val_class, X_test, y_test_class):
    """
    Trains and evaluates a Logistic Regression model.
    
    Args:
        X_train, y_train_class: Training data and classification targets.
        X_val, y_val_class: Validation data and classification targets.
        X_test, y_test_class: Test data and classification targets.

    Returns:
        dict: A dictionary containing model name, metrics, and plot base64 strings.
    """
    print(f"\n--- Training and Evaluating {MODEL_NAME} ---")
    results = {
        "model_name": MODEL_NAME,
        "task_type": TASK_TYPE,
        "metrics": {},
        "plots": {}
    }

    # Logistic Regression in scikit-learn can use 'saga' solver for L1/L2 with large datasets
    # and it supports multi_class='multinomial' for true multinomial LR.
    # For binary, 'liblinear' is often a good default. max_iter might need adjustment.
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=200) # Increased max_iter

    start_time = time.time()
    model.fit(X_train, y_train_class)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # --- Evaluation on Training Data ---
    y_train_pred_labels = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1] # Prob for positive class
    train_accuracy = accuracy_score(y_train_class, y_train_pred_labels)
    train_logloss = log_loss(y_train_class, y_train_pred_proba)

    # --- Evaluation on Validation Data ---
    y_val_pred_labels = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_accuracy = accuracy_score(y_val_class, y_val_pred_labels)
    val_logloss = log_loss(y_val_class, y_val_pred_proba)

    # --- Evaluation on Test Data ---
    start_time = time.time()
    y_test_pred_labels = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test) # Get all probabilities for log_loss
    inference_time_test = time.time() - start_time

    test_accuracy = accuracy_score(y_test_class, y_test_pred_labels)
    # For binary classification, predict_proba returns [prob_class_0, prob_class_1]
    # log_loss expects probabilities for each class if y_true has multiple classes,
    # or just prob_class_1 if y_true is binary {0,1}.
    # If y_test_class.nunique() > 2, use y_test_pred_proba directly.
    # For binary, y_test_pred_proba[:, 1] is also fine, but using the full y_test_pred_proba is more general.
    test_logloss = log_loss(y_test_class, y_test_pred_proba)
    test_precision = precision_score(y_test_class, y_test_pred_labels, average='binary', zero_division=0)
    test_recall = recall_score(y_test_class, y_test_pred_labels, average='binary', zero_division=0)
    test_f1 = f1_score(y_test_class, y_test_pred_labels, average='binary', zero_division=0)
    
    results["metrics"] = {
        "Training Time (s)": f"{training_time:.4f}",
        "Inference Time (Test) (s)": f"{inference_time_test:.4f}",
        "Train Accuracy": f"{train_accuracy:.4f}",
        "Train LogLoss": f"{train_logloss:.4f}",
        "Validation Accuracy": f"{val_accuracy:.4f}",
        "Validation LogLoss": f"{val_logloss:.4f}",
        "Test Accuracy": f"{test_accuracy:.4f}",
        "Test LogLoss (Loss)": f"{test_logloss:.4f}", # LogLoss is the typical loss for Logistic Regression
        "Test Precision": f"{test_precision:.4f}",
        "Test Recall": f"{test_recall:.4f}",
        "Test F1-score": f"{test_f1:.4f}",
        "RMSE": "N/A (Classification)" # RMSE not applicable for classification
    }
    print(f"Test Accuracy: {test_accuracy:.4f}, Test LogLoss: {test_logloss:.4f}")

    # --- Generate Plots ---
    # For classification, "datapoints vs fit type" is typically a Confusion Matrix
    class_names = [str(c) for c in sorted(y_test_class.unique())]
    results["plots"]["confusion_matrix_test"] = plot_classification_confusion_matrix(
        y_test_class, y_test_pred_labels, class_names=class_names,
        title=f"{MODEL_NAME}: Confusion Matrix (Test)"
    )
    # No loss curve per epoch for scikit-learn's Logistic Regression

    return results

if __name__ == '__main__':
    print(f"{MODEL_NAME} model script loaded.")
    # Standalone test (requires dataset_generator.py in PYTHONPATH):
    # from dataset_generator import get_prepared_data_splits
    # X_train_base, X_val, X_test, \
    # y_train_base_r, y_val_r, y_test_r, \
    # y_train_base_c, y_val_c, y_test_c, \
    # features = get_prepared_data_splits()
    # X_train_sample = X_train_base.sample(frac=0.1, random_state=42)
    # y_train_sample_c = y_train_base_c.loc[X_train_sample.index]
    # results = train_evaluate_logistic_regression(X_train_sample, y_train_sample_c, X_val, y_val_c, X_test, y_test_c)
    # print("\nResults:")
    # for key, value in results["metrics"].items():
    #     print(f"  {key}: {value}")
    # print(f"  Plot 'confusion_matrix_test' generated: {'Yes' if results['plots'].get('confusion_matrix_test') else 'No'}")