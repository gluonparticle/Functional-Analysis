# models/naive_bayes_model.py
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from plotting_utils import plot_classification_confusion_matrix
import time

MODEL_NAME = "Naive Bayes (Gaussian)"
TASK_TYPE = "classification" # GaussianNB is for classification

def train_evaluate_naive_bayes(X_train, y_train_class, X_val, y_val_class, X_test, y_test_class):
    """
    Trains and evaluates a Gaussian Naive Bayes model.
    
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

    model = GaussianNB()

    start_time = time.time()
    model.fit(X_train, y_train_class)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # --- Evaluation on Training Data ---
    y_train_pred_labels = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)
    train_accuracy = accuracy_score(y_train_class, y_train_pred_labels)
    train_logloss = log_loss(y_train_class, y_train_pred_proba)

    # --- Evaluation on Validation Data ---
    y_val_pred_labels = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)
    val_accuracy = accuracy_score(y_val_class, y_val_pred_labels)
    val_logloss = log_loss(y_val_class, y_val_pred_proba)
    
    # --- Evaluation on Test Data ---
    start_time_inf = time.time()
    y_test_pred_labels = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)
    inference_time_test = time.time() - start_time_inf

    test_accuracy = accuracy_score(y_test_class, y_test_pred_labels)
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
        "Test LogLoss (Loss)": f"{test_logloss:.4f}",
        "Test Precision": f"{test_precision:.4f}",
        "Test Recall": f"{test_recall:.4f}",
        "Test F1-score": f"{test_f1:.4f}",
        "RMSE": "N/A (Classification)"
    }
    print(f"Test Accuracy: {test_accuracy:.4f}, Test LogLoss: {test_logloss:.4f}")

    # --- Generate Plots ---
    class_names = [str(c) for c in sorted(y_test_class.unique())]
    results["plots"]["confusion_matrix_test"] = plot_classification_confusion_matrix(
        y_test_class, y_test_pred_labels, class_names=class_names,
        title=f"{MODEL_NAME}: Confusion Matrix (Test)"
    )

    return results

if __name__ == '__main__':
    print(f"{MODEL_NAME} model script loaded.")