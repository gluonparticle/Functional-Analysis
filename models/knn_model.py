# models/knn_model.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier # Using Classifier version
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from plotting_utils import plot_classification_confusion_matrix
import time

MODEL_NAME = "k-Nearest Neighbors (k-NN)"
TASK_TYPE = "classification" # Primarily for classification

def train_evaluate_knn(X_train, y_train_class, X_val, y_val_class, X_test, y_test_class, n_neighbors=5):
    """
    Trains and evaluates a k-NN model.
    
    Args:
        X_train, y_train_class: Training data and classification targets.
        X_val, y_val_class: Validation data and classification targets.
        X_test, y_test_class: Test data and classification targets.
        n_neighbors (int): Number of neighbors for k-NN.

    Returns:
        dict: A dictionary containing model name, metrics, and plot base64 strings.
    """
    print(f"\n--- Training and Evaluating {MODEL_NAME} (k={n_neighbors}) ---")
    results = {
        "model_name": f"{MODEL_NAME} (k={n_neighbors})",
        "task_type": TASK_TYPE,
        "metrics": {},
        "plots": {}
    }

    # k-NN: n_neighbors is a key hyperparameter. Data should be scaled.
    # n_jobs=-1 uses all available cores for neighbor search.
    model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

    start_time = time.time()
    # k-NN's "fitting" is mostly just storing the training data.
    model.fit(X_train, y_train_class)
    training_time = time.time() - start_time # Will be very fast
    print(f"Training (data storing) completed in {training_time:.4f} seconds.")

    # --- Evaluation on Training Data ---
    # Prediction on train data can be slow for k-NN if train set is large
    # y_train_pred_labels = model.predict(X_train) 
    # y_train_pred_proba = model.predict_proba(X_train)
    # train_accuracy = accuracy_score(y_train_class, y_train_pred_labels)
    # train_logloss = log_loss(y_train_class, y_train_pred_proba)
    # For speed, we might skip detailed train metrics for k-NN in a large demo.
    results["metrics"]["Train Accuracy"] = "N/A (Skipped for k-NN speed)"
    results["metrics"]["Train LogLoss"] = "N/A (Skipped for k-NN speed)"


    # --- Evaluation on Validation Data ---
    y_val_pred_labels = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)
    val_accuracy = accuracy_score(y_val_class, y_val_pred_labels)
    val_logloss = log_loss(y_val_class, y_val_pred_proba)
    
    # --- Evaluation on Test Data ---
    start_time_inf = time.time()
    y_test_pred_labels = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)
    inference_time_test = time.time() - start_time_inf # Can be slow for large test sets / high dim
    print(f"Inference on test set completed in {inference_time_test:.2f} seconds.")

    test_accuracy = accuracy_score(y_test_class, y_test_pred_labels)
    test_logloss = log_loss(y_test_class, y_test_pred_proba)
    test_precision = precision_score(y_test_class, y_test_pred_labels, average='binary', zero_division=0)
    test_recall = recall_score(y_test_class, y_test_pred_labels, average='binary', zero_division=0)
    test_f1 = f1_score(y_test_class, y_test_pred_labels, average='binary', zero_division=0)
    
    results["metrics"].update({
        "Training Time (s)": f"{training_time:.4f}",
        "Inference Time (Test) (s)": f"{inference_time_test:.4f}",
        "Validation Accuracy": f"{val_accuracy:.4f}",
        "Validation LogLoss": f"{val_logloss:.4f}",
        "Test Accuracy": f"{test_accuracy:.4f}",
        "Test LogLoss (Loss)": f"{test_logloss:.4f}",
        "Test Precision": f"{test_precision:.4f}",
        "Test Recall": f"{test_recall:.4f}",
        "Test F1-score": f"{test_f1:.4f}",
        "RMSE": "N/A (Classification)"
    })
    print(f"Test Accuracy: {test_accuracy:.4f}, Test LogLoss: {test_logloss:.4f}")

    # --- Generate Plots ---
    class_names = [str(c) for c in sorted(y_test_class.unique())]
    results["plots"]["confusion_matrix_test"] = plot_classification_confusion_matrix(
        y_test_class, y_test_pred_labels, class_names=class_names,
        title=f"{MODEL_NAME} (k={n_neighbors}): Confusion Matrix (Test)"
    )

    return results

if __name__ == '__main__':
    print(f"{MODEL_NAME} model script loaded.")
    # Standalone test (requires dataset_generator.py in PYTHONPATH):
    # from dataset_generator import get_prepared_data_splits
    # X_train_base, X_val, X_test, \
    # y_train_base_r, y_val_r, y_test_r, \
    # y_train_base_c, y_val_c, y_test_c, \
    # features = get_prepared_data_splits()
    # # k-NN can be slow, so use a very small fraction for testing
    # X_train_sample = X_train_base.sample(frac=0.05, random_state=42) 
    # y_train_sample_c = y_train_base_c.loc[X_train_sample.index]
    # X_val_sample = X_val.sample(frac=0.1, random_state=42)
    # y_val_sample_c = y_val_c.loc[X_val_sample.index]
    # X_test_sample = X_test.sample(frac=0.1, random_state=42)
    # y_test_sample_c = y_test_c.loc[X_test_sample.index]

    # results = train_evaluate_knn(X_train_sample, y_train_sample_c, X_val_sample, y_val_sample_c, X_test_sample, y_test_sample_c, n_neighbors=7)
    # print("\nResults:")
    # for key, value in results["metrics"].items():
    #     print(f"  {key}: {value}")
    # print(f"  Plot 'confusion_matrix_test' generated: {'Yes' if results['plots'].get('confusion_matrix_test') else 'No'}")