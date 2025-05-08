# models/mlp_model.py
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from plotting_utils import plot_classification_confusion_matrix, plot_loss_curve
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

MODEL_NAME = "Multi-Layer Perceptron (MLP)" # Serving as "RNN"
TASK_TYPE = "classification"

# Map user-friendly activation names to scikit-learn's MLPClassifier activation names
ACTIVATION_MAP = {
    "relu": "relu",
    "tanh": "tanh",
    "logistic": "logistic", # Sigmoid equivalent for MLPClassifier
    # scikit-learn's MLPClassifier doesn't directly support "signum", "leaky_relu", or "softmax" as hidden layer activations.
    # "softmax" is typically an output layer activation for multi-class, which MLPClassifier handles internally if needed.
    # We will use 'relu' as a default if an unsupported one is chosen.
    # For "Signum", it's highly non-differentiable and not standard.
    # For "Leaky ReLU", it's not built-in.
}
DEFAULT_ACTIVATION = "relu" # scikit-learn default

def train_evaluate_mlp(X_train, y_train_class, X_val, y_val_class, X_test, y_test_class,
                       epochs=100, activation_function="relu"):
    """
    Trains and evaluates an MLP Classifier model.
    
    Args:
        X_train, y_train_class: Training data and classification targets.
        X_val, y_val_class: Validation data and classification targets.
        X_test, y_test_class: Test data and classification targets.
        epochs (int): Number of training epochs.
        activation_function (str): Name of the activation function for hidden layers.

    Returns:
        dict: A dictionary containing model name, metrics, and plot base64 strings.
    """
    
    mapped_activation = ACTIVATION_MAP.get(activation_function.lower(), DEFAULT_ACTIVATION)
    if activation_function.lower() not in ACTIVATION_MAP:
        print(f"Warning: Activation '{activation_function}' not directly supported or optimal for MLP hidden layers. Using '{mapped_activation}'.")
        print("Supported by this script for MLP: relu, tanh, logistic (sigmoid).")

    model_display_name = f"{MODEL_NAME} (act: {mapped_activation}, epochs: {epochs})"
    print(f"\n--- Training and Evaluating {model_display_name} ---")
    
    results = {
        "model_name": model_display_name,
        "task_type": TASK_TYPE,
        "metrics": {},
        "plots": {}
    }

    # MLPClassifier:
    # hidden_layer_sizes: e.g., (100,) for one hidden layer with 100 neurons. (64, 32) for two.
    # activation: 'relu', 'tanh', 'logistic', 'identity'
    # solver: 'adam' is a good default. 'sgd' is another option.
    # alpha: L2 penalty (regularization term) parameter.
    # batch_size: 'auto' or integer.
    # learning_rate_init: Initial learning rate.
    # max_iter: Number of epochs.
    # early_stopping: Can be useful to prevent overfitting.
    # validation_fraction: Proportion of training data to set aside as validation set for early stopping.
    # warm_start=False: Ensures training from scratch each time.
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation=mapped_activation,
        solver='adam',
        max_iter=epochs,
        random_state=42,
        early_stopping=True, # Enable early stopping based on validation score
        validation_fraction=0.1, # Use 10% of training data for early stopping validation
        n_iter_no_change=10, # Stop if no improvement for 10 consecutive iterations (epochs)
        warm_start=False
    )

    start_time = time.time()
    with warnings.catch_warnings():
        # Suppress ConvergenceWarning if max_iter is reached before convergence
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        model.fit(X_train, y_train_class)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds. Actual epochs run: {model.n_iter_}")

    # --- Evaluation on Training Data ---
    y_train_pred_labels = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)
    train_accuracy = accuracy_score(y_train_class, y_train_pred_labels)
    train_logloss = log_loss(y_train_class, y_train_pred_proba)

    # --- Evaluation on Validation Data (using the dedicated X_val, y_val_class) ---
    # Note: MLPClassifier's internal early stopping uses a split of X_train.
    # Here we evaluate on our explicit validation set.
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
        "Actual Epochs Run": model.n_iter_,
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
        title=f"{model_display_name}: Confusion Matrix (Test)"
    )
    
    # Loss curve from MLPClassifier (validation_scores_ for early stopping, loss_curve_ for training loss)
    # model.loss_curve_ is the training loss
    # model.validation_scores_ is the score on the internal validation set if early_stopping=True
    # We'll plot the training loss. If you want validation loss from early stopping, it's model.validation_scores_
    # but it's a score (e.g. accuracy), not necessarily a "loss" in the same scale as training loss.
    # For simplicity, we'll plot model.loss_curve_
    if hasattr(model, 'loss_curve_'):
        results["plots"]["loss_curve"] = plot_loss_curve(
            model.loss_curve_, 
            val_loss_history=model.validation_scores_ if hasattr(model, 'validation_scores_') and model.early_stopping else None, # Plot validation score if available
            title=f"{model_display_name}: Loss Curve"
        )
        # Note: model.validation_scores_ is a score (higher is better), not a loss.
        # If you want true validation loss, you'd need to calculate it manually per epoch.
        # The plot_loss_curve function can take a val_loss_history. If validation_scores_ is used,
        # it will be plotted on the same graph but represents a score.
        # For a simple demo, plotting training loss (model.loss_curve_) is standard.

    return results

if __name__ == '__main__':
    print(f"{MODEL_NAME} model script loaded.")