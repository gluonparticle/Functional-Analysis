# models/linear_regression_model.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from plotting_utils import plot_regression_true_vs_predicted, plot_residuals
import time

MODEL_NAME = "Linear Regression"
TASK_TYPE = "regression" # This model is primarily for regression

def train_evaluate_linear_regression(X_train, y_train_reg, X_val, y_val_reg, X_test, y_test_reg):
    """
    Trains and evaluates a Linear Regression model.
    
    Args:
        X_train, y_train_reg: Training data and regression targets.
        X_val, y_val_reg: Validation data and regression targets.
        X_test, y_test_reg: Test data and regression targets.

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

    model = LinearRegression()

    start_time = time.time()
    model.fit(X_train, y_train_reg)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # --- Evaluation on Training Data (Optional, but good for seeing fit) ---
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_train_pred))
    train_r2 = r2_score(y_train_reg, y_train_pred)
    train_mae = mean_absolute_error(y_train_reg, y_train_pred)

    # --- Evaluation on Validation Data ---
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val_reg, y_val_pred))
    val_r2 = r2_score(y_val_reg, y_val_pred)
    val_mae = mean_absolute_error(y_val_reg, y_val_pred)
    
    # --- Evaluation on Test Data ---
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    inference_time_test = time.time() - start_time
    
    test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))
    test_r2 = r2_score(y_test_reg, y_test_pred)
    test_mae = mean_absolute_error(y_test_reg, y_test_pred)
    
    results["metrics"] = {
        "Training Time (s)": f"{training_time:.4f}",
        "Inference Time (Test) (s)": f"{inference_time_test:.4f}",
        "Train RMSE": f"{train_rmse:.4f}",
        "Train R2": f"{train_r2:.4f}",
        "Train MAE": f"{train_mae:.4f}",
        "Validation RMSE": f"{val_rmse:.4f}",
        "Validation R2": f"{val_r2:.4f}",
        "Validation MAE": f"{val_mae:.4f}",
        "Test RMSE": f"{test_rmse:.4f}",
        "Test R2": f"{test_r2:.4f}",
        "Test MAE": f"{test_mae:.4f}",
        # "Loss" for linear regression is typically MSE (RMSE^2)
        "Test Loss (MSE)": f"{test_rmse**2:.4f}" 
    }
    print(f"Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")

    # --- Generate Plots ---
    # For regression, "datapoints vs fit type" can be True vs. Predicted
    results["plots"]["true_vs_predicted_test"] = plot_regression_true_vs_predicted(
        y_test_reg, y_test_pred, title=f"{MODEL_NAME}: True vs. Predicted (Test)"
    )
    results["plots"]["residuals_test"] = plot_residuals(
        y_test_reg, y_test_pred, title=f"{MODEL_NAME}: Residuals (Test)"
    )
    # No loss curve for Linear Regression in the typical sense of epochs

    return results

if __name__ == '__main__':
    print(f"{MODEL_NAME} model script loaded.")
    # This script is intended to be imported and used by a main runner.
    # For a quick standalone test (requires dataset_generator.py in PYTHONPATH):
    # from dataset_generator import get_prepared_data_splits
    # X_train_base, X_val, X_test, \
    # y_train_base_r, y_val_r, y_test_r, \
    # y_train_base_c, y_val_c, y_test_c, \
    # features = get_prepared_data_splits()
    # # Use a fraction of the training data for a quick test
    # X_train_sample = X_train_base.sample(frac=0.1, random_state=42)
    # y_train_sample_r = y_train_base_r.loc[X_train_sample.index]
    # results = train_evaluate_linear_regression(X_train_sample, y_train_sample_r, X_val, y_val_r, X_test, y_test_r)
    # print("\nResults:")
    # for key, value in results["metrics"].items():
    #     print(f"  {key}: {value}")
    # print(f"  Plot 'true_vs_predicted_test' generated: {'Yes' if results['plots'].get('true_vs_predicted_test') else 'No'}")