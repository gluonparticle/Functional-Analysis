# main.py
import pandas as pd
from dataset_generator import get_prepared_data_splits, TRAIN_BASE_SIZE
from models.linear_regression_model import train_evaluate_linear_regression
from models.logistic_regression_model import train_evaluate_logistic_regression
from models.random_forest_model import train_evaluate_random_forest
from models.knn_model import train_evaluate_knn
from models.naive_bayes_model import train_evaluate_naive_bayes
from models.mlp_model import train_evaluate_mlp, DEFAULT_ACTIVATION as MLP_DEFAULT_ACTIVATION, ACTIVATION_MAP as MLP_ACTIVATION_MAP
from report_generator import generate_html_report
import time

def get_user_input():
    """Gets experiment parameters from the user."""
    print("\n--- Configure Experiment ---")

    # 1. Dataset Size (Fraction of TRAIN_BASE_SIZE)
    while True:
        try:
            fraction_str = input(f"Enter training data fraction (0.01 to 1.0, applies to {TRAIN_BASE_SIZE} base samples) [default: 1.0]: ")
            if not fraction_str:
                train_data_fraction = 1.0
                break
            train_data_fraction = float(fraction_str)
            if 0.01 <= train_data_fraction <= 1.0:
                break
            else:
                print("Invalid fraction. Please enter a value between 0.01 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # 2. MLP Epochs
    while True:
        try:
            epochs_str = input("Enter number of epochs for MLP [default: 100]: ")
            if not epochs_str:
                mlp_epochs = 100
                break
            mlp_epochs = int(epochs_str)
            if mlp_epochs > 0:
                break
            else:
                print("Epochs must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # 3. MLP Activation Function
    print(f"Available MLP activation functions: {', '.join(MLP_ACTIVATION_MAP.keys())}")
    while True:
        activation_str = input(f"Enter MLP activation function [default: {MLP_DEFAULT_ACTIVATION}]: ").lower()
        if not activation_str:
            mlp_activation = MLP_DEFAULT_ACTIVATION
            break
        if activation_str in MLP_ACTIVATION_MAP or activation_str in ["signum", "leaky_relu", "softmax"]: # Allow user to type these, MLP model will handle mapping/warning
            mlp_activation = activation_str
            break
        else:
            print(f"Invalid activation. Choose from available or type one like 'signum', 'leaky_relu', 'softmax' (will be mapped or warned).")
            
    return train_data_fraction, mlp_epochs, mlp_activation

def main():
    print("Welcome to the Functional Analysis of Deep Learning Algorithms Project!")
    
    train_data_fraction, mlp_epochs, mlp_activation = get_user_input()

    # 1. Prepare Data
    print("\n--- Preparing Data ---")
    # These are already scaled DataFrames/Series with appropriate indices
    X_train_base_scaled, X_val_scaled, X_test_scaled, \
    y_train_base_reg, y_val_reg, y_test_reg, \
    y_train_base_class, y_val_class, y_test_class, \
    feature_names = get_prepared_data_splits()

    # Apply train_data_fraction to the base training set
    num_train_samples_to_use = int(TRAIN_BASE_SIZE * train_data_fraction)
    if num_train_samples_to_use < 1: # Ensure at least 1 sample
        num_train_samples_to_use = 1
    if num_train_samples_to_use > len(X_train_base_scaled): # Cap at available
        num_train_samples_to_use = len(X_train_base_scaled)

    # Sample the required fraction from the base training data
    # Use .sample() for random sampling or slicing for taking the first N
    # For reproducibility and to ensure stratification if targets are imbalanced,
    # sampling is generally better if the base set is large and well-shuffled.
    # Here, since it's already split, we can just take a slice.
    # Or, for true random subsample:
    if train_data_fraction < 1.0:
        print(f"Using {num_train_samples_to_use} training samples ({train_data_fraction*100:.0f}% of base).")
        # Ensure consistent sampling if run multiple times with same fraction
        X_train_final = X_train_base_scaled.sample(n=num_train_samples_to_use, random_state=42)
        y_train_final_reg = y_train_base_reg.loc[X_train_final.index]
        y_train_final_class = y_train_base_class.loc[X_train_final.index]
    else:
        print(f"Using all {len(X_train_base_scaled)} base training samples.")
        X_train_final = X_train_base_scaled
        y_train_final_reg = y_train_base_reg
        y_train_final_class = y_train_base_class
        
    print(f"Final training set size: {len(X_train_final)}")
    print(f"Validation set size: {len(X_val_scaled)}")
    print(f"Test set size: {len(X_test_scaled)}")


    all_results = []
    overall_start_time = time.time()

    # 2. Run Models
    # Each function should return a dictionary with "model_name", "metrics", "plots"
    
    try:
        results_lr = train_evaluate_linear_regression(
            X_train_final, y_train_final_reg, X_val_scaled, y_val_reg, X_test_scaled, y_test_reg
        )
        all_results.append(results_lr)
    except Exception as e:
        print(f"Error running Linear Regression: {e}")
        all_results.append({"model_name": "Linear Regression", "error": str(e), "metrics": {}, "plots":{}})

    try:
        results_logr = train_evaluate_logistic_regression(
            X_train_final, y_train_final_class, X_val_scaled, y_val_class, X_test_scaled, y_test_class
        )
        all_results.append(results_logr)
    except Exception as e:
        print(f"Error running Logistic Regression: {e}")
        all_results.append({"model_name": "Logistic Regression", "error": str(e), "metrics": {}, "plots":{}})

    try:
        results_rf = train_evaluate_random_forest( # Defaults to classification
            X_train_final, y_train_final_class, X_val_scaled, y_val_class, X_test_scaled, y_test_class
        )
        all_results.append(results_rf)
    except Exception as e:
        print(f"Error running Random Forest: {e}")
        all_results.append({"model_name": "Random Forest (classification)", "error": str(e), "metrics": {}, "plots":{}})

    try:
        # k-NN can be slow, consider a smaller k or ensure data is not excessively large for demo
        results_knn = train_evaluate_knn(
            X_train_final, y_train_final_class, X_val_scaled, y_val_class, X_test_scaled, y_test_class, n_neighbors=7 
        )
        all_results.append(results_knn)
    except Exception as e:
        print(f"Error running k-NN: {e}")
        all_results.append({"model_name": "k-Nearest Neighbors (k=7)", "error": str(e), "metrics": {}, "plots":{}})

    try:
        results_nb = train_evaluate_naive_bayes(
            X_train_final, y_train_final_class, X_val_scaled, y_val_class, X_test_scaled, y_test_class
        )
        all_results.append(results_nb)
    except Exception as e:
        print(f"Error running Naive Bayes: {e}")
        all_results.append({"model_name": "Naive Bayes (Gaussian)", "error": str(e), "metrics": {}, "plots":{}})

    try:
        results_mlp = train_evaluate_mlp(
            X_train_final, y_train_final_class, X_val_scaled, y_val_class, X_test_scaled, y_test_class,
            epochs=mlp_epochs, activation_function=mlp_activation
        )
        all_results.append(results_mlp)
    except Exception as e:
        print(f"Error running MLP: {e}")
        all_results.append({"model_name": f"MLP (act: {mlp_activation}, epochs: {mlp_epochs})", "error": str(e), "metrics": {}, "plots":{}})

    overall_end_time = time.time()
    print(f"\nAll models processed in {overall_end_time - overall_start_time:.2f} seconds.")

    # 3. Generate Report
    generate_html_report(all_results, train_data_fraction, mlp_epochs, mlp_activation)

    print("\nAnalysis complete. Check the 'reports' directory for the HTML file.")

if __name__ == "__main__":
    main()