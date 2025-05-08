# main.py (This is your data processing script, now with argparse)
import pandas as pd
import argparse # For command line arguments
import os
import shutil # For copying report file

# Your existing imports
from dataset_generator import get_prepared_data_splits, TRAIN_BASE_SIZE, RAW_PARAMETER_NAMES, TRANSFORMED_FEATURE_MAPPING, ALL_TRANSFORMED_FEATURE_NAMES
from models.linear_regression_model import train_evaluate_linear_regression
from models.logistic_regression_model import train_evaluate_logistic_regression
from models.random_forest_model import train_evaluate_random_forest
from models.knn_model import train_evaluate_knn
from models.naive_bayes_model import train_evaluate_naive_bayes
from models.mlp_model import train_evaluate_mlp, DEFAULT_ACTIVATION as MLP_DEFAULT_ACTIVATION, ACTIVATION_MAP as MLP_ACTIVATION_MAP
from report_generator import generate_html_report
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Functional Analysis of Deep Learning Algorithms.")
    parser.add_argument("--features", type=str, required=True,
                        help="Comma-separated list of original parameter numbers (e.g., '1,2,7'), or 'all', or 'all_but_noisy'.")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Training data fraction (0.01 to 1.0).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for MLP.")
    parser.add_argument("--activation", type=str, default=MLP_DEFAULT_ACTIVATION,
                        help="MLP activation function.")
    parser.add_argument("--output_report_name", type=str, default=None, # e.g., "report_JOBID.html"
                        help="Specific filename for the output HTML report in the 'reports' directory.")
    return parser.parse_args()

def run_analysis_logic(selected_features_input_str, train_data_fraction, mlp_epochs, mlp_activation, output_report_filename_override=None):
    """
    Contains the core logic of your original main function, but parameterized.
    Returns the path to the generated report.
    """
    print(f"Starting analysis with: features='{selected_features_input_str}', fraction={train_data_fraction}, epochs={mlp_epochs}, activation='{mlp_activation}'")

    # 1. Prepare Data
    X_train_base_all_feats, X_val_all_feats, X_test_all_feats, \
    y_train_base_reg, y_val_reg, y_test_reg, \
    y_train_base_class, y_val_class, y_test_class, \
    all_generated_feature_names_from_generator = get_prepared_data_splits() # This is ALL_TRANSFORMED_FEATURE_NAMES

    if X_train_base_all_feats.empty:
        print("Critical error: Base training data is empty after generation and splitting.")
        return None # Indicate failure

    # --- Feature Selection Logic (moved from old main's get_user_input) ---
    selected_transformed_features = []
    if selected_features_input_str.lower() == 'all':
        selected_transformed_features = list(all_generated_feature_names_from_generator)
    elif selected_features_input_str.lower() == 'all_but_noisy':
        noisy_features_to_exclude = [
            TRANSFORMED_FEATURE_MAPPING.get('p6'), 
            TRANSFORMED_FEATURE_MAPPING.get('p13'),
            TRANSFORMED_FEATURE_MAPPING.get('p15')
        ] # Assuming these are the keys for noisy features in your 15-param setup
        selected_transformed_features = [f for f in all_generated_feature_names_from_generator if f not in noisy_features_to_exclude]
    else:
        try:
            selected_p_indices = [int(p.strip()) for p in selected_features_input_str.split(',')]
            for p_idx in selected_p_indices:
                raw_param_name = f'p{p_idx}'
                if raw_param_name in TRANSFORMED_FEATURE_MAPPING:
                    transformed_name = TRANSFORMED_FEATURE_MAPPING[raw_param_name]
                    if transformed_name in all_generated_feature_names_from_generator:
                        selected_transformed_features.append(transformed_name)
                # else: (silently ignore invalid p_idx for CLI, or add verbose warning)
            selected_transformed_features = list(set(selected_transformed_features)) # Unique
        except ValueError:
            print(f"Error parsing features string '{selected_features_input_str}'. Using 'all_but_noisy'.")
            noisy_features_to_exclude = [TRANSFORMED_FEATURE_MAPPING.get('p6'), TRANSFORMED_FEATURE_MAPPING.get('p13'), TRANSFORMED_FEATURE_MAPPING.get('p15')]
            selected_transformed_features = [f for f in all_generated_feature_names_from_generator if f not in noisy_features_to_exclude]
    
    if not selected_transformed_features:
        print("No valid features selected for model training. Aborting analysis for these parameters.")
        return None

    print(f"Features being used for this run: {selected_transformed_features}")

    X_train_base_selected_feats = X_train_base_all_feats[selected_transformed_features]
    X_val_selected_feats = X_val_all_feats[selected_transformed_features]
    X_test_selected_feats = X_test_all_feats[selected_transformed_features]

    # Apply train_data_fraction
    num_train_samples_to_use = int(len(X_train_base_selected_feats) * train_data_fraction)
    if num_train_samples_to_use < 1 and len(X_train_base_selected_feats) > 0: num_train_samples_to_use = 1
    
    if train_data_fraction < 1.0 and len(X_train_base_selected_feats) > 0:
        X_train_final = X_train_base_selected_feats.sample(n=num_train_samples_to_use, random_state=42)
    else:
        X_train_final = X_train_base_selected_feats
        
    y_train_final_reg = y_train_base_reg.loc[X_train_final.index] if not X_train_final.empty else pd.Series(dtype='float64')
    y_train_final_class = y_train_base_class.loc[X_train_final.index] if not X_train_final.empty else pd.Series(dtype='int64')
        
    if X_train_final.empty:
        print("Warning: Final training dataset (X_train_final) is empty. No models will be trained.")
        # Still generate a report indicating this
        all_results_for_report = [{"model_name": "Setup Error", "error": "Training data became empty after selection/sampling.", "metrics":{}, "plots":{}}]
        report_path = generate_html_report(
            all_results_for_report, train_data_fraction, mlp_epochs, mlp_activation,
            selected_features_for_report=selected_transformed_features,
            output_filename_override=output_report_filename_override # Pass override
        )
        return report_path


    print(f"Final training set size: {len(X_train_final)} (Features: {X_train_final.shape[1]})")

    all_results_for_report = []
    # ... (Your model training loops - same as before, using X_train_final, y_train_final_*, X_val_selected_feats etc.)
    # --- Linear Regression ---
    if not X_train_final.empty and not y_train_final_reg.empty:
        try:
            results_lr = train_evaluate_linear_regression(
                X_train_final, y_train_final_reg, 
                X_val_selected_feats, y_val_reg,
                X_test_selected_feats, y_test_reg
            )
            all_results_for_report.append(results_lr)
        except Exception as e:
            print(f"Error running Linear Regression: {e}")
            all_results_for_report.append({"model_name": "Linear Regression", "error": str(e), "metrics": {}, "plots":{}})
    
    # --- Classification Models ---
    if not X_train_final.empty and not y_train_final_class.empty:
        common_args_class = (X_train_final, y_train_final_class, 
                             X_val_selected_feats, y_val_class, 
                             X_test_selected_feats, y_test_class)
        try:
            results_logr = train_evaluate_logistic_regression(*common_args_class)
            all_results_for_report.append(results_logr)
        except Exception as e: print(f"Error LogR: {e}"); all_results_for_report.append({"model_name":"LogR", "error":str(e)})
        try:
            results_rf = train_evaluate_random_forest(*common_args_class)
            all_results_for_report.append(results_rf)
        except Exception as e: print(f"Error RF: {e}"); all_results_for_report.append({"model_name":"RF", "error":str(e)})
        try:
            results_knn = train_evaluate_knn(*common_args_class, n_neighbors=7)
            all_results_for_report.append(results_knn)
        except Exception as e: print(f"Error KNN: {e}"); all_results_for_report.append({"model_name":"KNN", "error":str(e)})
        try:
            results_nb = train_evaluate_naive_bayes(*common_args_class)
            all_results_for_report.append(results_nb)
        except Exception as e: print(f"Error NB: {e}"); all_results_for_report.append({"model_name":"NB", "error":str(e)})
        try:
            results_mlp = train_evaluate_mlp(*common_args_class, epochs=mlp_epochs, activation_function=mlp_activation)
            all_results_for_report.append(results_mlp)
        except Exception as e: print(f"Error MLP: {e}"); all_results_for_report.append({"model_name":"MLP", "error":str(e)})


    # 3. Generate Report
    # Modify generate_html_report to accept output_filename_override
    report_path = generate_html_report(
        all_results_for_report, 
        train_data_fraction, 
        mlp_epochs, 
        mlp_activation,
        selected_features_for_report=selected_transformed_features,
        output_filename_override=output_report_filename_override # Pass the override
    )
    return report_path


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Main.py called with CLI args: {args}")
    
    # Make sure reports directory exists
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Call the main logic function
    generated_report_path = run_analysis_logic(
        args.features, 
        args.fraction, 
        args.epochs, 
        args.activation,
        args.output_report_name # This will be like "report_JOBID.html"
    )

    if generated_report_path:
        print(f"Analysis complete. Report generated at: {generated_report_path}")
        # If app.py needs to know the exact path, this print can be captured by subprocess
        # Or, if output_report_name is predictable, app.py already knows it.
    else:
        print("Analysis failed or no report was generated.")
        # Exit with an error code if called by subprocess
        exit(1) 