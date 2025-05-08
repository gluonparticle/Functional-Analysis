# dataset_generator.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TOTAL_SAMPLES = 70000
TRAIN_BASE_SIZE = 50000  # The fraction will apply to this
VALIDATION_SIZE = 10000
TEST_SIZE = TOTAL_SAMPLES - TRAIN_BASE_SIZE - VALIDATION_SIZE  # Should be 10000

def generate_raw_data(num_samples):
    """Generates raw parameters p1 to p7."""
    return pd.DataFrame(np.random.uniform(-100, 100, size=(num_samples, 7)),
                        columns=[f'p{i+1}' for i in range(7)])

def transform_data(raw_params_df):
    """Applies transformations to raw parameters."""
    df = pd.DataFrame(index=raw_params_df.index) # Preserve index from raw_params_df
    num_samples = len(raw_params_df)

    df['A'] = (raw_params_df['p1'] + np.random.uniform(-10, 10, num_samples))**2 / 1000.0
    df['B'] = raw_params_df['p2'] * 10 + np.random.uniform(-3, 3, num_samples)
    df['C'] = np.sin(raw_params_df['p3'] * 0.1) * 20 + np.random.normal(0, 2, num_samples)
    df['D'] = (raw_params_df['p4'] / (np.abs(raw_params_df['p5']) + 1e-6)) * 5 + np.random.uniform(-5, 5, num_samples) # Added 1e-6 to avoid division by zero
    df['E'] = np.cos(raw_params_df['p5'] * 0.05) * 15 + np.random.normal(0, 1.5, num_samples)
    df['F_noisy'] = raw_params_df['p6'] * 0.01 + np.random.uniform(-50, 50, num_samples) # Intentionally noisy
    df['G_signal'] = raw_params_df['p7'] * 5 + np.random.uniform(-2, 2, num_samples)

    # Regression Target (F_noisy is NOT part of the true signal)
    df['Output_true_reg'] = df['A'] + df['B'] + df['C'] + df['D'] + df['E'] + df['G_signal'] \
                            + np.random.normal(0, 5, num_samples) # Add some irreducible noise to target

    # Classification Target (binary, based on median split)
    median_output_reg = np.median(df['Output_true_reg'])
    df['Output_class'] = (df['Output_true_reg'] > median_output_reg).astype(int)
    
    return df

def get_prepared_data_splits():
    """
    Generates the full dataset, transforms it, splits into train_base, validation, and test sets.
    Features are scaled.
    Returns:
        X_train_base_scaled_df, X_val_scaled_df, X_test_scaled_df,
        y_train_base_reg_series, y_val_reg_series, y_test_reg_series,
        y_train_base_class_series, y_val_class_series, y_test_class_series,
        feature_names
    """
    print(f"Generating dataset with {TOTAL_SAMPLES} total samples...")
    raw_data_full = generate_raw_data(TOTAL_SAMPLES)
    transformed_full_df = transform_data(raw_data_full)

    feature_names = ['A', 'B', 'C', 'D', 'E', 'F_noisy', 'G_signal']
    X_full = transformed_full_df[feature_names]
    y_reg_full = transformed_full_df['Output_true_reg']
    y_class_full = transformed_full_df['Output_class']

    # Split into (Train_Base + Validation) and Test
    # Test set is fixed at TEST_SIZE
    X_train_val, X_test, y_train_val_reg, y_test_reg, y_train_val_class, y_test_class = train_test_split(
        X_full, y_reg_full, y_class_full, 
        test_size=TEST_SIZE, 
        random_state=42, 
        stratify=y_class_full # Stratify based on classification target
    )

    # Split (Train_Base + Validation) into Train_Base and Validation
    # Validation size relative to the X_train_val set
    # (TRAIN_BASE_SIZE + VALIDATION_SIZE) is the size of X_train_val
    val_split_ratio_for_train_val = VALIDATION_SIZE / (len(X_train_val))
    
    X_train_base, X_val, y_train_base_reg, y_val_reg, y_train_base_class, y_val_class = train_test_split(
        X_train_val, y_train_val_reg, y_train_val_class, 
        test_size=val_split_ratio_for_train_val, 
        random_state=42, 
        stratify=y_train_val_class # Stratify this split as well
    )
    
    print(f"Initial dataset split: Train_base={len(X_train_base)}, Validation={len(X_val)}, Test={len(X_test)}")
    if len(X_train_base) != TRAIN_BASE_SIZE or len(X_val) != VALIDATION_SIZE or len(X_test) != TEST_SIZE:
        print("Warning: Split sizes do not exactly match target due to rounding or stratification. Actual sizes:")
        print(f"  Train_base: {len(X_train_base)}, Validation: {len(X_val)}, Test: {len(X_test)}")


    # Scale features based on X_train_base
    scaler = StandardScaler()
    X_train_base_scaled = scaler.fit_transform(X_train_base)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames with original indices (if needed, mostly for consistency)
    X_train_base_scaled_df = pd.DataFrame(X_train_base_scaled, columns=feature_names, index=X_train_base.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=feature_names, index=X_val.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

    # Ensure targets are Series with corresponding indices
    y_train_base_reg_series = pd.Series(y_train_base_reg.values, index=X_train_base.index, name='Output_true_reg')
    y_val_reg_series = pd.Series(y_val_reg.values, index=X_val.index, name='Output_true_reg')
    y_test_reg_series = pd.Series(y_test_reg.values, index=X_test.index, name='Output_true_reg')

    y_train_base_class_series = pd.Series(y_train_base_class.values, index=X_train_base.index, name='Output_class')
    y_val_class_series = pd.Series(y_val_class.values, index=X_val.index, name='Output_class')
    y_test_class_series = pd.Series(y_test_class.values, index=X_test.index, name='Output_class')

    return (
        X_train_base_scaled_df, X_val_scaled_df, X_test_scaled_df,
        y_train_base_reg_series, y_val_reg_series, y_test_reg_series,
        y_train_base_class_series, y_val_class_series, y_test_class_series,
        feature_names
    )

if __name__ == '__main__':
    print("Dataset generator module loaded.")
    # (No example execution code here as per user request)