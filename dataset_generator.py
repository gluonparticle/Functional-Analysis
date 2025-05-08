# dataset_generator.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Dataset Configuration ---
TOTAL_SAMPLES = 70000 # You can adjust this
TRAIN_BASE_SIZE = 50000
VALIDATION_SIZE = 10000
TEST_SIZE = TOTAL_SAMPLES - TRAIN_BASE_SIZE - VALIDATION_SIZE

# --- Feature Definitions for 15 Parameters ---
NUM_RAW_PARAMS = 15
RAW_PARAMETER_NAMES = [f'p{i+1}' for i in range(NUM_RAW_PARAMS)]

# p1-p15 mapped to transformed features.
# We'll designate up to 3 as "fully noisy".
# Others will have varying degrees of signal and noise.
TRANSFORMED_FEATURE_MAPPING = {
    # Original 7, slightly adjusted names for clarity if needed
    'p1': 'T1_power_noisy',         # Signal with noise
    'p2': 'T2_linear_high_noise',   # Signal with high noise
    'p3': 'T3_trig_complex',        # Complex signal
    'p4': 'T4_interaction_A',       # Interaction (p4, p5)
    'p5': 'T5_exponential_var',     # Signal with noise (also used in T4)
    'p6': 'T6_FULLY_NOISY_A',       # <<< FULLY NOISY 1
    'p7': 'T7_cubic_strong_signal', # Strong signal

    # New parameters p8-p15
    'p8': 'T8_log_mixed_noise',     # Logarithmic signal with noise
    'p9': 'T9_polynomial_signal',   # Polynomial signal
    'p10': 'T10_periodic_subtle',   # Subtle periodic signal
    'p11': 'T11_interaction_B',     # Interaction (p11, p12)
    'p12': 'T12_ratio_unstable',    # Ratio, can be unstable (also used in T11)
    'p13': 'T13_FULLY_NOISY_B',     # <<< FULLY NOISY 2
    'p14': 'T14_conditional_logic', # Feature based on some conditions of p14
    'p15': 'T15_FULLY_NOISY_C'      # <<< FULLY NOISY 3
}
ALL_TRANSFORMED_FEATURE_NAMES = list(TRANSFORMED_FEATURE_MAPPING.values())


def generate_raw_data_df(num_samples):
    """Generates a DataFrame of raw parameters p1 to pNUM_RAW_PARAMS."""
    return pd.DataFrame(np.random.uniform(-100, 100, size=(num_samples, NUM_RAW_PARAMS)),
                        columns=RAW_PARAMETER_NAMES)

def transform_data_df(raw_params_df):
    """Applies transformations for 15 parameters."""
    df = pd.DataFrame(index=raw_params_df.index)
    num_samples = len(raw_params_df)

    # Transformations (ensure all raw_params_df['pX'] are used)
    # p1 -> T1_power_noisy
    df['T1_power_noisy'] = (raw_params_df['p1'] + np.random.uniform(-70, 70, num_samples))**3 / 2e5 
    
    # p2 -> T2_linear_high_noise
    df['T2_linear_high_noise'] = raw_params_df['p2'] * 15 * (1 + np.random.uniform(-0.7, 0.7, num_samples))

    # p3 -> T3_trig_complex
    scaled_p3 = raw_params_df['p3'] * 0.02
    df['T3_trig_complex'] = np.cos(scaled_p3 * 5) * np.sin(raw_params_df['p3'] * 0.1 + np.random.uniform(-1,1,num_samples)) * 30 \
                           + np.random.normal(0, 10, num_samples)
    df['T3_trig_complex'] = np.clip(df['T3_trig_complex'], -1000, 1000)

    # p4, p5 -> T4_interaction_A
    df['T4_interaction_A'] = ((raw_params_df['p4'] * raw_params_df['p5'] * np.random.normal(1, 0.5, num_samples)) / 300) \
                                 + np.random.uniform(-40, 40, num_samples)

    # p5 -> T5_exponential_var
    df['T5_exponential_var'] = np.exp(raw_params_df['p5'] * 0.015 * np.random.choice([-1,1], size=num_samples)) \
                                  * np.random.uniform(0.3, 1.7, num_samples) + np.random.normal(0,12,num_samples)
    df['T5_exponential_var'] = np.clip(df['T5_exponential_var'], -1200, 1200)

    # p6 -> T6_FULLY_NOISY_A (Dominated by random noise)
    df['T6_FULLY_NOISY_A'] = raw_params_df['p6'] * 0.0001 + np.random.uniform(-300, 300, num_samples)

    # p7 -> T7_cubic_strong_signal
    df['T7_cubic_strong_signal'] = (raw_params_df['p7']**3 / 700) + (raw_params_df['p7'] * 3.0) \
                                  + np.random.normal(0, 10, num_samples)

    # p8 -> T8_log_mixed_noise
    # Handle potential log(0) or log(<0) by using abs and adding small constant
    df['T8_log_mixed_noise'] = np.log(np.abs(raw_params_df['p8']) + 1e-3) * np.random.uniform(5,15,num_samples) \
                               + raw_params_df['p8']*0.1 + np.random.normal(0,5,num_samples)
    df['T8_log_mixed_noise'] = np.clip(df['T8_log_mixed_noise'], -500, 500)

    # p9 -> T9_polynomial_signal
    df['T9_polynomial_signal'] = (0.001 * raw_params_df['p9']**3) - (0.05 * raw_params_df['p9']**2) + \
                                 (2 * raw_params_df['p9']) + np.random.normal(0, 15, num_samples)

    # p10 -> T10_periodic_subtle
    df['T10_periodic_subtle'] = np.sin(raw_params_df['p10'] * 0.05 + np.pi/3) * 10 + \
                                np.cos(raw_params_df['p10'] * 0.2) * 5 + np.random.normal(0,3,num_samples)

    # p11, p12 -> T11_interaction_B
    df['T11_interaction_B'] = (raw_params_df['p11'] / (np.abs(raw_params_df['p12'] * 0.5) + 5)) * \
                              np.sin(raw_params_df['p11']*0.1) * 20 + np.random.uniform(-20,20,num_samples)
    df['T11_interaction_B'] = np.clip(df['T11_interaction_B'], -800, 800)
    
    # p12 -> T12_ratio_unstable (also used in T11)
    # Ensure p12 is not too close to zero in denominator
    df['T12_ratio_unstable'] = (raw_params_df['p12'] / (raw_params_df['p10'] + 101 + np.random.uniform(-10,10,num_samples))) * 50 \
                               + np.random.normal(0,25,num_samples)
    df['T12_ratio_unstable'] = np.clip(df['T12_ratio_unstable'], -1000, 1000)

    # p13 -> T13_FULLY_NOISY_B
    df['T13_FULLY_NOISY_B'] = raw_params_df['p13'] * np.random.uniform(-0.001,0.001,num_samples) + \
                              np.random.standard_t(df=3, size=num_samples) * 50 # t-distribution for heavy tails
    
    # p14 -> T14_conditional_logic
    cond1 = raw_params_df['p14'] > 50
    cond2 = raw_params_df['p14'] < -50
    df['T14_conditional_logic'] = np.select(
        [cond1, cond2], 
        [raw_params_df['p14'] * 2 + np.random.normal(0,5,num_samples), raw_params_df['p14'] * 0.5 + np.random.normal(0,10,num_samples)], 
        default=np.sin(raw_params_df['p14']*0.1)*10 + np.random.normal(0,2,num_samples)
    )
    
    # p15 -> T15_FULLY_NOISY_C
    df['T15_FULLY_NOISY_C'] = np.random.laplace(loc=raw_params_df['p15']*0.00001, scale=100, size=num_samples)


    # --- Target Variable Generation ---
    # Select a subset of transformed features (excluding fully noisy ones) to form the true signal
    # This makes it harder for models as not all non-noisy features contribute directly.
    true_signal_features = [
        'T1_power_noisy', 'T2_linear_high_noise', 'T3_trig_complex', 
        'T4_interaction_A', 'T5_exponential_var', 'T7_cubic_strong_signal',
        'T8_log_mixed_noise', 'T9_polynomial_signal', 'T10_periodic_subtle',
        'T11_interaction_B', 'T14_conditional_logic'
        # 'T12_ratio_unstable' is excluded from true signal to make it a partially relevant/confusing feature
    ]
    
    # Assign random coefficients to the true signal features
    # np.random.seed(42) # for reproducible coefficients if desired
    coefficients = np.random.uniform(0.5, 1.5, size=len(true_signal_features)) * np.random.choice([-1,1], size=len(true_signal_features))
    
    df['Output_true_reg'] = 0
    for i, feat_name in enumerate(true_signal_features):
        if feat_name in df.columns: # Check if feature was successfully generated
            df['Output_true_reg'] += df[feat_name] * coefficients[i]
    
    df['Output_true_reg'] = (df['Output_true_reg'] / (len(true_signal_features) * 0.75)) + \
                             np.random.normal(0, 40, num_samples) # Increased irreducible noise

    # NaN and Inf handling (critical with complex transformations)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cols_to_check_for_nan = ALL_TRANSFORMED_FEATURE_NAMES + ['Output_true_reg']
    df.dropna(subset=[col for col in cols_to_check_for_nan if col in df.columns], inplace=True) # Drop if any selected feature or target is NaN
    
    if df.empty:
        raise ValueError("All rows dropped after NaN handling. Check transformations for stability.")
        
    median_output_reg = np.median(df['Output_true_reg'])
    df['Output_class'] = (df['Output_true_reg'] > median_output_reg).astype(int)
    
    return df.reset_index(drop=True)


def get_prepared_data_splits():
    """
    Generates the full dataset for 15 parameters, transforms it, splits, and scales.
    Returns:
        X_train_base_scaled_df, X_val_scaled_df, X_test_scaled_df (containing ALL transformed features),
        y_train_base_reg_series, y_val_reg_series, y_test_reg_series,
        y_train_base_class_series, y_val_class_series, y_test_class_series,
        ALL_TRANSFORMED_FEATURE_NAMES (list of all possible feature names models could use)
    """
    print(f"Generating dataset with {TOTAL_SAMPLES} total samples for {NUM_RAW_PARAMS} raw parameters...")
    raw_full_df = generate_raw_data_df(TOTAL_SAMPLES)
    transformed_full_df = transform_data_df(raw_full_df) 

    if transformed_full_df.empty:
        raise Exception("Failed to generate transformed data; DataFrame is empty after transformations and NaN cleaning.")
    
    # Ensure all features defined in ALL_TRANSFORMED_FEATURE_NAMES are columns in the df
    # If a feature column was entirely NaNs and got dropped, this would be an issue.
    # The current transform_data_df tries to keep all columns unless all rows become NaN for that column.
    # For safety, select only from available columns that are also in our defined list.
    
    # features_present_in_df = [f_name for f_name in ALL_TRANSFORMED_FEATURE_NAMES if f_name in transformed_full_df.columns]
    # If any feature from ALL_TRANSFORMED_FEATURE_NAMES is missing, it's an issue with transform_data_df
    # For simplicity, we assume transform_data_df will produce all columns in ALL_TRANSFORMED_FEATURE_NAMES
    # or handle their absence internally (e.g., by not using them in target calculation if they become all NaNs).
    # The dropna in transform_data_df acts on rows, not columns.

    X_full = transformed_full_df[ALL_TRANSFORMED_FEATURE_NAMES] # Use all defined transformed features
    y_reg_full = transformed_full_df['Output_true_reg']
    y_class_full = transformed_full_df['Output_class']

    # Split into (Train_Base + Validation) and Test
    X_train_val, X_test, y_train_val_reg, y_test_reg, y_train_val_class, y_test_class = train_test_split(
        X_full, y_reg_full, y_class_full, 
        test_size=TEST_SIZE, 
        random_state=42, 
        stratify=y_class_full 
    )

    val_split_ratio_for_train_val = VALIDATION_SIZE / (len(X_train_val)) if len(X_train_val) > 0 else 0
    
    if not (0 < val_split_ratio_for_train_val < 1):
        # Fallback logic for small X_train_val (copied from previous version, seems robust)
        print(f"Warning: Train+Val set size ({len(X_train_val)}) is too small or problematic for desired validation split ({VALIDATION_SIZE}). Adjusting.")
        if len(X_train_val) > 1:
            actual_val_size = min(VALIDATION_SIZE, len(X_train_val) - 1)
            if actual_val_size <=0 : actual_val_size = int(len(X_train_val) * 0.2) 
            if actual_val_size <=0 : actual_val_size = 1 
            
            if len(X_train_val) - actual_val_size <= 0: 
                 X_train_base, y_train_base_reg, y_train_base_class = X_train_val, y_train_val_reg, y_train_val_class
                 X_val, y_val_reg, y_val_class = pd.DataFrame(columns=ALL_TRANSFORMED_FEATURE_NAMES), pd.Series(dtype='float64'), pd.Series(dtype='int64')
            else:
                X_train_base, X_val, y_train_base_reg, y_val_reg, y_train_base_class, y_val_class = train_test_split(
                    X_train_val, y_train_val_reg, y_train_val_class,
                    test_size=actual_val_size, random_state=42, stratify=y_train_val_class
                )
        else: 
            X_train_base, y_train_base_reg, y_train_base_class = X_train_val, y_train_val_reg, y_train_val_class
            X_val, y_val_reg, y_val_class = pd.DataFrame(columns=ALL_TRANSFORMED_FEATURE_NAMES), pd.Series(dtype='float64'), pd.Series(dtype='int64')
            print("Warning: Validation set is empty due to small Train+Val size.")
    else:
        X_train_base, X_val, y_train_base_reg, y_val_reg, y_train_base_class, y_val_class = train_test_split(
            X_train_val, y_train_val_reg, y_train_val_class, 
            test_size=val_split_ratio_for_train_val, 
            random_state=42, 
            stratify=y_train_val_class 
        )
    
    print(f"Dataset split: Train_base={len(X_train_base)}, Validation={len(X_val)}, Test={len(X_test)}")

    scaler = StandardScaler()
    if not X_train_base.empty:
        X_train_base_scaled_values = scaler.fit_transform(X_train_base)
        X_train_base_scaled_df = pd.DataFrame(X_train_base_scaled_values, columns=ALL_TRANSFORMED_FEATURE_NAMES, index=X_train_base.index)
    else:
        X_train_base_scaled_df = pd.DataFrame(columns=ALL_TRANSFORMED_FEATURE_NAMES)

    if not X_val.empty:
        X_val_scaled_values = scaler.transform(X_val)
        X_val_scaled_df = pd.DataFrame(X_val_scaled_values, columns=ALL_TRANSFORMED_FEATURE_NAMES, index=X_val.index)
    else:
        X_val_scaled_df = pd.DataFrame(columns=ALL_TRANSFORMED_FEATURE_NAMES)

    if not X_test.empty:
        X_test_scaled_values = scaler.transform(X_test)
        X_test_scaled_df = pd.DataFrame(X_test_scaled_values, columns=ALL_TRANSFORMED_FEATURE_NAMES, index=X_test.index)
    else:
        X_test_scaled_df = pd.DataFrame(columns=ALL_TRANSFORMED_FEATURE_NAMES)

    y_train_base_reg_series = pd.Series(y_train_base_reg.values, index=X_train_base.index, name='Output_true_reg') if not X_train_base.empty else pd.Series(dtype='float64', name='Output_true_reg')
    y_val_reg_series = pd.Series(y_val_reg.values, index=X_val.index, name='Output_true_reg') if not X_val.empty else pd.Series(dtype='float64', name='Output_true_reg')
    y_test_reg_series = pd.Series(y_test_reg.values, index=X_test.index, name='Output_true_reg') if not X_test.empty else pd.Series(dtype='float64', name='Output_true_reg')

    y_train_base_class_series = pd.Series(y_train_base_class.values, index=X_train_base.index, name='Output_class') if not X_train_base.empty else pd.Series(dtype='int64', name='Output_class')
    y_val_class_series = pd.Series(y_val_class.values, index=X_val.index, name='Output_class') if not X_val.empty else pd.Series(dtype='int64', name='Output_class')
    y_test_class_series = pd.Series(y_test_class.values, index=X_test.index, name='Output_class') if not X_test.empty else pd.Series(dtype='int64', name='Output_class')

    return (
        X_train_base_scaled_df, X_val_scaled_df, X_test_scaled_df,
        y_train_base_reg_series, y_val_reg_series, y_test_reg_series,
        y_train_base_class_series, y_val_class_series, y_test_class_series,
        ALL_TRANSFORMED_FEATURE_NAMES 
    )

if __name__ == '__main__':
    print("Dataset generator module (15 params, No MySQL).")
    print(f"Raw parameter names: {RAW_PARAMETER_NAMES}")
    print(f"Transformed feature mapping (raw_param -> transformed_feature_name):\n{TRANSFORMED_FEATURE_MAPPING}")
    print(f"All possible transformed feature names models might use: {ALL_TRANSFORMED_FEATURE_NAMES}")