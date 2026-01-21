import pandas as pd
import numpy as np
import os

def preprocess_algae_data(input_path, output_path):
    """
    Standardize and clean microalgae growth data for Machine Learning models.
    This pipeline ensures data integrity for high-fidelity predictive modeling.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Source file not found at: {input_path}")
        return None

    # Step 1: Load the dataset
    # We use pandas for efficient data frame manipulation
    df = pd.read_csv(input_path)
    print(f"[INFO] Initial dataset loaded. Shape: {df.shape}")

    # Step 2: Handle Missing Values (Data Imputation)
    # Using mean imputation to maintain statistical consistency in biological datasets
    if df.isnull().values.any():
        df = df.fillna(df.mean(numeric_only=True))
        print("[INFO] Missing values imputed with column mean.")

    # Step 3: Feature Engineering (Biosorption Efficiency)
    # Calculate the removal efficiency of heavy metals as a key performance indicator (KPI)
    # Formula: ((Initial - Final) / Initial) * 100
    if 'initial_metal_conc' in df.columns and 'final_metal_conc' in df.columns:
        df['removal_efficiency'] = ((df['initial_metal_conc'] - df['final_metal_conc']) / 
                                     df['initial_metal_conc']) * 100
        print("[INFO] Calculated feature: removal_efficiency.")

    # Step 4: Outlier Detection and Removal
    # Applying the Z-score method to filter out experimental noise or measurement errors
    # Threshold = 3 standard deviations
    z_scores = np.abs((df - df.mean(numeric_only=True)) / df.std(numeric_only=True))
    df_cleaned = df[(z_scores < 3).all(axis=1)]
    
    print(f"[INFO] Outliers removed. Cleaned dataset shape: {df_cleaned.shape}")

    # Step 5: Export processed data for AI Training
    df_cleaned.to_csv(output_path, index=False)
    print(f"[SUCCESS] Processed data saved to: {output_path}")
    return df_cleaned

def generate_dummy_data(file_path):
    """
    Generate synthetic data for Chlorella vulgaris growth parameters 
    to demonstrate the preprocessing pipeline.
    """
    data = {
        'pH': [7.2, 7.5, 6.8, 8.1, 7.4, 14.0], # 14.0 is an outlier
        'temperature': [25, 27, 24, 26, 25, 25],
        'initial_metal_conc': [100, 100, 100, 100, 100, 100],
        'final_metal_conc': [15, 12, 18, 14, 16, 95]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"[INIT] Synthetic dataset created at: {file_path}")

if __name__ == "__main__":
    # Define file paths
    RAW_DATA = 'data/chlorella_raw.csv'
    PROCESSED_DATA = 'data/chlorella_cleaned.csv'

    # Create dummy data if no real dataset is present in the /data folder
    if not os.path.exists(RAW_DATA):
        generate_dummy_data(RAW_DATA)

    # Execute the preprocessing pipeline
    preprocess_algae_data(RAW_DATA, PROCESSED_DATA)