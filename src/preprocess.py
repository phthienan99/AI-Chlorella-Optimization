import pandas as pd
import numpy as np
import os

# ==============================
# COLUMN NAME MAPPING
# ==============================

COLUMN_MAP = {
    "pH": "pH",
    "Light": "Light (μmol/m²/s)",
    "Temperature": "Temp (°C)",
    "TN0": "TN₀ (mg/L)",
    "TP0": "TP₀ (mg/L)",
    "COD0": "COD₀",
    "Time": "Cultivation Time (days)",
    "N_rem": "N Removal Efficiency (%)",
    "P_rem": "P Removal Efficiency (%)",
    "COD_rem": "COD rem (%)",
    "Biomass": "Biomass Yield (g/L)",
    "Source": "Source"
}

INPUT_FEATURES = [
    "pH",
    "Light (μmol/m²/s)",
    "Temp (°C)",
    "TN₀ (mg/L)",
    "TP₀ (mg/L)",
    "COD₀",
    "Cultivation Time (days)"
]

OUTPUT_TARGETS = [
    "N Removal Efficiency (%)",
    "P Removal Efficiency (%)",
    "COD rem (%)",
    "Biomass Yield (g/L)"
]

def preprocess_algae_data(input_path, output_path):

    # Check if the input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found: {input_path}")

    # Load raw data from CSV file
    df = pd.read_csv(input_path)
    print(f"[INFO] Raw dataset loaded: {df.shape}")

    # Rename columns to standardized scientific schema
    df = df.rename(columns=COLUMN_MAP)

    # Drop the 'Metal' and 'Source' columns for model input (but keep 'Source' for documentation)
    if "Metal" in df.columns:
        df = df.drop(columns=["Metal"])
    if "Source" in df.columns:
        df = df.drop(columns=["Source"])

    # Ensure all required columns exist, and if missing, add them with NaN values
    for col in INPUT_FEATURES + OUTPUT_TARGETS:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only the necessary columns (input features and output targets)
    df = df[INPUT_FEATURES + OUTPUT_TARGETS]

    # Remove rows where all output targets are NaN
    df = df.dropna(how="all", subset=OUTPUT_TARGETS)
    print(f"[INFO] After filtering unusable rows: {df.shape}")

    # Impute missing values in input features with the median of the respective column
    for col in INPUT_FEATURES:
        # Only fill NaN for numerical columns (not string columns)
        if pd.api.types.is_numeric_dtype(df[col]):
            # Ensure median is calculated only on numerical values (ignore strings)
            df[col] = df[col].apply(pd.to_numeric, errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Save the cleaned dataset to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved: {output_path}")

    return df


if __name__ == "__main__":
    # Define input and output file paths
    RAW_DATA = "data/chlorella_raw.csv"
    CLEAN_DATA = "data/chlorella_cleaned.csv"
    
    # Run the preprocessing function
    preprocess_algae_data(RAW_DATA, CLEAN_DATA)
