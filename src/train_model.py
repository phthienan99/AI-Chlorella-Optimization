import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_algae_model(data_path, model_save_path):
    """
    Train a Random Forest model to predict heavy metal removal efficiency.
    This demonstrates advanced predictive modeling capabilities for NIW evidence.
    """
    if not os.path.exists(data_path):
        print(f"[ERROR] Cleaned data not found at {data_path}")
        return

    # Step 1: Load the processed dataset
    df = pd.read_csv(data_path)
    print("[INFO] Cleaned dataset loaded successfully.")

    # Step 2: Define Features (X) and Target (y)
    # We use pH and temperature to predict removal_efficiency
    features = ['pH', 'temperature', 'initial_metal_conc']
    target = 'removal_efficiency'
    
    X = df[features]
    y = df[target]

    # Step 3: Split data into Training and Testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Data split complete. Training samples: {len(X_train)}")

    # Step 4: Initialize and Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")

    # Step 5: Model Evaluation
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    
    print(f"--- Model Performance Metrics ---")
    print(f"R-squared Score (R2): {r2:.4f} (Closer to 1.0 is better)")

    # Step 6: Save the trained model for deployment
    joblib.dump(model, model_save_path)
    print(f"[SUCCESS] Trained model saved to: {model_save_path}")

if __name__ == "__main__":
    DATA_INPUT = 'data/chlorella_cleaned.csv'
    MODEL_OUTPUT = 'data/algae_rf_model.pkl'
    
    train_algae_model(DATA_INPUT, MODEL_OUTPUT)