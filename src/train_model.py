import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# =====================================================
# 1. TRAIN MULTI-OUTPUT RANDOM FOREST
# =====================================================

def train_multioutput_model(data_path, model_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError("Cleaned dataset not found")

    df = pd.read_csv(data_path)

    # Ensure there are no missing values in target columns before training
    if df.isnull().sum().any():
        print("[INFO] Missing values detected in target columns. Handling...")
        df = df.dropna(subset=["N Removal Efficiency (%)", "P Removal Efficiency (%)", "COD rem (%)", "Biomass Yield (g/L)"])

    # -------- INPUT FEATURES (decision variables) --------
    features = [
        "pH",
        "Light (μmol/m²/s)",
        "Temp (°C)",
        "TN₀ (mg/L)",
        "TP₀ (mg/L)",
        "COD₀",
        "Cultivation Time (days)"
    ]

    # -------- OUTPUTS (objectives) --------
    targets = [
        "N Removal Efficiency (%)",
        "P Removal Efficiency (%)",
        "COD rem (%)",
        "Biomass Yield (g/L)"
    ]

    df_ml = df[features + targets].dropna()  # Drop rows with missing feature values
    print("[DEBUG] df_ml shape:", df_ml.shape)
    print(df_ml.head())

    X = df_ml[features]
    Y = df_ml[targets]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    model = MultiOutputRegressor(rf)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    r2_scores = {
        targets[i]: r2_score(Y_test.iloc[:, i], Y_pred[:, i])
        for i in range(len(targets))
    }

    print("\n=== Multi-output RF performance (R²) ===")
    for k, v in r2_scores.items():
        print(f"{k}: {v:.3f}")

    # Save model for later use
    joblib.dump(model, model_path)
    print(f"\n[SUCCESS] Model saved → {model_path}")

    return model, features, targets


# =====================================================
# 2. NSGA-II OPTIMIZATION
# =====================================================

class ChlorellaOptimizationProblem(Problem):

    def __init__(self, model):
        super().__init__(
            n_var=7,
            n_obj=4,
            xl=np.array([6.5, 80, 20, 20, 2, 150, 3]),
            xu=np.array([8.5, 200, 32, 80, 10, 600, 10])
        )
        self.model = model

    def _evaluate(self, X, out, *args, **kwargs):
        # Predict outputs using the trained model
        preds = self.model.predict(X)

        # NSGA-II minimizes, so negate the values to maximize them
        out["F"] = -preds


def run_nsga2(model):
    problem = ChlorellaOptimizationProblem(model)

    algorithm = NSGA2(
        pop_size=80,
        eliminate_duplicates=True
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 150),
        seed=42,
        verbose=True
    )

    pareto_X = res.X
    pareto_F = -res.F  # Convert the output back to maximization

    # Convert to DataFrame for saving
    pareto_df = pd.DataFrame(
        np.hstack([pareto_X, pareto_F]),
        columns=[
            "pH", "Light", "Temp", "TN0", "TP0", "COD0", "Time",
            "N_rem", "P_rem", "COD_rem", "Biomass"
        ]
    )

    # Create directory for saving results
    os.makedirs("results/pareto", exist_ok=True)
    pareto_df.to_csv("results/pareto/pareto_front.csv", index=False)

    print("\n[SUCCESS] Pareto front saved → results/pareto/pareto_front.csv")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    DATA = "data/chlorella_cleaned.csv"
    MODEL_OUT = "results/models/multioutput_rf.pkl"

    # Ensure results directories exist
    os.makedirs("results/models", exist_ok=True)

    # Train model and perform optimization
    model, features, targets = train_multioutput_model(DATA, MODEL_OUT)
    run_nsga2(model)
