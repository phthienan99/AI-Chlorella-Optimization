import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# Plot settings (journal-quality)
# =========================
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15
})

def plot_pareto_front(pareto_path, save_dir="results/figures"):
    if not os.path.exists(pareto_path):
        raise FileNotFoundError(f"Pareto file not found: {pareto_path}")

    os.makedirs(save_dir, exist_ok=True)

    # Load Pareto data
    df = pd.read_csv(pareto_path)
    print("[INFO] Pareto front loaded:", df.shape)

    # =========================
    # Figure 1: Biomass vs N removal
    # =========================
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="N_rem",
        y="Biomass",
        hue="COD_rem",
        palette="viridis",
        s=70,
        edgecolor="black"
    )
    plt.xlabel("Nitrogen Removal Efficiency (%)")
    plt.ylabel("Biomass Yield (g/L)")
    plt.title("Pareto Front: Biomass vs Nitrogen Removal")
    plt.legend(title="COD Removal (%)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pareto_biomass_vs_N.png", dpi=300)
    plt.close()

    # =========================
    # Figure 2: Biomass vs COD removal
    # =========================
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="COD_rem",
        y="Biomass",
        hue="N_rem",
        palette="plasma",
        s=70,
        edgecolor="black"
    )
    plt.xlabel("COD Removal Efficiency (%)")
    plt.ylabel("Biomass Yield (g/L)")
    plt.title("Pareto Front: Biomass vs COD Removal")
    plt.legend(title="N Removal (%)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pareto_biomass_vs_COD.png", dpi=300)
    plt.close()

    print("[SUCCESS] Pareto plots saved to results/figures/")

if __name__ == "__main__":
    PARETO_FILE = "results/pareto/pareto_front.csv"
    plot_pareto_front(PARETO_FILE)
