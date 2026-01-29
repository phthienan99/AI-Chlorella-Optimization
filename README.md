# AI-Chlorella-Optimization  
### AI-assisted Pareto Decision Support for Nutrient and Biomass Trade-offs in Chlorella vulgaris Aquaculture Wastewater Treatment

## Overview
This repository provides the computational framework and supplementary materials supporting the research presented at **CITA 2026**, focusing on **AI-driven multi-objective optimization** of *Chlorella vulgaris* cultivation in aquaculture wastewater.

The framework integrates:
- Machine Learning regression models (Random Forest / XGBoost)
- Multi-objective evolutionary optimization (NSGA-II)
- Pareto-front analysis for sustainable decision-making

The goal is to identify optimal trade-offs between **nutrient removal efficiency** and **biomass productivity**, enabling scalable and environmentally safe aquaculture wastewater treatment.

---

## Key Research Objectives
- Maximize **Nitrogen (N)** and **Phosphorus (P)** removal efficiencies
- Enhance **COD reduction**
- Maximize **biomass yield** of *Chlorella vulgaris*
- Identify **Pareto-optimal operating conditions** using NSGA-II

---

## Scientific Contributions
- AI-assisted surrogate modeling for biological wastewater systems to predict multiple outputs (N removal, COD reduction, biomass yield)
- An applications of **NSGA-II–based Pareto optimization** applied to *Chlorella vulgaris* aquaculture wastewater treatment
- Data-driven insight into trade-offs between remediation efficiency and biomass growth

---

## Project Structure
The repository follows a standard data science layout:
AI-Chlorella-Optimization/
│
├── data/
│ ├── chlorella_raw.csv               # Raw experimental data
│ └── chlorella_cleaned.csv           # Cleaned and processed dataset for training
│
├── src/
│ ├── preprocess.py                  # Data cleaning and imputation
│ ├── train_model.py                 # Multi-output ML + NSGA-II optimization
│ └── visualize_pareto.py            # Pareto front visualization
│
├── results/
│ ├── models/                        # Saved machine learning models
│ ├── pareto/                        # Pareto front data from optimization
│ │ └── pareto_front.csv             # Pareto front of optimal solutions
│ └── figures/                       # Figures for results and publications
│
├── requirements.txt                 # Required dependencies for environment setup
└── README.md                        # Project overview and instructions


---

## Methodology
1. **Data preprocessing**
   - Missing-value imputation for incomplete records
   - Standardization of input features to ensure uniformity in model performance.
2. **Multi-output regression**
   - Random Forest and XGBoost are trained to predict:
        Nitrogen (N) removal efficiency
        Phosphorus (P) removal efficiency
        COD reduction efficiency
        Biomass yield
    These models are used to develop surrogate models that can predict the outcomes based on experimental conditions.
3. **NSGA-II optimization**
   - Using NSGA-II (Non-dominated Sorting Genetic Algorithm II), the framework identifies Pareto-optimal solutions that balance competing objectives:
        Maximize N removal efficiency
        Maximize COD removal efficiency
        Maximize biomass yield
The optimization identifies the best trade-offs between these objectives and provides a spectrum of feasible solutions.
4. **Pareto front analysis**
   - Pareto front analysis is performed to visually explore the trade-offs between nutrient recovery and biomass yield.
   - The generated Pareto front guides decision-making for sustainable treatment strategies.

---

## Getting Started
To replicate this study, first ensure you have a Python environment set up.

1. **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/AI-Chlorella-Optimization
    cd AI-Chlorella-Optimization
    ```
2. **Create & Activate the virtual environment:**
    *   python -m venv venv
    *   Windows: `venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the main processing script:**
    ```bash
    python src/preprocess.py
    ```
5.  **Train model + generate Pareto front**
    ```bash
    python src/train_model.py
    ```
6.  **Visualize results**
    ```bash
    python src/visualize_pareto.py
    ```

## Data Sources
The dataset used in this study was compiled and processed from multiple peer-reviewed
publications and publicly available repositories, including:
* Mendeley Data
* MDPI journals (e.g., *Water*, *Toxics*, *Algal Research*)
* Elsevier journals
* National and international aquaculture-related studies

Only processed and derived data are provided in this repository.
Original raw data remain available in the respective source publications.
All data are used strictly for academic research purposes.


## Relevance & Impact

This work supports:
*   Sustainable aquaculture management
*   Circular bioeconomy strategies
*   Data-driven environmental policy and sustainable resource management (EPA, DOE-aligned)
*   The framework is designed for scalability, transparency, and reproducibility.

## Contributing & Contact
We welcome collaboration and feedback. If you are interested in this research area or potential partnerships, please reach out via [phthienan99@gmail.com] or connect on [https://www.linkedin.com/in/phthienan99/].

---
Developed by [AnPham] | Presented at CITA 2026.