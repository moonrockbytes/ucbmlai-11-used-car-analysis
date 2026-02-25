# Used Car Price Analysis

An end-to-end regression analysis of 426K used car listings to identify what factors drive vehicle prices and give a used car dealership actionable inventory and pricing guidance.

---

## Key Findings

1. **Buy newer vehicles** — Vehicle age is the single strongest price driver.
2. **Low mileage commands a premium** — Seek sub-50K listings where possible.
3. **Only acquire clean-title vehicles** — Parts-only and salvage titles carry the largest price penalty in the dataset.
4. **V6 and V8 engines fetch more** — Higher cylinder counts consistently correlate with higher prices.
5. **Stock AWD and 4WD models** — These are priced above front-wheel drive equivalents.
6. **Avoid fair and salvage condition stock** — The price discount buyers apply outweighs acquisition savings.
7. **Paint colour and state of origin don't matter** — Minimal impact on price; not worth factoring into acquisition decisions.
8. **Use the model as a pricing cross-check** — If an asking price is well above the model's prediction for a vehicle's age, mileage, and title, expect it to sit on the lot.

---

## Methodology

The analysis follows the **CRISP-DM** framework — business understanding, data understanding, data preparation, modelling, evaluation, and deployment. See the notebook for full details.

---

## Data Processing

- Source: 426,880 used car listings with 18 attributes
- Removed listings priced below $500 or above $150,000 (scrap entries and anomalies)
- Imputed missing numeric values with the column median; missing categoricals with `'unknown'`
- Engineered features: `vehicle_age`, `log_odometer`, `log_price`, `cylinders_num`
- Final dataset: 384,590 rows split 80/20 into training and test sets

---

## Model Selection & Evaluation

Four models were evaluated — Linear Regression, Ridge, Lasso, and Ridge with Polynomial Features — all using a consistent sklearn Pipeline with StandardScaler and OneHotEncoder. Hyperparameters were tuned via 5-fold GridSearchCV.

| Model | Test RMSE | Test R² |
|---|---|---|
| Linear Regression | 0.6701 | 0.4531 |
| Ridge | 0.6701 | 0.4531 |
| Lasso | 0.6937 | 0.4139 |
| **Ridge + Polynomial (degree 2)** | **0.5710** | **0.6029** |

Ridge with Polynomial Features was the best model, explaining ~60% of price variation with no signs of overfitting.

---

## Setup Instructions

**Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyterlab
```

**Launch JupyterLab:**

```bash
jupyter lab
```

---

## Dealership Report

A plain-language report summarising the key findings and inventory recommendations is available for sharing with non-technical stakeholders.

- **[reports/used_car_dealership_report.md](reports/used_car_dealership_report.md)** — Markdown source
- To export to PDF: open in VS Code with the Markdown PDF extension, paste into Google Docs, or run `pandoc reports/used_car_dealership_report.md -o reports/used_car_dealership_report.pdf`

---

## Project Structure

```
ucbmlai-11-used-car-analysis/
├── used_car_price_analysis.ipynb   # Main analysis notebook
├── reports/
│   └── used_car_dealership_report.md  # Client-facing findings report
├── data/
│   ├── raw/
│   │   └── vehicles.csv            # Original dataset (426,880 listings)
│   └── processed/
│       └── vehicles_processed.csv  # Cleaned and feature-engineered dataset
├── images/
└── README.md
```
