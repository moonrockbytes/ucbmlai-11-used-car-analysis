# Used Car Price Analysis

An end-to-end regression analysis of 426K used car listings to identify what factors drive vehicle prices — and to give a used car dealership actionable inventory and pricing guidance.

---

## Project Structure

```
ucbmlai-11-used-car-analysis/
├── used_car_price_analysis.ipynb   # Main analysis notebook
├── data/
│   ├── raw/
│   │   └── vehicles.csv            # Original dataset (426,880 listings)
│   └── processed/
│       └── vehicles_processed.csv  # Cleaned and feature-engineered dataset
├── images/
│   ├── kurt.jpeg
│   └── crisp.png
└── README.md
```

---

## Setup Instructions

### 1. Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyterlab
```

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scipy` | Statistical tests |
| `scikit-learn` | Preprocessing, modelling, evaluation |
| `jupyterlab` | Notebook environment |

### 3. Launch JupyterLab

```bash
jupyter lab
```

Then open `used_car_price_analysis.ipynb` from the file browser.

---

## Methodology

The analysis follows the **CRISP-DM** framework:

| Phase | What Was Done |
|---|---|
| Business Understanding | Reframed as a supervised regression problem — predict listing price from vehicle attributes |
| Data Understanding | EDA on distributions, missing values, and categorical breakdowns across 18 features |
| Data Preparation | Price filtering, median/unknown imputation, feature engineering, train/test split |
| Modelling | Evaluated four regression models with cross-validation and grid search tuning |
| Evaluation | Compared models on RMSE and R²; analysed feature coefficients for interpretability |
| Deployment | Plain-language recommendations for inventory acquisition and pricing strategy |

---

## Data Processing

### Source Data

- **File**: `data/raw/vehicles.csv`
- **Size**: 426,880 rows × 18 columns
- **Features include**: price, year, manufacturer, model, condition, odometer, fuel, transmission, drive type, cylinders, title status, paint color, state

### Columns Removed

| Column | Reason |
|---|---|
| `id` | Identifier — not predictive |
| `VIN` | Identifier — not predictive |
| `region` | Redundant with `state` |
| `size` | 71.77% missing — too sparse to impute reliably |

### Row Filtering

Listings were restricted to prices between **$500 and $150,000**:
- Below $500 — scrap/non-functional entries
- Above $150,000 — auction anomalies and data errors
- **42,290 rows removed (~9.9%)**
- **Remaining: 384,590 rows**

### Imputation

| Column Type | Columns | Strategy |
|---|---|---|
| Numeric | `year`, `odometer` | Filled with column **median** (distributions are right-skewed) |
| Categorical | `manufacturer`, `model`, `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `type`, `paint_color` | Filled with `'unknown'` |

### Feature Engineering

| Feature | Description |
|---|---|
| `vehicle_age` | `current_year − year` — captures depreciation more directly than raw year |
| `log_odometer` | `log1p(odometer)` — compresses right-skewed mileage distribution |
| `log_price` | `log1p(price)` — model target; normalises the heavily skewed price distribution |
| `cylinders_num` | Numeric extracted from strings (e.g. `"6 cylinders"` → `6`) via regex; missing filled with median |

High-cardinality categorical columns were reduced by retaining only the **top-10 most frequent** categories per feature; all others were recoded as `'other'`.

### Train / Test Split

| Set | Rows | Share |
|---|---|---|
| Training | 307,672 | 80% |
| Test | 76,918 | 20% |

Processed data saved to `data/processed/vehicles_processed.csv`.

---

## Model Selection

Four regression models were evaluated, all using a consistent `sklearn` Pipeline:

**Preprocessing pipeline**
- Numeric features (`vehicle_age`, `log_odometer`, `cylinders_num`): `StandardScaler`
- Categorical features (9 columns): `OneHotEncoder(handle_unknown='ignore')`
- Composed via `ColumnTransformer`

**Models**

| Model | Tuning |
|---|---|
| Linear Regression | Baseline — no hyperparameters |
| Ridge (L2) | Alpha grid: `[0.01, 0.1, 1, 10, 100]` — 5-fold CV |
| Lasso (L1) | Alpha grid: `[0.01, 0.1, 1, 10, 100]` — 5-fold CV |
| Ridge + Polynomial Features | 3-parameter grid: degree `[1, 2]` × alpha `[0.01, 0.1, 1, 10, 100]` × fit_intercept `[True, False]` (20 combinations) — 5-fold CV |

**Best hyperparameters found**

| Model | Best Parameters |
|---|---|
| Ridge | alpha = 1 |
| Lasso | alpha = 0.01 |
| Ridge + Polynomial | degree = 2, alpha = 10, fit_intercept = True |

---

## Evaluation

All metrics are on the **log-price** scale. RMSE values are therefore in log-dollar units; a lower value means predictions are closer to true prices on a proportional basis.

| Model | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|
| Linear Regression | 0.6676 | 0.6701 | 0.4531 |
| Ridge Regression | 0.6676 | 0.6701 | 0.4531 |
| Lasso Regression | 0.6906 | 0.6937 | 0.4139 |
| **Ridge + Polynomial (degree 2)** | **0.5706** | **0.5710** | **0.6029** |

The **Ridge + Polynomial** model is the clear winner — it explains approximately **60% of the variation in used car prices** and generalises well (CV RMSE and test RMSE are nearly identical, indicating no overfitting).

---

## Key Findings

The following is a prioritised list of insights for the used car dealership:

1. **Buy newer vehicles** — Vehicle age is the single strongest price driver. Every additional year of age reduces expected price meaningfully. Prioritise acquiring recent model-year inventory.

2. **Low mileage commands a premium** — Odometer reading is the second-strongest driver. The relationship is non-linear: the first 30K miles of wear depresses value more than miles accumulated later. Seek sub-50K listings where possible.

3. **Only acquire clean-title vehicles** — Title status has an outsized impact. Parts-only and salvage titles are the two most damaging attributes in the entire dataset. The price penalty buyers apply to these vehicles exceeds any savings on acquisition cost.

4. **V6 and V8 engines fetch more** — More cylinders consistently correlates with a higher price. This reflects buyer demand for the truck and SUV segments, which dominate the upper price tiers.

5. **Stock AWD and 4WD models** — Four-wheel and all-wheel drive vehicles are priced above front-wheel drive equivalents. Buyers pay for versatility and perceived safety.

6. **Condition matters — avoid fair and salvage stock** — Good and like-new condition vehicles command a clear premium. Fair and salvage condition listings are priced significantly lower by buyers, and the data confirms this pricing pressure.

7. **Do not pay a premium for paint colour or state of origin** — These features have minimal impact on price after controlling for age, mileage, and title. They are not worth factoring into acquisition decisions.

8. **Use the model as a pricing cross-check** — The model explains ~60% of price variation. If an asking price is well above the model's prediction for a given vehicle's age, mileage, and title, expect it to sit on the lot. The remaining ~40% of variation comes from factors not in the data (service history, interior condition, seller motivation).

9. **Retrain periodically** — Fuel prices, interest rates, and consumer demand shift. The model reflects market conditions at the time the data was collected and should be refreshed as new listings data becomes available.
