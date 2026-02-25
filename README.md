# Module 11 — What Drives the Price of a Car?

**UC Berkeley ML/AI Professional Certificate — Practical Application 11.1**

## Overview

This project applies the **CRISP-DM framework** to a used car pricing dataset (426K listings from Kaggle) to identify what factors make a used car more or less expensive. The goal is to provide actionable recommendations to a used car dealership about what attributes consumers value.

## Notebook

**[prompt_II.ipynb](prompt_II.ipynb)** — Main analysis notebook covering:

| CRISP-DM Phase | Contents |
|---|---|
| Business Understanding | Reframe as a supervised regression problem |
| Data Understanding | EDA: distributions, missing values, categorical breakdowns |
| Data Preparation | Filtering, imputation, feature engineering, train/test split |
| Modeling | Linear Regression, Ridge, Lasso, Ridge + Polynomial (3-param grid search) |
| Evaluation | Model comparison, feature importance, residual analysis |
| Deployment | Plain-language report with inventory recommendations |

## Dataset

`data/vehicles.csv` — 426,880 used car listings with 18 attributes (price, year, manufacturer, model, condition, odometer, fuel, transmission, drive, type, etc.)

## Key Findings

- **Vehicle age** and **odometer reading** are the strongest price drivers
- **Clean title** status commands a significant premium over salvage/rebuilt titles
- **4WD/AWD drive** type and **higher cylinder counts** correlate with higher prices
- The best model (Ridge + Polynomial features) achieves **R² ≈ 0.65** on held-out data

## Requirements

```
pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
```

Install with: `pip install pandas numpy matplotlib seaborn scipy scikit-learn`
