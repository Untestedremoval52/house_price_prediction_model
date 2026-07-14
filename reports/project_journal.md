
# Project Journal — House Price Prediction (Linear Regression)

Author: Garv Bhardwaj

Dataset: King County House Sales

---

## Overview

This journal documents how I built the house price prediction model end-to-end — the decisions I made during building phase, all the experiments that worked, and also the ones that didn't. I kept it because the whole *process* taught me more than the final number.

---

## Development Log

### 1. Dataset & Preprocessing

- King County dataset: 21,613 rows, 22 columns, target = `price`.
- Dropped `id` and `Unnamed: 0` (identifiers, no predictive value).
- Converted `date` → `sale_year`, `sale_month`, `sale_day`, then dropped the original.
- Handled missing values: `bedrooms` (13) and `bathrooms` (10) via **median imputation** (robust to outliers, minimal information loss).
- Saved the cleaned dataset to `data/processed/kc_house_data_cleaned.csv`.

### 2. Baseline Model

- `LinearRegression` on all features, 80/20 train-test split, `random_state=42`.
- **Baseline: R² = 0.7025, RMSE ≈ $212,074.**

### 3. Debugging

- Fixed an **indentation bug** where the metrics, model-save, and CSV-write ran 10× inside a print loop instead of just running once.
- Removed **stray import-time code** in `train_model.py` (which referenced to a non-existent file) and fixed a function that ignored its argument.
- Removed a duplicate function call and junk all the auto-imports.

### 4. EDA — Correlation Analysis

- Built a correlation heatmap, which became the strongest drivers of price: `sqft_living`, `grade`, `sqft_above`.
- Noticed that `zipcode` had near-zero correlation (it was being treated as a *number*), and a multicollinearity cluster among the `sqft_*` features.

### 5. Experiment — Log-transform the target (reverted)

- Hypothesis: log-transforming `price` would fix its right-skew and the funnel-shaped residuals.
- Result: **R² dropped to 0.5103, RMSE rose to ~$272k** — because `expm1` amplified a few high-end predictions into multi-million-dollar blowups which dominated this error.
- Decision: **reverted.** Key lesson — an improvement is a *hypothesis*; validate it by measurement and practical implementation, not just theory.

### 6. Improvement — Zipcode as categorical (one-hot)

- Encoded `zipcode` with `pd.get_dummies(drop_first=True)` so that the model treats location as categories, not magnitudes.
- **R² 0.7025 → 0.8068, RMSE $212,074 → $170,901.** Location was the missing signal all along.

### 7. Feature Engineering

- Added `house_age` (`sale_year - yr_built`) and `was_renovated` (`yr_renovated > 0`).
- Negligible R² change — learned that a feature which is a **linear combination** of existing features adds no new information to a linear model (`house_age` was recoverable from columns already present).

### 8. Validation — Cross-validation

- Ran **5-fold shuffled cross-validation** to confirm my model's stability across different data splits, rather than trusting a single lucky split.
- _(Exact mean CV R² and standard deviation to be inserted from the run output.)_

---

## Final Results

| Metric    | Value                        |
| --------- | ---------------------------- |
| R² Score | **0.8068**             |
| RMSE      | **$170,901**           |
| Model     | Linear Regression (multiple) |

---

## Key Learnings

- **Measure, don't assume** — the log-transform *should* have helped as per theory but it does hurt in practice.
- **Categorical vs numeric encoding** matters enormously —  `zipcode` was the single biggest win.
- **Linear models can't benefit from linear-combination features** — feature engineering pays off more when working with non-linear models.
- **Cross-validation** turns a single-split number into a trustworthy estimates.

---
