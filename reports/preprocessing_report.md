# Preprocessing Report

## House Price Prediction using Linear Regression

---

## Objective

Prepare the King County Housing Dataset for machine learning by cleaning, transforming, and engineering features while also preserving useful predictive information.

---

## Preprocessing Pipeline (in order)

1. Load the raw dataset (`data/raw/kc_house_data_NaN.csv`).
2. Remove all the non-predictive columns (`Unnamed: 0`, `id`).
3. Convert `date` into its numeric parts (`sale_year`, `sale_month`, `sale_day`).
4. Engineer all the new features (`house_age`, `was_renovated`).
5. Handle all the missing values (`bedrooms`, `bathrooms`) via median imputation.
6. Save the cleaned dataset (`data/processed/kc_house_data_cleaned.csv`).

> Note: `zipcode` is left as it is here and **one-hot encoded later at the modeling stage** (see `model_report.md`), because here, the encoding is a modeling concern, not a cleaning one.

---

## Decision Table

| Feature | Action | Reason |
|---------|--------|--------|
| Unnamed: 0 | Remove | CSV index generated during export |
| id | Remove | Property identifier; not predictive |
| date | Convert | Split into `sale_year`, `sale_month`, `sale_day`; drop original |
| bedrooms | Median imputation | 13 missing values |
| bathrooms | Median imputation | 10 missing values |
| yr_built | Keep | Used to derive `house_age` |
| yr_renovated | Keep | Used to derive `was_renovated` |
| zipcode | Keep (encode later) | Location; one-hot encoded at modeling stage |
| price | Keep | Target variable |
| all other columns | Keep | Structural / geographic / neighborhood signals |

---

## Missing Value Handling

Only two numerical features had missing values:

| Feature | Missing | Strategy |
|---------|--------:|----------|
| bedrooms | 13 | Median imputation |
| bathrooms | 10 | Median imputation |

**Why median:** it is robust to the outliers present and preserves the distribution better than the mean. Since less than 0.1% of values were missing, this minimizes the overall information loss without discarding any of the valid rows.

---

## Date Processing

The `date` column was stored as a string, unusable by a linear model. It was then converted to datetime, then split into `sale_year`, `sale_month`, and `sale_day`, and the original column was dropped. This preserves the overall temporal information in its numeric form.

---

## Feature Engineering

Two new features were created from existing columns:

- **`house_age`** = `sale_year − yr_built` — a house's age at sale, more directly meaningful than a raw build year.
- **`was_renovated`** = `1 if yr_renovated > 0 else 0` — collapses a mostly-empty year column into a clean binary signal.

---

## Final Processed Dataset

- **File:** `data/processed/kc_house_data_cleaned.csv`
- **Rows:** 21,613
- **Columns:** 24 (target `price` + 23 predictors)
- **Missing values remaining:** 0

The processed dataset is ready for modeling.

---

## Status

- Dataset audit is completed.
- Preprocessing pipeline is implemented and, thus, executed.
- Processed dataset is then saved and verified.

---
