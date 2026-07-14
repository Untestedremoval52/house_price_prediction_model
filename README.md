# House Price Prediction — Linear Regression

Predicting the house prices in King County using multiple linear regression model, built an end-to-end with a clean preprocessing → EDA → modeling → evaluation pipeline.

---

## Overview

This project trains a linear regression model to predict house sale prices from features like living area, grade, location, etc. It covers the full ML workflow: data cleaning, exploratory analysis, feature engineering, categorical encoding, model training, evaluation, and cross-validation.

---

## Results

| Metric                                 | Value                      |
| -------------------------------------- | -------------------------- |
| **R² Score**                    | **0.8068**           |
| **RMSE**                         | **$170,900.96**      |
| **Cross-Validated R² (5-fold)** | **0.8000 ± 0.0190** |

The biggest accuracy gain came from **one-hot encoding `zipcode`** (treating location as categories rather than si,ple numbers), which lifted R² from 0.70 intially to 0.81.

---

## Dataset

- **Source:** King County House Sales dataset
- **Size:** 21,613 houses, 24 features after preprocessing
- **Target:** `price`

---

## Project Structure

```
house_price_prediction_model/
├── data/
│   ├── raw/                # original dataset
│   └── processed/          # cleaned dataset
├── src/
│   ├── preprocessing.py    # cleaning + feature engineering
│   ├── eda.py              # correlation analysis
│   └── train_model.py      # split, train, save
├── evaluation.py           # metrics + plots
├── main.py                 # runs the full pipeline
├── models/                 # saved model (.pkl)
├── images/                 # heatmap, actual-vs-predicted, residuals
├── reports/                # dataset, preprocessing, model, evaluation, journal
└── requirements.txt
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

This cleans the data, runs the EDA, trains the model, evaluates it, saves the model to `models/`, and writes plots to `images/`.

---

## Workflow

1. **Preprocessing** — drop all the identifiers, splits the `date`, engineers the `house_age` & `was_renovated`, and finally median-impute missing values.
2. **EDA** — correlation heatmap helps to identify all the key price drivers (`sqft_living`, `grade`, `sqft_above`).
3. **Encoding** — one-hot encode `zipcode` so that the location is treated as categories.
4. **Modeling** — `LinearRegression` on an 80/20 split.
5. **Evaluation** — MSE, RMSE, R², actual-vs-predicted and residual plots.
6. **Validation** — 5-fold shuffled cross-validation to confirm the stability of the model.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib

---

## Reports

Detailed write-ups are in [`reports/`](reports/): dataset audit, preprocessing, model, evaluation, and a development journal documenting the full build.

---

## Author

**Garv Bhardwaj**

---
