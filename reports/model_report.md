# Model Report

## House Price Prediction using Linear Regression

---

## Objective

Develop a Linear Regression model to predict the house prices using a preprocessed King County Housing dataset.

---

## Model Selection

**Algorithm:** Linear Regression

**Reason:** simple, interpretable, well-suited for a continuous target prediction and has an excellent baseline.

---

## Dataset Used

- **Input:** `data/processed/kc_house_data_cleaned.csv`
- **Target variable:** `price`

---

## Feature Handling

- After performing cleaning and feature engineering, the dataset has in total 23 predictor columns plus the target.
- `zipcode` is the **one-hot encoded** at the modeling stage (`pd.get_dummies`, `drop_first=True`), which expands the current input to **91 features**. This lets the model treat each location as its own category rather than a numeric magnitude.
- **Features (X):** all columns except `price` (after encoding).
- **Target (y):** `price`.

---

## Train-Test Split

- Training set: **80%** → 17,290 samples
- Testing set: **20%** → 4,323 samples
- `random_state = 42` for reproducibility.

An 80/20 split provides enough data to train while also reserving an independent test set for honest evaluation.

---

## Model Training

The model was trained with scikit-learn's `LinearRegression` on **17,290 samples × 91 features**. Training completed successfully and the model was used to predict on the 4,323 unseen test houses.

---

## Results

| Metric | Value |
|--------|-------|
| R² Score | **0.8068** |
| RMSE | **$170,900.96** |
| Cross-Validated R² (5-fold) | **0.8000 ± 0.0190** |

See `evaluation_report.md` for the full breakdown.

---

## Model Persistence

The trained model was saved to `models/linear_regression_model.pkl` (via `joblib`), enabling the reuse for future predictions without any retraining.

---

## Status

- Dataset preprocessing is completed.
- Model implementation is completed.
- Evaluation is completed.

---
