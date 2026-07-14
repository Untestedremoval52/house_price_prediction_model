# Evaluation Report

## House Price Prediction using Linear Regression

---

## Objective

Evaluate the final Linear Regression model (with one-hot encoded location zipcode) using standard regression metrics and cross-validation.

---

## Final Metrics (on the 20% hold-out test set)

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | ≈ 29,207,138,000 |
| Root Mean Squared Error (RMSE) | **$170,900.96** |
| R² Score | **0.8068** |

### Mean Squared Error (MSE)

MSE = (1/n) × Σ (Actual − Predicted)². It averages all the existing squared errors and is mainly useful for comparison purposes. Because prices are in hundreds of thousands, the squared value is very large; RMSE (its square root) is better interpretable version than MSE.

### Root Mean Squared Error (RMSE)

RMSE = √MSE = **$170,900.96**. On average, the model's predicted price differs from the actual price by about this amount — which is expressed in the same unit as the target, so it becomes directly interpretable.

### R² Score

R² = 1 − (SS_res / SS_tot) = **0.8068**. The model explains approximately **80.7%** of the variation in house prices of King County. The remaining ~19% is due to factors which are not captured in the data or are in non-linear relationships which a linear model cannot simply represent.

---

## Cross-Validation (5-fold, shuffled)

To confirm that the score is not solely the result of a single lucky split, 5-fold shuffled cross-validation was also performed after this:

- Per-fold R²: 0.8068, 0.7658, 0.8000, 0.8033, 0.8240
- **Mean CV R²: 0.8000**
- **Standard Deviation: 0.0190**

The very low spread across folds indicates that the model is **stable and reliable**, and is not dependent on a particular data split.

---

## Model Improvement Journey

| Stage | R² | RMSE |
|-------|-----|------|
| Baseline (raw features) | 0.7025 | $212,074 |
| Log-transform target (reverted) | 0.5103 | $272,096 |
| **Zipcode one-hot encoding (final)** | **0.8068** | **$170,901** |
| + Feature engineering (house_age, was_renovated) | 0.8068 | $170,901 |

The largest gain of this model came from **treating `zipcode` as a categorical feature (one-hot encoding)** rather than a number, which helped to unlock the location signal. The log-transform experiment was also tested, measured, and reverted as it worsened the score and overall metrics.

---

## Actual vs Predicted Visualization

Output: `images/model_evaluation/actual_vs_predicted.png`

The scatter plot shows the clear positive relationship along the diagonal shown — the model captures the overall pricing trend very well. The spread widens for very high-value homes, which is ,thus, expected for a linear model on the premium properties.

---

## Residual Plot

Output: `images/model_evaluation/residual_plot.png`

Residuals are generally centered around the zero-error line, indicating that there is no strong systematic bias. The spread increases for higher-priced houses (indicating mild heteroscedasticity), a common trait for the housing data and a known limitation of linear regression which could be addressed with the non-linear models.

---

## Conclusion

The final model is a strong, having stable baseline (R² as 0.81, cross-validated). Further improvement would come from the regularization (Ridge/Lasso, given the 91 features) or non-linear models such as Random Forest / XGBoost.

---

## Status

- Evaluation completed.

---
