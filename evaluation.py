import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, r2_score)
def calculate_mse(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    return mse
def calculate_rmse(mse):
    return np.sqrt(mse)
def calculate_r2(y_test, predictions):
    return r2_score(y_test, predictions)
def actual_vs_predicted_plot(y_test, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.2, s=20)
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    plt.plot( [min_value, max_value], [min_value, max_value], color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
    plt.legend()
    plt.xlabel("Actual House Prices")
    plt.ylabel("Predicted House Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(True)
    plt.savefig("images/model_evaluation/actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
def residual_plot(y_test, predictions):
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    plt.xlabel("Predicted House Prices")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/model_evaluation/residual_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()