import joblib
import pandas as pd
import numpy as np
from src.eda import corr
from src.train_model import (load_processed_dataset, save_model, split_features_target, split_train_test, train_linear_regression, generate_predictions)
from src.preprocessing import (convert_date, load_dataset, inspect_dataset, remove_unnecessary_columns, handle_missing_values, save_processed_dataset, verify_processed_dataset)
from evaluation import (calculate_mse, calculate_rmse, calculate_r2)
from evaluation import actual_vs_predicted_plot
from evaluation import residual_plot
from src.preprocessing import engineer_features
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
def main():
    df = load_dataset("data/raw/kc_house_data_NaN.csv")
    inspect_dataset(df)
    df = remove_unnecessary_columns(df)
    df = convert_date(df)
    df = engineer_features(df)
    df = handle_missing_values(df)
    save_processed_dataset(df, "data/processed/kc_house_data_cleaned.csv")
    verify_processed_dataset(df)
    processed_df = load_processed_dataset("data/processed/kc_house_data_cleaned.csv")
    corr(processed_df)
    processed_df = pd.get_dummies(processed_df, columns=["zipcode"], prefix="zip", drop_first=True)
    X, y = split_features_target(processed_df)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    print("CV R² per fold:", cv_scores)
    print(f"Mean CV R²: {cv_scores.mean():.4f}")
    print(f"Std Dev:    {cv_scores.std():.4f}")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = train_linear_regression(X_train, y_train)
    predictions = generate_predictions(model, X_test)
    print("=====HOUSE PRICE PREDICTIONS=====")
    print("First 10 Predicted Prices:\n")
    for i, prediction in enumerate(predictions[:10], start=1):
        print(f"House {i:2d}: ${prediction:,.2f}")
    print("=====LINEAR REGRESSION MODEL=====")
    print("Model trained successfully!")
    print("=====ACTUAL vs PREDICTED=====")
    comparison = pd.DataFrame({"Actual Price": y_test.values, "Predicted Price": predictions})
    comparison["Difference"] = (comparison["Predicted Price"]- comparison["Actual Price"])
    print("=====MODEL PREDICTION VERIFICATION=====")
    print(comparison.head(10))
    comparison.to_csv("reports/predictions.csv", index=False)
    save_model(model, "models/linear_regression_model.pkl")
    mse = calculate_mse(y_test, predictions)
    rmse = calculate_rmse(mse)
    print(f"Root Mean Squared Error : {rmse:,.2f}")
    r2 = calculate_r2(y_test, predictions)
    print(f"R² Score: {r2:.4f}")
    print("=====MODEL EVALUATION=====")
    print(f"Mean Squared Error : {mse:,.2f}")
    actual_vs_predicted_plot(y_test, predictions)
    residual_plot(y_test, predictions)
if __name__ == "__main__":
    main()