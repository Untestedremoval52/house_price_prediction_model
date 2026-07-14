import pandas as pd
import os
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df
def inspect_dataset(df):
    print("=====Dataset Inspection=====")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nMissing Values")
    print(df.isnull().sum())
def remove_unnecessary_columns(df):
    columns_to_remove = ["Unnamed: 0", "id"]
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df
def convert_date(df):
    df["date"] = pd.to_datetime(df["date"])
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_day"] = df["date"].dt.day
    df = df.drop(columns=["date"], errors='ignore')
    return df
def handle_missing_values(df):
    numerical_columns = ["bedrooms", "bathrooms"]
    for column in numerical_columns:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)
    return df
def save_processed_dataset(df, output_path):
    df.to_csv(output_path, index=False)
    print("=====PROCESSED DATASET SAVED SUCCESSFULLY=====")
    print(f"Location : {output_path}")
    print(f"Rows     : {df.shape[0]}")
    print(f"Columns  : {df.shape[1]}")
def verify_processed_dataset(df):
    print("=====PROCESSED DATASET VERIFICATION=====")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nRemaining Missing Values:")
    print(df.isnull().sum().sum())
    print("\nCurrent Columns:")
    for column in df.columns:
        print(f"✓ {column}")
def engineer_features(df):
       df["house_age"] = df["sale_year"] - df["yr_built"]
       df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
       return df