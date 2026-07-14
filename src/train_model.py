import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
def load_processed_dataset(file_path):
    df = pd.read_csv(file_path)
    return df
def split_features_target(df):
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def generate_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions
def save_model(model, file_path):
    joblib.dump(model, file_path)
    print("\nModel saved successfully.")
