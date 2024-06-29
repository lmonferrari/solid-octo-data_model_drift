import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from model import Model


def data_drift(X, drift_factor):
    drifted_X = X + np.random.normal(0, drift_factor, X.shape)
    return drifted_X


def model_monitor_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


if __name__ == "__main__":
    np.random.seed(41)
    basedir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(basedir)

    data_dir = "data"
    dataset_name = "dataset.csv"
    data_path = os.path.join(parent_dir, data_dir)

    production_dir = "production"
    production_path = os.path.join(parent_dir, production_dir)

    df = pd.read_csv(os.path.join(data_path, dataset_name), delimiter=";")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    _, X_test, _, y_test = Model.data_split(X, y)

    model = Model.load_model(production_path)
    scaler = Model.load_scaler(os.path.join(production_path, "scaler.pkl"))

    X_test_scaled = scaler.transform(X_test)

    X_test_drifted = data_drift(X_test_scaled, drift_factor=0.7)

    drifted_predictions = model.predict(X_test_drifted)
    drifted_accuracy = accuracy_score(y_test, drifted_predictions)

    pre_drift_accuracy = round(
        model_monitor_performance(model, X_test_scaled, y_test), 2
    )
    post_drift_accuracy = round(
        model_monitor_performance(model, X_test_drifted, y_test), 2
    )

print(f"Before data drift {pre_drift_accuracy}")
print(f"After data drift {post_drift_accuracy}")
