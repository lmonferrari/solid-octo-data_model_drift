import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from model import Model

if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(basedir)

    production_dir = "production"
    artifacts_dir = "artifacts"
    model_subdir = "model"
    data_dir = "data"
    dataset_name = "dataset.csv"
    scaler_name = "scaler.pkl"
    data_path = os.path.join(parent_dir, data_dir)

    execution_id = str(len(os.listdir(os.path.join(parent_dir, artifacts_dir))) + 1)
    model_path = os.path.join(parent_dir, artifacts_dir, execution_id, model_subdir)
    scaler_path = os.path.join(parent_dir, artifacts_dir, execution_id, scaler_name)
    production_path = os.path.join(parent_dir, production_dir)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(production_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(data_path, dataset_name), sep=";")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = Model.data_split(X, y)
    X_train_scaled, X_test_scaled, scaler = Model.data_scaler(X_train, X_test)
    Model.scaler_save(scaler, scaler_path)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    model_instance = Model(rf_model)
    model_instance.train(X_train_scaled, y_train)

    score, _ = model_instance.evaluate(X_test_scaled, y_test)
    print(f"Accuracy: {score * 100:.2f}%")

    model_instance.save_model(model_path)

    model_instance.save_model(production_path)
    Model.save_scaler(scaler, os.path.join(production_path, "scaler.pkl"))
