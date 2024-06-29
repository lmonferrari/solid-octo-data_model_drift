import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)


class Model:
    def __init__(self, model):
        self.model = model

    @classmethod
    def data_split(cls, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=4
        )
        return X_train, X_test, y_train, y_test

    @classmethod
    def data_scaler(cls, X_train, X_test):
        sdf = StandardScaler()
        X_train_scaled = sdf.fit_transform(X_train)
        X_test_scaled = sdf.transform(X_test)

        return X_train_scaled, X_test_scaled, sdf

    @classmethod
    def save_scaler(cls, scaler, scaler_path):
        joblib.dump(scaler, scaler_path)

    @classmethod
    def load_scaler(cls, scaler_path):
        return joblib.load(scaler_path)

    @classmethod
    def load_model(cls, model_path):
        model = joblib.load(f"{model_path}/model.pkl")
        return cls(model)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average="weighted")
        return score, recall

    def save_model(self, model_path):
        joblib.dump(self.model, f"{model_path}/model.pkl")
