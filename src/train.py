import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from preprocess import load_data, preprocess
from feature_engineering import create_features

df = load_data("data/printer_data.csv")
df = preprocess(df)
df = create_features(df)

X = df.drop("failure", axis=1)
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = XGBClassifier(n_estimators=200, max_depth=5)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", acc)
    mlflow.xgboost.log_model(model, "model")

joblib.dump(model, "model/model.pkl")
