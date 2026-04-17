from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Transformer opcional para reproducir parte del feature engineering
    realizado en PB-08.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "PurchDate" in X.columns:
            X["PurchDate"] = pd.to_datetime(X["PurchDate"], errors="coerce")
            X["purchase_year"] = X["PurchDate"].dt.year
            X["purchase_month"] = X["PurchDate"].dt.month
            X["purchase_day"] = X["PurchDate"].dt.day
            X["purchase_dow"] = X["PurchDate"].dt.dayofweek

        if {"VehOdo", "VehicleAge"}.issubset(X.columns):
            X["odo_per_year"] = X["VehOdo"] / (X["VehicleAge"] + 1)

        if {"WarrantyCost", "VehBCost"}.issubset(X.columns):
            X["warranty_cost_ratio"] = X["WarrantyCost"] / (X["VehBCost"] + 1)

        if {"VehicleAge", "VehOdo"}.issubset(X.columns):
            X["age_x_odo"] = X["VehicleAge"] * X["VehOdo"]

        if "VehicleAge" in X.columns:
            X["vehicle_age_group"] = pd.cut(
                X["VehicleAge"],
                bins=[-1, 2, 5, 10, 30],
                labels=["nuevo", "semi_nuevo", "usado", "antiguo"]
            )

        for col in ["VehOdo", "VehBCost", "WarrantyCost"]:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                if (X[col] >= 0).all():
                    X[f"{col}_log1p"] = np.log1p(X[col])

        return X


def get_column_groups(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target], errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocessor


def save_artifact(obj, path: str) -> None:
    joblib.dump(obj, path)


def load_artifact(path: str):
    return joblib.load(path)