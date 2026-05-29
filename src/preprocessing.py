"""Preprocesamiento y feature engineering para el dataset de subastas automotrices.

Este módulo es el contrato entre los datos crudos, los notebooks, la API y el
modelo final. El modelo guardado incluye `FeatureEngineeringTransformer`, por lo
que la API puede recibir columnas originales y el pipeline se encarga de crear
las variables derivadas antes de predecir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from src.config import TARGET


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza variables categóricas sin convertir nulos reales en texto."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        values = df[col].astype("object").where(df[col].notna(), np.nan)
        df[col] = (
            values.astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "<NA>": np.nan, "": np.nan})
        )
    return df


def safe_divide(numerator, denominator):
    """Divide evitando infinitos cuando el denominador es cero o nulo."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return np.where((denominator.notna()) & (denominator != 0), numerator / denominator, np.nan)


def parse_purchase_date(series: pd.Series) -> pd.Series:
    """Parsea PurchDate tanto si viene como timestamp Unix como si viene como fecha."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        median_value = values.dropna().median() if values.notna().any() else np.nan
        if median_value > 1_000_000_000:
            return pd.to_datetime(values, unit="s", errors="coerce")
        if median_value > 1_000_000:
            return pd.to_datetime(values, unit="ms", errors="coerce")
        return pd.to_datetime(values, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas desde el dataset crudo.

    La función no hace `fit`, no usa la variable objetivo y puede correrse en
    train, test, API o dashboard sin data leakage.
    """
    df = pd.DataFrame(df).copy()
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    df = normalize_text_columns(df)

    if "PurchDate" in df.columns:
        purch_dt = parse_purchase_date(df["PurchDate"])
        df["PurchYear"] = purch_dt.dt.year
        df["PurchMonth"] = purch_dt.dt.month
        df["PurchQuarter"] = purch_dt.dt.quarter
        df["PurchDayOfWeek"] = purch_dt.dt.dayofweek
        df = df.drop(columns=["PurchDate"])

    # Identificadores/códigos: se tratan como categorías, no como magnitudes.
    for col in ["BYRNO", "VNZIP1", "WheelTypeID", "IsOnlineSale"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .astype("Int64")
                .astype("object")
                .astype(str)
                .replace({"<NA>": np.nan, "NAN": np.nan})
            )

    # MMR igual a cero no representa precio real, sino precio no disponible.
    mmr_cols = [c for c in df.columns if c.startswith("MMR")]
    for col in mmr_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= 0, col] = np.nan

    if {"VehOdo", "VehicleAge"}.issubset(df.columns):
        vehicle_age_plus_one = pd.to_numeric(df["VehicleAge"], errors="coerce") + 1
        df["odo_per_year"] = safe_divide(df["VehOdo"], vehicle_age_plus_one)
        df["old_high_mileage_flag"] = (
            (pd.to_numeric(df["VehOdo"], errors="coerce") >= 85000)
            & (pd.to_numeric(df["VehicleAge"], errors="coerce") >= 5)
        ).astype(int)

    if {"VehBCost", "MMRAcquisitionAuctionAveragePrice"}.issubset(df.columns):
        df["cost_to_acq_auction_avg"] = safe_divide(df["VehBCost"], df["MMRAcquisitionAuctionAveragePrice"])
        df["acq_auction_margin"] = (
            pd.to_numeric(df["MMRAcquisitionAuctionAveragePrice"], errors="coerce")
            - pd.to_numeric(df["VehBCost"], errors="coerce")
        )

    if {"VehBCost", "MMRCurrentAuctionAveragePrice"}.issubset(df.columns):
        df["cost_to_current_auction_avg"] = safe_divide(df["VehBCost"], df["MMRCurrentAuctionAveragePrice"])
        df["current_auction_margin"] = (
            pd.to_numeric(df["MMRCurrentAuctionAveragePrice"], errors="coerce")
            - pd.to_numeric(df["VehBCost"], errors="coerce")
        )

    if {"WarrantyCost", "VehBCost"}.issubset(df.columns):
        df["warranty_to_cost"] = safe_divide(df["WarrantyCost"], df["VehBCost"])
        if "VehicleAge" in df.columns:
            df["warranty_per_vehicle_year"] = safe_divide(
                df["WarrantyCost"], pd.to_numeric(df["VehicleAge"], errors="coerce") + 1
            )

    if {"MMRCurrentAuctionAveragePrice", "MMRAcquisitionAuctionAveragePrice"}.issubset(df.columns):
        df["auction_avg_depreciation"] = (
            pd.to_numeric(df["MMRCurrentAuctionAveragePrice"], errors="coerce")
            - pd.to_numeric(df["MMRAcquisitionAuctionAveragePrice"], errors="coerce")
        )

    if {"MMRCurrentRetailAveragePrice", "MMRAcquisitionRetailAveragePrice"}.issubset(df.columns):
        df["retail_avg_depreciation"] = (
            pd.to_numeric(df["MMRCurrentRetailAveragePrice"], errors="coerce")
            - pd.to_numeric(df["MMRAcquisitionRetailAveragePrice"], errors="coerce")
        )

    if {"MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionAuctionAveragePrice"}.issubset(df.columns):
        df["acq_clean_avg_spread"] = (
            pd.to_numeric(df["MMRAcquisitionAuctionCleanPrice"], errors="coerce")
            - pd.to_numeric(df["MMRAcquisitionAuctionAveragePrice"], errors="coerce")
        )

    if {"MMRCurrentAuctionCleanPrice", "MMRCurrentAuctionAveragePrice"}.issubset(df.columns):
        df["current_clean_avg_spread"] = (
            pd.to_numeric(df["MMRCurrentAuctionCleanPrice"], errors="coerce")
            - pd.to_numeric(df["MMRCurrentAuctionAveragePrice"], errors="coerce")
        )

    if mmr_cols:
        df["mmr_missing_count"] = df[mmr_cols].isna().sum(axis=1)

    return df


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Transformer sklearn serializable para usar dentro del pipeline final."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return prepare_features(pd.DataFrame(X).copy())


def split_X_y(df: pd.DataFrame):
    y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def get_column_groups(X_prepared: pd.DataFrame):
    categorical_cols = X_prepared.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [c for c in X_prepared.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def make_ohe(min_frequency: float = 0.005):
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=True,
        )
    except TypeError:  # compatibilidad sklearn antiguo
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    X_raw: pd.DataFrame,
    mode: str = "linear",
    min_frequency: float = 0.005,
    scaler: str = "robust",
):
    """Construye un ColumnTransformer según familia de modelo.

    `X_raw` puede venir crudo; internamente se prepara solo para inferir columnas.
    El pipeline final seguirá ejecutando FeatureEngineeringTransformer antes.
    """
    X_prepared = prepare_features(X_raw)
    numeric_cols, categorical_cols = get_column_groups(X_prepared)

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if mode == "linear":
        num_steps.append(("scaler", StandardScaler() if scaler == "standard" else RobustScaler()))
    num_pipe = Pipeline(num_steps)

    if mode == "ordinal":
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
    else:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("ohe", make_ohe(min_frequency=min_frequency)),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
