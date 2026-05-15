import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from src.config import RANDOM_STATE, TARGET


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = (
            df[col]
            .astype("object")
            .where(df[col].notna(), np.nan)
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "<NA>": np.nan, "": np.nan})
        )
    return df


def safe_divide(numerator, denominator):
    return np.where((pd.notna(denominator)) & (denominator != 0), numerator / denominator, np.nan)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica decisiones del EDA sin hacer fit ni mirar test.

    Decisiones tomadas con los resultados encontrados:
    - PurchDate se convierte en componentes temporales.
    - WheelTypeID se elimina por redundancia con WheelType.
    - MMR con valor 0 se trata como missing porque representa precio no disponible.
    - PRIMEUNIT/AUCGUART NO se eliminan aunque tienen 95% nulos: el missing tiene señal.
    - BYRNO y VNZIP1 se tratan como categóricas, no como números ordinales.
    """
    df = df.copy()
    df = normalize_text_columns(df)

    if "PurchDate" in df.columns:
        df["PurchDate"] = pd.to_datetime(df["PurchDate"], errors="coerce")
        df["PurchYear"] = df["PurchDate"].dt.year
        df["PurchMonth"] = df["PurchDate"].dt.month
        df["PurchQuarter"] = df["PurchDate"].dt.quarter
        df["PurchDayOfWeek"] = df["PurchDate"].dt.dayofweek
        df = df.drop(columns=["PurchDate"])

    if "WheelTypeID" in df.columns:
        df = df.drop(columns=["WheelTypeID"])

    mmr_cols = [c for c in df.columns if c.startswith("MMR")]
    for col in mmr_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= 0, col] = np.nan

    if {"VehBCost", "MMRAcquisitionAuctionAveragePrice"}.issubset(df.columns):
        df["cost_to_acq_auction_avg"] = safe_divide(df["VehBCost"], df["MMRAcquisitionAuctionAveragePrice"])
        df["acq_auction_margin"] = df["MMRAcquisitionAuctionAveragePrice"] - df["VehBCost"]

    if {"VehBCost", "MMRCurrentAuctionAveragePrice"}.issubset(df.columns):
        df["cost_to_current_auction_avg"] = safe_divide(df["VehBCost"], df["MMRCurrentAuctionAveragePrice"])
        df["current_auction_margin"] = df["MMRCurrentAuctionAveragePrice"] - df["VehBCost"]

    if {"WarrantyCost", "VehBCost"}.issubset(df.columns):
        df["warranty_to_cost"] = safe_divide(df["WarrantyCost"], df["VehBCost"])

    if {"MMRCurrentAuctionAveragePrice", "MMRAcquisitionAuctionAveragePrice"}.issubset(df.columns):
        df["current_vs_acq_auction_avg"] = (
            df["MMRCurrentAuctionAveragePrice"] - df["MMRAcquisitionAuctionAveragePrice"]
        )

    for col in ["BYRNO", "VNZIP1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("object").astype(str).replace({"<NA>": np.nan})

    return df


def split_X_y(df: pd.DataFrame):
    y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def get_column_groups(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def make_ohe(min_frequency=0.01):
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame, mode: str = "linear", min_frequency: float = 0.01):
    """Construye preprocesadores según la familia del modelo.

    mode='linear': LR/SVM/KNN -> imputación + RobustScaler + OHE.
    mode='tree_ohe': RF/árboles -> imputación + OHE, sin escalado.
    mode='ordinal': boosting/HistGB/LightGBM -> imputación + OrdinalEncoder, sin escalado.
    """
    numeric_cols, categorical_cols = get_column_groups(X)

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if mode == "linear":
        num_steps.append(("scaler", RobustScaler()))
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
