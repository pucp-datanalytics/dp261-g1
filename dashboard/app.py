"""Sprint 6 dashboard: consumes the deployed API instead of loading the model locally."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD_PATH = PROJECT_ROOT / "handoff" / "contracts" / "example_request.json"
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "test_final.csv"
TARGET = "IsBadBuy"

st.set_page_config(page_title="Kick Automotriz | Sprint 6 MVP", layout="wide")


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


API_URL = env("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = env("API_KEY", "")
REQUEST_TIMEOUT = int(env("API_TIMEOUT_SECONDS", "20"))


@st.cache_data(ttl=60, show_spinner=False)
def call_health(api_url: str) -> Dict[str, Any]:
    r = requests.get(f"{api_url}/health", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def call_version(api_url: str) -> Dict[str, Any]:
    r = requests.get(f"{api_url}/version", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def headers() -> Dict[str, str]:
    return {"x-api-key": API_KEY} if API_KEY else {}


@st.cache_data(ttl=60, show_spinner=False)
def predict_record(api_url: str, payload_json: str, api_key: str) -> Dict[str, Any]:
    hdrs = {"x-api-key": api_key} if api_key else {}
    payload = json.loads(payload_json)
    r = requests.post(f"{api_url}/predict", json=payload, headers=hdrs, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def predict_dataframe(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    input_df = df.drop(columns=[TARGET], errors="ignore").head(limit).copy()
    for idx, row in input_df.iterrows():
        payload = row.where(pd.notna(row), None).to_dict()
        pred = predict_record(API_URL, json.dumps(payload, sort_keys=True, default=str), API_KEY)
        records.append({
            "row_id": idx,
            "risk_score": pred.get("risk_score"),
            "threshold": pred.get("threshold"),
            "prediction": pred.get("prediction"),
            "decision": pred.get("decision"),
            "model_version": pred.get("model_version"),
            "api_version": pred.get("api_version"),
            "explanation_method": pred.get("explanation_method"),
            "shap_top_features": pred.get("shap_top_features", []),
        })
    return pd.DataFrame(records)


def load_example_payload() -> Dict[str, Any]:
    if DEFAULT_PAYLOAD_PATH.exists():
        return json.loads(DEFAULT_PAYLOAD_PATH.read_text(encoding="utf-8"))
    return {
        "VehicleAge": 3,
        "VehOdo": 75415,
        "VehBCost": 6500,
        "WarrantyCost": 1500,
    }


def load_sample_df() -> pd.DataFrame:
    if DEFAULT_SAMPLE_PATH.exists():
        return pd.read_csv(DEFAULT_SAMPLE_PATH)
    return pd.DataFrame([load_example_payload()])


st.title("Dashboard Sprint 6 — MVP desplegado")
st.caption("Dashboard conectado a la API REST. En Sprint 6 ya no carga el modelo localmente: consume /health, /version y /predict.")

with st.sidebar:
    st.header("Configuración API")
    st.write(f"API_URL: `{API_URL}`")
    st.write("API Key: configurada" if API_KEY else "API Key: no configurada para modo local")
    st.caption("Configura API_URL y API_KEY como variables de entorno en Streamlit Cloud, EC2 o local.")
    max_rows = st.slider("Máximo de filas a enviar", 1, 200, 25, 1)

status_col, version_col = st.columns(2)
try:
    health = call_health(API_URL)
    status_col.success(f"API health: {health.get('status')}")
    status_col.json(health)
except Exception as exc:
    status_col.error(f"No se pudo conectar a /health: {exc}")
    st.stop()

try:
    version = call_version(API_URL)
    version_col.info("Versión API / modelo")
    version_col.json(version)
except Exception as exc:
    version_col.warning(f"No se pudo leer /version: {exc}")

upload_tab, example_tab, explain_tab = st.tabs(["Predicción CSV", "Contrato example_request", "SHAP / explicación"])

with upload_tab:
    st.subheader("Predicción de lote vía API")
    uploaded = st.file_uploader("Sube CSV con variables raw del vehículo", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = load_sample_df()
        st.caption("Usando muestra local si existe data/processed/test_final.csv; si no, example_request.json.")

    st.write("Input preview")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Enviar a API /predict", type="primary"):
        with st.spinner("Llamando API..."):
            try:
                predictions = predict_dataframe(df, max_rows)
                st.session_state["predictions"] = predictions
                st.success(f"Predicciones generadas: {len(predictions)}")
            except requests.HTTPError as exc:
                st.error(f"Error HTTP desde API: {exc.response.status_code} - {exc.response.text}")
            except Exception as exc:
                st.error(f"No se pudo completar el flujo end-to-end: {exc}")

    predictions = st.session_state.get("predictions")
    if predictions is not None and not predictions.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas evaluadas", f"{len(predictions):,}")
        c2.metric("Marcadas Bad Buy", f"{int(predictions['prediction'].sum()):,}")
        c3.metric("Positive rate", f"{predictions['prediction'].mean():.1%}")

        st.dataframe(predictions.drop(columns=["shap_top_features"], errors="ignore"), use_container_width=True)
        fig = px.histogram(predictions, x="risk_score", nbins=25, title="Distribución de risk_score devuelta por API")
        if "threshold" in predictions and predictions["threshold"].notna().any():
            fig.add_vline(x=float(predictions["threshold"].dropna().iloc[0]), line_dash="dash", annotation_text="threshold")
        st.plotly_chart(fig, use_container_width=True)

        csv = predictions.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar resultados API", csv, "sprint6_api_predictions.csv", "text/csv")

with example_tab:
    st.subheader("Validación del contrato example_request.json")
    payload = load_example_payload()
    editable = st.text_area("Payload JSON", json.dumps(payload, indent=2, ensure_ascii=False), height=420)
    if st.button("Probar contrato /predict"):
        try:
            parsed = json.loads(editable)
            response = predict_record(API_URL, json.dumps(parsed, sort_keys=True, default=str), API_KEY)
            st.success("Contrato respondido correctamente")
            st.json(response)
        except Exception as exc:
            st.error(f"Falló la validación del contrato: {exc}")

with explain_tab:
    st.subheader("Explicabilidad devuelta por API")
    predictions = st.session_state.get("predictions")
    if predictions is None or predictions.empty:
        st.info("Primero genera predicciones en la pestaña CSV.")
    else:
        selected = st.selectbox("Selecciona row_id", predictions["row_id"].tolist())
        row = predictions.loc[predictions["row_id"] == selected].iloc[0]
        st.write("Predicción seleccionada")
        st.json(row.drop(labels=["shap_top_features"], errors="ignore").to_dict())
        shap_items = row.get("shap_top_features", [])
        if isinstance(shap_items, str):
            try:
                shap_items = json.loads(shap_items)
            except Exception:
                shap_items = []
        shap_df = pd.DataFrame(shap_items)
        if shap_df.empty:
            st.warning("La API no devolvió SHAP/explanations para esta instancia.")
        else:
            st.dataframe(shap_df, use_container_width=True)
            fig = px.bar(
                shap_df.sort_values("shap_value"),
                x="shap_value",
                y="feature",
                orientation="h",
                title="Top features que empujan la predicción",
            )
            st.plotly_chart(fig, use_container_width=True)
