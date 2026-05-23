"""Dashboard Sprint 6 — Stakeholder view connected to the deployed API.

This Streamlit app is intentionally business-facing: it consumes the API
(/health, /version, /predict) but hides technical details in an expandable
section. It supports batch CSV scoring, risk segmentation, business-value
validation when the target column is available, and per-instance explainability.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD_PATH = PROJECT_ROOT / "handoff" / "contracts" / "example_request.json"
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "test_final.csv"
TARGET = "IsBadBuy"

DEFAULT_ASSUMPTIONS = {
    "benefit_tp": 2500,
    "cost_fp": -900,
    "cost_fn": 0,
    "benefit_tn": 0,
}

st.set_page_config(
    page_title="Evaluación de Riesgo de Compra Automotriz",
    page_icon="🚗",
    layout="wide",
)


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"USD {value:,.0f}"


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def normalize_api_url(url: str) -> str:
    return url.strip().rstrip("/")


def make_headers(api_key: str) -> Dict[str, str]:
    return {"x-api-key": api_key} if api_key else {}


@st.cache_data(ttl=60, show_spinner=False)
def call_health(api_url: str, timeout: int) -> Dict[str, Any]:
    response = requests.get(f"{api_url}/health", timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def call_version(api_url: str, timeout: int) -> Dict[str, Any]:
    response = requests.get(f"{api_url}/version", timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def predict_record(api_url: str, payload_json: str, api_key: str, timeout: int) -> Dict[str, Any]:
    payload = json.loads(payload_json)
    response = requests.post(
        f"{api_url}/predict",
        json=payload,
        headers=make_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


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


def clean_payload(row: pd.Series) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in row.items():
        if key == TARGET:
            continue
        if pd.isna(value):
            payload[key] = None
        elif hasattr(value, "item"):
            payload[key] = value.item()
        else:
            payload[key] = value
    return payload


def predict_dataframe(df: pd.DataFrame, api_url: str, api_key: str, timeout: int, limit: int) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    input_df = df.drop(columns=[TARGET], errors="ignore").head(limit).copy()

    progress = st.progress(0, text="Enviando vehículos a la API...")
    total = len(input_df)

    for pos, (idx, row) in enumerate(input_df.iterrows(), start=1):
        payload = clean_payload(row)
        result = predict_record(
            api_url,
            json.dumps(payload, sort_keys=True, default=str),
            api_key,
            timeout,
        )
        records.append(
            {
                "row_id": idx,
                "risk_score": result.get("risk_score"),
                "threshold": result.get("threshold"),
                "prediction": result.get("prediction"),
                "decision": result.get("decision"),
                "model_name": result.get("model_name"),
                "model_version": result.get("model_version"),
                "api_version": result.get("api_version"),
                "explanation_method": result.get("explanation_method"),
                "shap_top_features": result.get("shap_top_features", []),
                "raw_response": result,
            }
        )
        progress.progress(pos / max(total, 1), text=f"Procesados {pos}/{total} vehículos")

    progress.empty()
    predictions = pd.DataFrame(records)
    if TARGET in df.columns and not predictions.empty:
        y_true = df.loc[predictions["row_id"], TARGET].astype(int).values
        predictions["actual_is_bad_buy"] = y_true
    return predictions


def risk_segment(score: float | None, threshold: float | None) -> str:
    if score is None or pd.isna(score):
        return "Sin score"
    if threshold is None or pd.isna(threshold):
        threshold = 0.70
    if score >= threshold:
        return "Alto riesgo"
    if score >= max(threshold - 0.20, 0):
        return "Riesgo medio"
    return "Riesgo bajo"


def build_results_view(source_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    source = source_df.copy()
    source["row_id"] = source.index
    visible_cols = [
        "row_id",
        "risk_score",
        "threshold",
        "prediction",
        "risk_segment",
        "decision",
        "actual_is_bad_buy",
    ]
    preds = predictions.copy()
    preds["risk_segment"] = [risk_segment(s, t) for s, t in zip(preds["risk_score"], preds["threshold"])]
    existing_visible = [c for c in visible_cols if c in preds.columns]
    merged = preds[existing_visible].merge(source, on="row_id", how="left")
    return merged


def confusion_and_value(predictions: pd.DataFrame, assumptions: Dict[str, float]) -> Tuple[Optional[Dict[str, int]], Optional[float]]:
    if "actual_is_bad_buy" not in predictions.columns or predictions.empty:
        return None, None

    y_true = predictions["actual_is_bad_buy"].astype(int)
    y_pred = predictions["prediction"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    value = (
        tp * assumptions.get("benefit_tp", 2500)
        + fp * assumptions.get("cost_fp", -900)
        + fn * assumptions.get("cost_fn", 0)
        + tn * assumptions.get("benefit_tn", 0)
    )
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}, float(value)


def extract_assumptions(predictions: pd.DataFrame) -> Dict[str, float]:
    if predictions.empty or "raw_response" not in predictions.columns:
        return DEFAULT_ASSUMPTIONS
    raw = predictions["raw_response"].iloc[0]
    assumptions = raw.get("business_value_assumptions", {}) if isinstance(raw, dict) else {}
    return {
        "benefit_tp": float(assumptions.get("benefit_tp", DEFAULT_ASSUMPTIONS["benefit_tp"])),
        "cost_fp": float(assumptions.get("cost_fp", DEFAULT_ASSUMPTIONS["cost_fp"])),
        "cost_fn": float(assumptions.get("cost_fn", DEFAULT_ASSUMPTIONS["cost_fn"])),
        "benefit_tn": float(assumptions.get("benefit_tn", DEFAULT_ASSUMPTIONS["benefit_tn"])),
    }


def plot_confusion_matrix(counts: Dict[str, int]) -> go.Figure:
    matrix = [[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicción: Good Buy", "Predicción: Bad Buy"],
            y=["Real: Good Buy", "Real: Bad Buy"],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Casos: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Matriz de confusión del lote evaluado", height=420)
    return fig


def flatten_shap_items(items: Any) -> pd.DataFrame:
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except Exception:
            items = []
    if not isinstance(items, list):
        items = []
    df = pd.DataFrame(items)
    if df.empty:
        return df
    if "shap_value" in df.columns:
        df["abs_impact"] = df["shap_value"].abs()
        df = df.sort_values("abs_impact", ascending=False)
    return df


# -----------------------------------------------------------------------------
# Sidebar configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración del MVP")
    api_url = normalize_api_url(
        st.text_input("Endpoint de la API", value=get_env("API_URL", "http://localhost:8000"))
    )
    api_key = st.text_input(
        "API Key",
        value=get_env("API_KEY", ""),
        type="password",
        help="En local puede quedar vacío si la API corre con REQUIRE_API_KEY=false.",
    )
    timeout = int(st.number_input("Timeout por request (segundos)", min_value=5, max_value=120, value=int(get_env("API_TIMEOUT_SECONDS", "20"))))
    max_rows = int(st.slider("Máximo de vehículos a evaluar", min_value=1, max_value=500, value=50, step=1))
    st.caption("Para EC2, cambia el endpoint por http://<EC2_PUBLIC_IP> o el dominio HTTPS configurado.")

# -----------------------------------------------------------------------------
# Header and API diagnostics
# -----------------------------------------------------------------------------
st.title("Evaluación de Riesgo de Compra Automotriz")
st.write(
    "Dashboard para priorizar la revisión de vehículos en subastas, identificar posibles "
    "Bad Buys y cuantificar el impacto económico esperado del modelo."
)

health_data: Dict[str, Any] = {}
version_data: Dict[str, Any] = {}
api_ready = False

try:
    health_data = call_health(api_url, timeout)
    version_data = call_version(api_url, timeout)
    api_ready = health_data.get("status") == "ok" and bool(health_data.get("model_loaded", True))
except Exception as exc:
    st.error(f"No se pudo conectar con la API en {api_url}. Detalle: {exc}")
    st.info("Verifica que la API esté corriendo con Uvicorn o Docker antes de usar el dashboard.")

status_col, model_col, threshold_col, mode_col = st.columns(4)
status_col.metric("Estado del MVP", "Operativo" if api_ready else "No disponible")
model_col.metric("Modelo", version_data.get("model_version", "N/A"))
threshold_col.metric("Threshold operativo", version_data.get("threshold", 0.70))
mode_col.metric("Modo", "API REST")

with st.expander("Diagnóstico técnico de la API", expanded=False):
    st.write("Este bloque sirve para QA/Sprint Review. No es necesario para el usuario final.")
    c1, c2 = st.columns(2)
    c1.write(f"Endpoint configurado: `{api_url}`")
    c1.json(health_data or {"status": "unavailable"})
    c2.json(version_data or {"version": "unavailable"})

if not api_ready:
    st.stop()

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
tab_upload, tab_results, tab_explain, tab_contract = st.tabs(
    [
        "1. Evaluar vehículos",
        "2. Resultados y valor de negocio",
        "3. Explicabilidad por vehículo",
        "4. Contrato API",
    ]
)

with tab_upload:
    st.subheader("Carga de vehículos para evaluación")
    st.write(
        "Sube un CSV con las variables originales del vehículo. Si el archivo incluye "
        "`IsBadBuy`, el dashboard también calcula matriz de confusión y valor económico del lote."
    )

    uploaded = st.file_uploader("CSV de vehículos", type="csv")
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        source_label = "archivo cargado"
    else:
        input_df = load_sample_df()
        source_label = "muestra local"

    st.caption(f"Fuente actual: {source_label}. Filas disponibles: {len(input_df):,}")
    st.dataframe(input_df.head(20), use_container_width=True)

    missing_core = [c for c in ["VehicleAge", "VehOdo", "VehBCost", "WarrantyCost"] if c not in input_df.columns]
    if missing_core:
        st.warning(
            "El CSV no contiene algunas variables comunes del contrato: " + ", ".join(missing_core) +
            ". La API intentará predecir igual si el contrato mínimo se cumple."
        )

    if st.button("Generar evaluación de riesgo", type="primary", use_container_width=True):
        try:
            with st.spinner("Generando predicciones vía API..."):
                predictions_df = predict_dataframe(input_df, api_url, api_key, timeout, max_rows)
                st.session_state["input_df"] = input_df
                st.session_state["predictions_df"] = predictions_df
            st.success(f"Evaluación completada para {len(predictions_df):,} vehículos.")
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"La API respondió con error HTTP: {detail}")
        except Exception as exc:
            st.error(f"No se pudo completar la evaluación: {exc}")

with tab_results:
    st.subheader("Resultados del lote evaluado")
    predictions_df = st.session_state.get("predictions_df")
    input_df = st.session_state.get("input_df")

    if predictions_df is None or predictions_df.empty or input_df is None:
        st.info("Primero genera predicciones en la pestaña 'Evaluar vehículos'.")
    else:
        threshold = float(predictions_df["threshold"].dropna().iloc[0]) if predictions_df["threshold"].notna().any() else 0.70
        results_view = build_results_view(input_df, predictions_df)
        assumptions = extract_assumptions(predictions_df)
        counts, business_value = confusion_and_value(predictions_df, assumptions)

        total = len(predictions_df)
        high_risk = int((predictions_df["prediction"] == 1).sum())
        high_risk_rate = high_risk / total if total else 0
        avg_score = float(predictions_df["risk_score"].mean())

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Vehículos evaluados", f"{total:,}")
        kpi2.metric("Alto riesgo", f"{high_risk:,}", pct(high_risk_rate))
        kpi3.metric("Score promedio", f"{avg_score:.3f}")
        kpi4.metric("Valor económico", money(business_value) if business_value is not None else "Requiere IsBadBuy")

        st.markdown("#### Segmentación de riesgo")
        seg_df = results_view["risk_segment"].value_counts().rename_axis("segmento").reset_index(name="vehículos")
        fig_seg = px.bar(seg_df, x="segmento", y="vehículos", text="vehículos", title="Distribución por segmento de riesgo")
        st.plotly_chart(fig_seg, use_container_width=True)

        fig_score = px.histogram(
            predictions_df,
            x="risk_score",
            nbins=30,
            title="Distribución de probabilidad de Bad Buy",
        )
        fig_score.add_vline(x=threshold, line_dash="dash", annotation_text=f"threshold {threshold:.2f}")
        st.plotly_chart(fig_score, use_container_width=True)

        st.markdown("#### Tabla priorizada para revisión")
        filter_segment = st.multiselect(
            "Filtrar segmento",
            options=sorted(results_view["risk_segment"].dropna().unique().tolist()),
            default=sorted(results_view["risk_segment"].dropna().unique().tolist()),
        )
        filtered = results_view[results_view["risk_segment"].isin(filter_segment)].copy()
        filtered = filtered.sort_values("risk_score", ascending=False)
        st.dataframe(filtered, use_container_width=True, height=420)

        st.download_button(
            "Descargar resultados priorizados",
            filtered.to_csv(index=False).encode("utf-8"),
            "badbuy_risk_predictions.csv",
            "text/csv",
            use_container_width=True,
        )

        if counts is not None:
            st.markdown("#### Validación contra etiqueta real")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(plot_confusion_matrix(counts), use_container_width=True)
            with c2:
                st.write("Supuestos de valor usados por la API:")
                st.json(assumptions)
                st.write(
                    "Función: `BusinessValue = TP*2500 + FP*(-900) + FN*0 + TN*0` "
                    "bajo el threshold operativo definido en Sprint 5."
                )
                st.metric("Business Value del lote", money(business_value))
        else:
            st.info("Para calcular matriz de confusión y business value del lote, incluye la columna `IsBadBuy` en el CSV.")

with tab_explain:
    st.subheader("¿Por qué el modelo predijo ese riesgo?")
    st.write(
        "Esta sección muestra los factores que más empujan la predicción hacia mayor o menor riesgo "
        "para un vehículo seleccionado. La explicación llega desde la API."
    )

    predictions_df = st.session_state.get("predictions_df")
    input_df = st.session_state.get("input_df")

    if predictions_df is None or predictions_df.empty or input_df is None:
        st.info("Primero genera predicciones en la pestaña 'Evaluar vehículos'.")
    else:
        ranked = predictions_df.sort_values("risk_score", ascending=False)
        selected = st.selectbox(
            "Selecciona vehículo por row_id",
            ranked["row_id"].tolist(),
            format_func=lambda x: f"row_id {x} | score {float(predictions_df.loc[predictions_df['row_id'] == x, 'risk_score'].iloc[0]):.3f}",
        )
        row_pred = predictions_df.loc[predictions_df["row_id"] == selected].iloc[0]
        source_row = input_df.loc[selected].to_dict() if selected in input_df.index else {}

        p1, p2, p3 = st.columns(3)
        p1.metric("Risk score", f"{float(row_pred['risk_score']):.3f}")
        p2.metric("Decisión", "Bad Buy" if int(row_pred["prediction"]) == 1 else "Good Buy")
        p3.metric("Threshold", f"{float(row_pred['threshold']):.2f}")

        st.write("Datos del vehículo seleccionado")
        st.json(source_row)

        shap_df = flatten_shap_items(row_pred.get("shap_top_features", []))
        if shap_df.empty:
            st.warning("La API no devolvió explicación SHAP para esta instancia.")
            st.caption(f"Método reportado: {row_pred.get('explanation_method', 'N/A')}")
        else:
            st.caption(f"Método de explicación: {row_pred.get('explanation_method', 'N/A')}")
            st.dataframe(shap_df, use_container_width=True)
            fig_shap = px.bar(
                shap_df.sort_values("shap_value"),
                x="shap_value",
                y="feature",
                orientation="h",
                title="Variables con mayor impacto en la predicción",
                hover_data=[c for c in ["impact_direction", "abs_impact"] if c in shap_df.columns],
            )
            st.plotly_chart(fig_shap, use_container_width=True)

with tab_contract:
    st.subheader("Prueba controlada del contrato de entrada/salida")
    st.write(
        "Este bloque es útil para QA y Sprint Review: permite validar que `example_request.json` "
        "sigue produciendo una respuesta válida de `/predict`."
    )

    payload = load_example_payload()
    editable = st.text_area("Payload JSON", json.dumps(payload, indent=2, ensure_ascii=False), height=360)
    if st.button("Validar contrato con /predict"):
        try:
            parsed = json.loads(editable)
            response = predict_record(api_url, json.dumps(parsed, sort_keys=True, default=str), api_key, timeout)
            st.success("Contrato validado correctamente")
            st.json(response)
        except Exception as exc:
            st.error(f"Falló la validación del contrato: {exc}")
