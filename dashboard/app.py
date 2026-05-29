"""Dashboard comercial Streamlit para el MVP DP261.

Objetivo UX:
El usuario principal no es data scientist, sino un agente comercial o comprador que necesita
responder rápido: ¿procedo, reviso o detengo la compra del vehículo? Por eso el tablero prioriza:
1. semáforo visual y decisión accionable;
2. datos básicos del vehículo para identificarlo;
3. explicación simple de los principales factores;
4. detalle técnico concentrado en la barra lateral ocultable.

Uso local:
    API_URL=http://localhost:8000 streamlit run dashboard/app.py
"""
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
TARGET = "IsBadBuy"
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "processed" / "test_final.csv"

st.set_page_config(page_title="Semáforo Bad Buy", page_icon="🚦", layout="wide")

RISK_STYLE = {
    "ROJO": {
        "emoji": "🔴",
        "label": "Alto riesgo",
        "color": "#dc2626",
        "bg": "#fee2e2",
        "border": "#ef4444",
        "action": "Detener compra o derivar a revisión experta",
    },
    "ÁMBAR": {
        "emoji": "🟠",
        "label": "Riesgo medio",
        "color": "#d97706",
        "bg": "#fef3c7",
        "border": "#f59e0b",
        "action": "Revisar manualmente antes de comprar",
    },
    "VERDE": {
        "emoji": "🟢",
        "label": "Riesgo bajo",
        "color": "#16a34a",
        "bg": "#dcfce7",
        "border": "#22c55e",
        "action": "Continuar evaluación comercial",
    },
}

VEHICLE_ID_COLUMNS = [
    "row_id",
    "Auction",
    "Make",
    "Model",
    "VehYear",
    "VehicleAge",
    "VehOdo",
    "VehBCost",
    "WarrantyCost",
    "VNST",
]

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.05rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .step-box {
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 120px;
    }
    .risk-card {
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        border: 2px solid var(--risk-border);
        background: var(--risk-bg);
        color: #0f172a;
        margin-bottom: 1rem;
    }
    .risk-card .risk-emoji {
        font-size: 3rem;
        line-height: 1;
    }
    .risk-card .risk-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--risk-color);
    }
    .risk-card .risk-action {
        font-size: 1.05rem;
        font-weight: 650;
        margin-top: 0.25rem;
    }
    .metric-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: white;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .metric-card .metric-label {
        font-size: 0.86rem;
        color: #64748b;
    }
    .metric-card .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
    }
    .vehicle-chip {
        display: inline-block;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        margin: 0.15rem 0.15rem 0.15rem 0;
        background: #ffffff;
        font-size: 0.88rem;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_url() -> str:
    return os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def headers() -> Dict[str, str]:
    key = os.getenv("API_KEY")
    return {"x-api-key": key} if key else {}


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"USD {value:,.0f}"


def percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,.0f}"
    return str(value)


def clean_record(row: pd.Series) -> Dict[str, Any]:
    record = {}
    for key, value in row.items():
        if key == TARGET:
            continue
        if pd.isna(value):
            record[key] = None
        elif hasattr(value, "item"):
            record[key] = value.item()
        else:
            record[key] = value
    return record


@st.cache_data(ttl=60, show_spinner=False)
def call_get(path: str) -> Dict[str, Any]:
    r = requests.get(f"{api_url()}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def predict_one(payload_json: str) -> Dict[str, Any]:
    payload = json.loads(payload_json)
    r = requests.post(f"{api_url()}/predict", json=payload, headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def load_default_sample() -> pd.DataFrame:
    if DEFAULT_SAMPLE.exists():
        # Para demo se toma una muestra pequeña. Si existe target, solo se usa para validar internamente.
        return pd.read_csv(DEFAULT_SAMPLE).head(50)
    return pd.DataFrame()


def vehicle_summary(row: pd.Series) -> str:
    make = row.get("Make", "")
    model = row.get("Model", "")
    year = row.get("VehYear", "")
    auction = row.get("Auction", "")
    return " | ".join([str(x) for x in [make, model, year, auction] if str(x) not in {"", "nan", "None"}])


def predict_df(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    input_df = df.head(limit).copy()
    progress = st.progress(0, text="Evaluando vehículos...")
    for i, (idx, row) in enumerate(input_df.iterrows(), start=1):
        payload = clean_record(row)
        result = predict_one(json.dumps(payload, sort_keys=True, default=str))
        segment = result.get("risk_segment", "N/A")
        style = RISK_STYLE.get(segment, {})
        base_data = {col: row.get(col) for col in VEHICLE_ID_COLUMNS if col != "row_id" and col in input_df.columns}
        rows.append(
            {
                "row_id": idx,
                "Vehículo": vehicle_summary(row),
                "Semáforo": f"{style.get('emoji', '⚪')} {segment}",
                "risk_segment": segment,
                "Riesgo estimado": result.get("risk_score"),
                "Decisión comercial": result.get("decision"),
                "prediction": result.get("prediction"),
                "threshold": result.get("threshold"),
                "Factores clave": result.get("shap_top_features", []),
                "Método explicación": result.get("explanation_method"),
                "raw_response": result,
                **base_data,
            }
        )
        progress.progress(i / max(len(input_df), 1), text=f"Evaluados {i}/{len(input_df)}")
    progress.empty()
    pred = pd.DataFrame(rows)
    if TARGET in input_df.columns and not pred.empty:
        pred["actual"] = input_df.loc[pred["row_id"], TARGET].astype(int).values
    return pred


def business_value(pred: pd.DataFrame, assumptions: Dict[str, float]) -> float | None:
    if "actual" not in pred.columns or pred.empty:
        return None
    y = pred["actual"].astype(int)
    p = pred["prediction"].astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return (
        tp * assumptions.get("benefit_tp", 2500)
        + tn * assumptions.get("benefit_tn", 600)
        + fp * assumptions.get("cost_fp", -900)
        + fn * assumptions.get("cost_fn", -4500)
    )


def render_metric_card(label: str, value: str, color: str | None = None) -> None:
    style = f"color:{color};" if color else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="{style}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(segment: str, risk_score: float | None, decision: str) -> None:
    style = RISK_STYLE.get(segment, RISK_STYLE["ÁMBAR"])
    st.markdown(
        f"""
        <div class="risk-card" style="--risk-bg:{style['bg']}; --risk-border:{style['border']}; --risk-color:{style['color']};">
            <div style="display:flex; gap:1rem; align-items:center;">
                <div class="risk-emoji">{style['emoji']}</div>
                <div>
                    <div class="risk-title">{segment} · {style['label']}</div>
                    <div class="risk-action">{style['action']}</div>
                    <div class="small-muted">Score de riesgo del modelo: <b>{percent(risk_score)}</b></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(decision)


def render_vehicle_chips(row: pd.Series) -> None:
    chips = []
    labels = {
        "Auction": "Subasta",
        "Make": "Marca",
        "Model": "Modelo",
        "VehYear": "Año",
        "VehicleAge": "Antigüedad",
        "VehOdo": "Kilometraje",
        "VehBCost": "Costo compra",
        "WarrantyCost": "Garantía",
        "VNST": "Estado",
    }
    for col, label in labels.items():
        if col in row.index:
            value = row.get(col)
            if col in {"VehBCost", "WarrantyCost"}:
                value_text = money(value)
            else:
                value_text = fmt_number(value)
            chips.append(f'<span class="vehicle-chip"><b>{label}:</b> {value_text}</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)


def factors_to_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    rows = []
    for item in items:
        rows.append(
            {
                "Factor": item.get("display_name") or item.get("feature"),
                "Valor del vehículo": item.get("value"),
                "Relevancia": abs(float(item.get("impact_abs") or item.get("impact") or 0.0)),
                "Lectura": item.get("impact_direction", "factor_relevante"),
                "Detalle técnico": item.get("detail"),
            }
        )
    return pd.DataFrame(rows).sort_values("Relevancia", ascending=False)


st.markdown('<div class="main-title">🚦 Semáforo comercial de riesgo de compra</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Herramienta para decidir rápidamente si un vehículo puede continuar, requiere revisión manual o debe detenerse por alto riesgo.</div>',
    unsafe_allow_html=True,
)

with st.expander("📌 ¿Cómo usar este reporte?", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="step-box">
            <b>1. Sube el archivo</b><br>
            Carga un CSV con las columnas originales del vehículo. No necesitas crear variables nuevas: la API hace el preprocesamiento.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="step-box">
            <b>2. Lee el semáforo</b><br>
            🟢 continuar · 🟠 revisar manualmente · 🔴 detener o escalar. La decisión está pensada para negocio, no para técnicos.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="step-box">
            <b>3. Revisa el motivo</b><br>
            Para cada vehículo se muestran los factores más relevantes, como kilometraje, costo, antigüedad, garantía o precios MMR.
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Panel técnico lateral
# -----------------------------------------------------------------------------
# Streamlit permite ocultar/mostrar la barra lateral. Por eso todo lo técnico
# se concentra aquí: estado de API, versión del modelo, supuestos de negocio y
# respuesta JSON cruda. El cuerpo principal queda reservado para el usuario
# comercial: instrucciones, carga, semáforos, gráficos y tablas.
with st.sidebar:
    st.title("⚙️ Panel técnico")
    st.caption(
        "Sección para equipo técnico o sustentación. El agente comercial puede ocultarla desde la flecha lateral."
    )
    st.markdown("### Conexión API")
    st.code(api_url(), language="text")

try:
    health = call_get("/health")
    version = call_get("/version")
except Exception as exc:
    with st.sidebar:
        st.error("API no conectada")
        st.caption(str(exc))
    st.error("No se pudo conectar con la API. Primero levanta Docker/FastAPI y vuelve a cargar el dashboard.")
    st.stop()

with st.sidebar:
    st.success("API conectada")
    st.markdown("### Estado y versión")
    st.write(f"**Estado:** `{health.get('status', 'N/A')}`")
    st.write(f"**Modelo:** `{version.get('model_name', 'N/A')}`")
    st.write(f"**Versión modelo:** `{version.get('model_version', 'N/A')}`")
    st.write(f"**Versión API:** `{version.get('api_version', 'N/A')}`")
    with st.expander("Ver JSON técnico de health/version", expanded=False):
        st.json({"health": health, "version": version})
    st.markdown("---")
    st.markdown("### Leyenda técnica")
    for segment, style in RISK_STYLE.items():
        st.markdown(f"{style['emoji']} **{segment}**: {style['action']}")
    st.caption("La leyenda también se explica en el cuerpo para el usuario comercial.")

st.subheader("Carga de datos")
col_upload, col_limit = st.columns([2.2, 0.8])
with col_upload:
    uploaded = st.file_uploader("Sube un CSV de vehículos para evaluar", type="csv")
with col_limit:
    limit = st.slider("Máximo de vehículos", 1, 300, 30)
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Archivo cargado: {len(df):,} vehículos y {len(df.columns):,} columnas.")
else:
    st.info("Aún no subiste un archivo. Para demo se puede usar una muestra local del test final.")
    use_sample = st.toggle("Usar muestra local de demo", value=True)
    df = load_default_sample() if use_sample else pd.DataFrame()

if df.empty:
    st.warning("Sube un CSV o activa la muestra local para iniciar la evaluación.")
    st.stop()

with st.expander("Vista previa del archivo cargado", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

if st.button("🚦 Evaluar vehículos", type="primary", use_container_width=True):
    pred = predict_df(df, limit=limit)
    st.session_state["pred"] = pred
else:
    pred = st.session_state.get("pred", pd.DataFrame())

if pred.empty:
    st.stop()

# Summary cards
rojos = int((pred["risk_segment"] == "ROJO").sum())
ambar = int((pred["risk_segment"] == "ÁMBAR").sum())
verdes = int((pred["risk_segment"] == "VERDE").sum())

st.subheader("Resumen ejecutivo del lote")
m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Vehículos evaluados", f"{len(pred):,}")
with m2:
    render_metric_card("🔴 Detener / escalar", f"{rojos:,}", RISK_STYLE["ROJO"]["color"])
with m3:
    render_metric_card("🟠 Revisión manual", f"{ambar:,}", RISK_STYLE["ÁMBAR"]["color"])
with m4:
    render_metric_card("🟢 Continuar", f"{verdes:,}", RISK_STYLE["VERDE"]["color"])

assumptions = pred["raw_response"].iloc[0].get("business_value_assumptions", {})
value = business_value(pred, assumptions)
if value is not None:
    st.metric("Valor económico estimado del lote evaluado", money(value))

# Visual distribution
chart_df = pred["risk_segment"].value_counts().reindex(["ROJO", "ÁMBAR", "VERDE"]).fillna(0).reset_index()
chart_df.columns = ["Semáforo", "Vehículos"]
fig = px.bar(
    chart_df,
    x="Semáforo",
    y="Vehículos",
    text="Vehículos",
    color="Semáforo",
    color_discrete_map={"ROJO": "#dc2626", "ÁMBAR": "#f59e0b", "VERDE": "#16a34a"},
    title="Distribución del lote por decisión comercial",
)
fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Resultado por vehículo")
show_cols = [
    "row_id",
    "Semáforo",
    "Riesgo estimado",
    "Decisión comercial",
    "Auction",
    "Make",
    "Model",
    "VehYear",
    "VehicleAge",
    "VehOdo",
    "VehBCost",
    "WarrantyCost",
    "VNST",
]
show_cols = [c for c in show_cols if c in pred.columns]
result_table = pred[show_cols].copy()
if "Riesgo estimado" in result_table.columns:
    result_table["Riesgo estimado"] = result_table["Riesgo estimado"].map(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "N/A")
st.dataframe(result_table, use_container_width=True, hide_index=True)

csv_download = pred.drop(columns=["raw_response"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar resultados del lote",
    data=csv_download,
    file_name="resultados_semaforo_badbuy.csv",
    mime="text/csv",
    use_container_width=True,
)

st.subheader("Detalle de un vehículo")
selected = st.selectbox(
    "Selecciona una fila para ver decisión y factores",
    pred["row_id"].tolist(),
    format_func=lambda x: f"Fila {x} · {pred.loc[pred['row_id'] == x, 'Semáforo'].iloc[0]} · {pred.loc[pred['row_id'] == x, 'Vehículo'].iloc[0]}",
)
selected_row = pred.loc[pred["row_id"] == selected].iloc[0]

left, right = st.columns([0.95, 1.05])
with left:
    render_risk_card(
        str(selected_row.get("risk_segment")),
        selected_row.get("Riesgo estimado"),
        str(selected_row.get("Decisión comercial")),
    )
    st.markdown("**Datos para identificar el vehículo**")
    render_vehicle_chips(selected_row)

with right:
    st.markdown("**Factores que explican la alerta**")
    items = selected_row.get("Factores clave", [])
    factors = factors_to_df(items)
    if not factors.empty:
        fig2 = px.bar(
            factors.head(5),
            x="Relevancia",
            y="Factor",
            orientation="h",
            text="Valor del vehículo",
            title="Top factores del vehículo seleccionado",
        )
        fig2.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(factors[["Factor", "Valor del vehículo", "Lectura"]], hide_index=True, use_container_width=True)
        st.caption(
            "La relevancia muestra qué variables pesaron más en la evaluación. Si el método es fallback de importancia, la dirección exacta se interpreta con cautela."
        )
    else:
        st.info("La API no devolvió factores explicativos para este caso. Revisar configuración de SHAP/fallback en la API.")

with st.sidebar:
    st.markdown("---")
    st.markdown("### Predicción seleccionada")
    st.write(f"**Fila:** `{selected}`")
    st.write(f"**Método explicación:** `{selected_row.get('Método explicación')}`")
    with st.expander("Ver respuesta JSON completa", expanded=False):
        st.json(selected_row.get("raw_response", {}))
