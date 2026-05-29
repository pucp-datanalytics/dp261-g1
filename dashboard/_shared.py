"""Funciones y componentes compartidos para los dashboards Streamlit DP261.

Este módulo concentra estilos, utilidades, carga de archivos, llamadas a la API
y renderizado de bloques comunes. No define navegación: cada dashboard se corre
por separado para tener links independientes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET = "IsBadBuy"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "06-kickAutomotriz.csv"
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "processed" / "test_final.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
RANKING_PATH = PROJECT_ROOT / "reports" / "business_model_ranking.csv"
FINAL_METRICS_PATH = PROJECT_ROOT / "reports" / "final_validation_metrics.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"

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

STAGE_LABELS = {
    "baseline_sprint3": "Baseline Sprint 3",
    "tuned_sprint4": "Tuning Sprint 4",
    "advanced_sprint4": "Avanzado Sprint 4",
    "ensembles_sprint4": "Ensamble Sprint 4",
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


FEATURE_FRIENDLY_NAMES = {
    "VehOdo": "Kilometraje del vehículo",
    "VehicleAge": "Antigüedad del vehículo",
    "VehBCost": "Costo base de compra",
    "WarrantyCost": "Costo de garantía",
    "Auction": "Casa de subasta",
    "Make": "Marca",
    "Model": "Modelo",
    "Size": "Tamaño / segmento",
    "Nationality": "Procedencia / nacionalidad",
    "TopThreeAmericanName": "Grupo de marca",
    "MMRAcquisitionAuctionAveragePrice": "Precio MMR adquisición subasta",
    "MMRAcquisitionAuctionCleanPrice": "Precio MMR clean adquisición",
    "MMRAcquisitionRetailAveragePrice": "Precio MMR retail adquisición",
    "MMRAcquisitonRetailCleanPrice": "Precio MMR retail clean adquisición",
    "MMRCurrentAuctionAveragePrice": "Precio MMR actual subasta",
    "MMRCurrentAuctionCleanPrice": "Precio MMR clean actual",
    "MMRCurrentRetailAveragePrice": "Precio MMR retail actual",
    "MMRCurrentRetailCleanPrice": "Precio MMR retail clean actual",
    "PRIMEUNIT": "Indicador PRIMEUNIT",
    "AUCGUART": "Indicador AUCGUART",
    "VNST": "Estado de venta",
    "IsOnlineSale": "Venta online",
    "odo_per_year": "Kilometraje por año",
    "old_high_mileage_flag": "Vehículo antiguo con alto kilometraje",
    "cost_to_acq_auction_avg": "Costo vs. MMR adquisición",
    "acq_auction_margin": "Margen frente a MMR adquisición",
    "cost_to_current_auction_avg": "Costo vs. MMR actual",
    "current_auction_margin": "Margen frente a MMR actual",
    "warranty_to_cost": "Garantía como proporción del costo",
    "warranty_per_vehicle_year": "Garantía por año de antigüedad",
    "auction_avg_depreciation": "Depreciación estimada subasta",
    "retail_avg_depreciation": "Depreciación estimada retail",
    "acq_clean_avg_spread": "Spread adquisición clean vs. average",
    "current_clean_avg_spread": "Spread actual clean vs. average",
    "mmr_missing_count": "Precios MMR faltantes",
}

FEATURE_GROUPS = [
    {
        "grupo": "Tiempo de compra",
        "features": "PurchYear, PurchMonth, PurchQuarter, PurchDayOfWeek",
        "para_que_sirve": "Captura patrones estacionales y cambios de mercado por momento de compra.",
    },
    {
        "grupo": "Uso y desgaste",
        "features": "odo_per_year, old_high_mileage_flag",
        "para_que_sirve": "Detecta vehículos con kilometraje excesivo para su edad, una señal directa de mayor riesgo operativo.",
    },
    {
        "grupo": "Costo vs. precio de mercado",
        "features": "cost_to_acq_auction_avg, acq_auction_margin, cost_to_current_auction_avg, current_auction_margin",
        "para_que_sirve": "Compara el costo pagado contra referencias MMR para identificar compras caras o inconsistentes.",
    },
    {
        "grupo": "Garantía y antigüedad",
        "features": "warranty_to_cost, warranty_per_vehicle_year",
        "para_que_sirve": "Relaciona garantía con costo/edad; garantías altas pueden reflejar mayor probabilidad de falla.",
    },
    {
        "grupo": "Depreciación y spreads",
        "features": "auction_avg_depreciation, retail_avg_depreciation, acq_clean_avg_spread, current_clean_avg_spread",
        "para_que_sirve": "Mide deterioro de precio y diferencia entre condición promedio y limpia; útil para detectar pérdida de valor.",
    },
    {
        "grupo": "Calidad de información",
        "features": "mmr_missing_count",
        "para_que_sirve": "Cuenta referencias MMR faltantes; la ausencia de información también puede ser señal de incertidumbre.",
    },
]

MODEL_CARDS = [
    ("Logistic Regression", "Baseline lineal", "Modelo interpretable y rápido; exige estandarización."),
    ("Decision Tree", "Baseline árbol", "Referencia simple no lineal; ayuda a entender reglas, pero puede sobreajustar."),
    ("Random Forest", "Baseline árbol robusto", "Reduce varianza combinando muchos árboles."),
    ("SVM", "Baseline margen", "Útil para fronteras complejas; requiere escala y suele ser más costoso."),
    ("KNN", "Baseline distancia", "Compara por cercanía; sensible a escala y dimensionalidad."),
    ("XGBoost", "Avanzado boosting", "Modelo tabular competitivo; corrige errores de forma secuencial."),
    ("LightGBM", "Avanzado boosting", "Muy eficiente en datos tabulares grandes; buen balance rendimiento/tiempo."),
    ("Bagging", "Ensamble", "Reduce inestabilidad de un árbol individual entrenando múltiples árboles sobre muestras."),
    ("Voting", "Ensamble", "Combina modelos diversos promediando probabilidades."),
    ("Stacking", "Ensamble", "Usa un meta-modelo para aprender cómo combinar predictores base."),
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
    .step-box, .phase-card, .model-card, .formula-card {
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 112px;
    }
    .phase-card b, .model-card b {
        color: #0f172a;
        font-size: 1.0rem;
    }
    .phase-card small, .model-card small {
        color: #64748b;
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
    .callout {
        border-left: 5px solid #2563eb;
        background: #eff6ff;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin: 0.4rem 0 1rem 0;
    }
    .formula {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        background: #ffffff;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Utilidades generales
# =============================================================================

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


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


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


@st.cache_data(show_spinner=False)
def read_json(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def load_project_profile() -> Dict[str, Any]:
    """Resume el dataset sin exponer toda la data en memoria de UI."""
    profile = {
        "rows": None,
        "columns": None,
        "target_rate": None,
        "target_count_1": None,
        "target_count_0": None,
        "numeric_cols": None,
        "categorical_cols": None,
        "missing_total_pct": None,
        "duplicate_rows": None,
    }
    if not RAW_DATA_PATH.exists():
        return profile
    df = pd.read_csv(RAW_DATA_PATH)
    profile["rows"] = len(df)
    profile["columns"] = len(df.columns)
    profile["numeric_cols"] = len(df.select_dtypes(include="number").columns)
    profile["categorical_cols"] = len(df.select_dtypes(exclude="number").columns)
    profile["missing_total_pct"] = float(df.isna().sum().sum() / max(df.size, 1))
    profile["duplicate_rows"] = int(df.duplicated().sum())
    if TARGET in df.columns:
        counts = df[TARGET].value_counts().to_dict()
        profile["target_count_1"] = int(counts.get(1, 0))
        profile["target_count_0"] = int(counts.get(0, 0))
        profile["target_rate"] = safe_div(counts.get(1, 0), len(df))
    return profile


def get_api_status() -> Tuple[Dict[str, Any], Dict[str, Any], str | None]:
    try:
        health = call_get("/health")
        version = call_get("/version")
        return health, version, None
    except Exception as exc:  # noqa: BLE001 - UI debe mostrar error amigable
        return {}, {}, str(exc)


# =============================================================================
# Utilidades UI compartidas
# =============================================================================

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
    """Convierte la explicación de la API en una tabla comercial.

    `impact` es la influencia local de la variable en la predicción del vehículo:
    - valor positivo: empuja hacia mayor riesgo de BadBuy;
    - valor negativo: reduce el riesgo estimado;
    - si la API usa fallback de importancia no firmada, se marca como factor relevante.
    """
    if not items:
        return pd.DataFrame()

    direction_labels = {
        "sube_riesgo": "Aumenta riesgo",
        "baja_riesgo": "Reduce riesgo",
        "factor_relevante": "Factor relevante",
    }

    rows = []
    for item in items:
        impact = float(item.get("impact") or 0.0)
        impact_abs = item.get("impact_abs")
        impact_abs = abs(impact) if impact_abs is None else float(impact_abs)
        direction_code = item.get("impact_direction", "factor_relevante")
        direction = direction_labels.get(direction_code, str(direction_code))

        if direction_code == "sube_riesgo":
            lectura = "Empuja esta predicción hacia mayor riesgo."
        elif direction_code == "baja_riesgo":
            lectura = "Compensa la alerta y reduce el riesgo estimado."
        else:
            lectura = "Variable relevante para la evaluación del caso."

        rows.append(
            {
                "Factor": item.get("display_name") or item.get("feature"),
                "Valor del vehículo": item.get("value"),
                "Influencia en riesgo": impact,
                "Influencia absoluta": impact_abs,
                "Dirección": direction,
                "Lectura comercial": lectura,
                "Detalle técnico": item.get("detail"),
            }
        )

    return pd.DataFrame(rows).sort_values("Influencia absoluta", ascending=False)


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


# =============================================================================
# Vista 1: Presentación ejecutiva
# =============================================================================

def render_phase_card(num: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="phase-card">
            <small>Paso {num}</small><br>
            <b>{title}</b><br>
            <span class="small-muted">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(name: str, kind: str, reason: str) -> None:
    st.markdown(
        f"""
        <div class="model-card">
            <b>{name}</b><br>
            <small>{kind}</small><br>
            <span class="small-muted">{reason}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_presentation_dashboard() -> None:
    metadata = read_json(str(METADATA_PATH))
    ranking = read_csv(str(RANKING_PATH))
    final_metrics = read_csv(str(FINAL_METRICS_PATH))
    profile = load_project_profile()
    test_metrics = metadata.get("test_metrics", {})
    assumptions = metadata.get("business_assumptions", {})
    model_name = metadata.get("model_name", "Modelo final")

    st.markdown('<div class="main-title">📊 Presentación ejecutiva del proyecto Bad Buy</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Vista de apoyo para contar en 5 minutos qué se hizo antes de mostrar el dashboard comercial.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="callout">
        <b>Mensaje central:</b> el proyecto transforma un problema técnico de clasificación en una decisión comercial clara:
        identificar vehículos con alto riesgo de ser una mala compra y reducir pérdidas económicas antes de adquirirlos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Contexto de negocio y objetivo")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_phase_card("1", "Dolor de negocio", "Comprar un vehículo defectuoso creyendo que está en buen estado genera reparación, garantía, pérdida económica y riesgo reputacional.")
    with c2:
        render_phase_card("2", "Variable objetivo", "IsBadBuy = 1 representa una mala compra. El modelo estima la probabilidad de que un vehículo sea riesgoso.")
    with c3:
        render_phase_card("3", "Decisión esperada", "No mostramos solo una probabilidad: convertimos el resultado en semáforo para decidir continuar, revisar o detener.")

    st.subheader("2. Datos, EDA y principales hallazgos")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Registros", fmt_number(profile.get("rows")))
    with m2:
        render_metric_card("Variables originales", fmt_number(profile.get("columns")))
    with m3:
        render_metric_card("Tasa Bad Buy", percent(profile.get("target_rate")))
    with m4:
        render_metric_card("Nulos totales", percent(profile.get("missing_total_pct")))

    eda_cols = st.columns(2)
    with eda_cols[0]:
        st.markdown(
            """
            **Qué revisamos en EDA**
            - Balance de clases para confirmar desbalance del target.
            - Nulos por columna y posibles mecanismos de ausencia.
            - Distribuciones de costo, kilometraje, antigüedad y garantía.
            - Relación de variables con `IsBadBuy`.
            - Variables categóricas como subasta, marca, tamaño, nacionalidad y estado.
            """
        )
    with eda_cols[1]:
        if profile.get("target_count_0") is not None:
            target_df = pd.DataFrame(
                {
                    "Clase": ["Good Buy (0)", "Bad Buy (1)"],
                    "Registros": [profile.get("target_count_0", 0), profile.get("target_count_1", 0)],
                }
            )
            fig_target = px.bar(
                target_df,
                x="Clase",
                y="Registros",
                text="Registros",
                title="Distribución del target",
            )
            fig_target.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            st.info("No se encontró el target en el archivo local para graficar el balance.")

    st.subheader("3. Limpieza, preprocesamiento y prevención de data leakage")
    p1, p2, p3 = st.columns(3)
    with p1:
        render_phase_card("1", "Limpieza", "Se corrigen tipos, fechas, nulos y valores inconsistentes sin borrar información crítica sin justificación.")
    with p2:
        render_phase_card("2", "Pipeline", "Las transformaciones viven en un pipeline para que train, test, API y dashboard usen exactamente el mismo flujo.")
    with p3:
        render_phase_card("3", "Sin leakage", "El split se hace antes de imputar, escalar o balancear. Todo lo que aprende del dato se ajusta solo con train/folds.")

    st.subheader("4. Feature engineering: qué variables creamos y por qué")
    st.dataframe(pd.DataFrame(FEATURE_GROUPS), use_container_width=True, hide_index=True)
    st.caption(
        f"El metadata del modelo registra {metadata.get('engineered_features_count', 'N/A')} features derivadas. "
        "La API recibe columnas originales y el pipeline crea estas variables automáticamente."
    )

    st.subheader("5. Estrategia de modelado")
    st.markdown(
        """
        Se siguió la estructura de los sprints: primero baselines para tener referencia, luego tuning, modelos avanzados y ensambles.
        La comparación no se hizo solo por accuracy, porque el problema está desbalanceado y los errores tienen costos distintos.
        """
    )
    model_cols = st.columns(5)
    for i, (name, kind, reason) in enumerate(MODEL_CARDS):
        with model_cols[i % 5]:
            render_model_card(name, kind, reason)

    st.markdown("**Tratamiento por familia de modelo**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Familia": "Lineales / distancia",
                    "Modelos": "Logistic Regression, SVM, KNN",
                    "Tratamiento": "Imputación + OneHot/Ordinal + StandardScaler + class_weight o balanceo dentro de CV.",
                    "Motivo": "Son sensibles a escala y a distancia entre observaciones.",
                },
                {
                    "Familia": "Árboles",
                    "Modelos": "Decision Tree, Random Forest, Bagging",
                    "Tratamiento": "Imputación + encoding; no requieren estandarización obligatoria.",
                    "Motivo": "Particionan por umbrales y capturan no linealidades.",
                },
                {
                    "Familia": "Boosting",
                    "Modelos": "XGBoost, LightGBM",
                    "Tratamiento": "Hiperparámetros de peso de clase / scale_pos_weight y tuning.",
                    "Motivo": "Suelen ser fuertes en data tabular y manejar relaciones complejas.",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("6. Optimización y ensambles")
    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        render_phase_card("1", "RandomizedSearchCV", "Explora muchas combinaciones de hiperparámetros sin el costo de GridSearch exhaustivo.")
    with opt2:
        render_phase_card("2", "Optuna / TPE", "Búsqueda bayesiana: aprende de trials previos para explorar regiones prometedoras del espacio.")
    with opt3:
        render_phase_card("3", "Ensamblados", "Bagging, Voting y Stacking comparan si combinar modelos mejora estabilidad o valor económico.")

    st.subheader("7. Función de negocio: cómo elegimos el mejor modelo")
    st.markdown(
        f"""
        <div class="formula-card">
        <b>Objetivo:</b> elegir el modelo que maximiza valor económico, no el que solo sube una métrica técnica.<br>
        <div class="formula">Valor = TP×{assumptions.get('benefit_tp', 2500)} + TN×{assumptions.get('benefit_tn', 600)} + FP×({assumptions.get('cost_fp', -900)}) + FN×({assumptions.get('cost_fn', -4500)})</div>
        <span class="small-muted">La penalización de FN es la más alta porque representa comprar un vehículo defectuoso creyendo que era bueno.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bv_cols = st.columns(4)
    with bv_cols[0]:
        render_metric_card("TP", f"+{money(assumptions.get('benefit_tp', 2500))}")
    with bv_cols[1]:
        render_metric_card("TN", f"+{money(assumptions.get('benefit_tn', 600))}")
    with bv_cols[2]:
        render_metric_card("FP", money(assumptions.get("cost_fp", -900)), "#dc2626")
    with bv_cols[3]:
        render_metric_card("FN", money(assumptions.get("cost_fn", -4500)), "#dc2626")

    st.subheader("8. Comparación de modelos y resultado final")
    if not ranking.empty:
        view = ranking.copy()
        view["stage_label"] = view["stage"].map(STAGE_LABELS).fillna(view["stage"])
        view = view.sort_values("business_value", ascending=False).head(10)
        fig_rank = px.bar(
            view.sort_values("business_value"),
            x="business_value",
            y="model",
            orientation="h",
            color="stage_label",
            title="Top modelos por valor de negocio en validación muestral",
            labels={"business_value": "Valor de negocio", "model": "Modelo", "stage_label": "Etapa"},
        )
        fig_rank.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_rank, use_container_width=True)

        cols_to_show = [
            "stage_label",
            "model",
            "business_value",
            "recall",
            "precision",
            "f2",
            "roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        cols_to_show = [c for c in cols_to_show if c in view.columns]
        st.dataframe(view[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.info("No se encontró `reports/business_model_ranking.csv` para graficar la comparación.")

    st.markdown("**Modelo final validado**")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        render_metric_card("Modelo final", model_name)
    with f2:
        render_metric_card("Recall clase 1", percent(test_metrics.get("recall")))
    with f3:
        render_metric_card("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}" if test_metrics else "N/A")
    with f4:
        render_metric_card("Valor negocio test", money(test_metrics.get("business_value")))

    cm1, cm2 = st.columns([0.9, 1.1])
    with cm1:
        if test_metrics:
            z = [[test_metrics.get("tn", 0), test_metrics.get("fp", 0)], [test_metrics.get("fn", 0), test_metrics.get("tp", 0)]]
            fig_cm = go.Figure(data=go.Heatmap(z=z, x=["Pred 0", "Pred 1"], y=["Real 0", "Real 1"], text=z, texttemplate="%{text}", colorscale="Blues"))
            fig_cm.update_layout(title="Matriz de confusión en test final", height=330, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
    with cm2:
        st.markdown(
            f"""
            **Por qué se selecciona `{model_name}`**
            - Maximiza el criterio de negocio bajo la matriz costo–beneficio definida.
            - Mantiene un recall relevante para la clase 1, reduciendo el riesgo de dejar pasar malas compras.
            - Como ensamble tipo Bagging, reduce la inestabilidad de un árbol individual.
            - Fue reentrenado y validado en test final antes de ser guardado como `models/final_model.pkl`.
            """
        )

    st.subheader("9. MLOps y despliegue MVP")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        render_phase_card("1", "MLflow", "Registra métricas, parámetros, función de negocio y artefactos del modelo.")
    with d2:
        render_phase_card("2", "FastAPI", "Expone /health, /version, /predict y /predict_batch para consumir el modelo.")
    with d3:
        render_phase_card("3", "Docker", "Permite ejecutar la API de forma reproducible y lista para AWS.")
    with d4:
        render_phase_card("4", "Streamlit", "Traduce la predicción en semáforo comercial para el usuario final.")

    st.markdown("---")
    st.markdown("### Siguiente paso de la demo")
    st.write("Una vez explicado el proceso, abre el segundo link de la demo: **Dashboard comercial operativo**.")
    st.info("Link recomendado si lo corres en paralelo: http://localhost:8502")


# =============================================================================
# Vista 2: Dashboard comercial operativo
# =============================================================================

def _render_commercial_evaluation(api_error: str | None) -> None:
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

    if api_error:
        st.error("No se pudo conectar con la API. Primero levanta Docker/FastAPI y vuelve a cargar el dashboard.")
        st.caption(api_error)
        return

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
        return

    with st.expander("Vista previa del archivo cargado", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    if st.button("🚦 Evaluar vehículos", type="primary", use_container_width=True):
        pred = predict_df(df, limit=limit)
        st.session_state["pred"] = pred
    else:
        pred = st.session_state.get("pred", pd.DataFrame())

    if pred.empty:
        return

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
        st.markdown("**Factores que más influyeron en esta predicción**")
        st.caption(
            "La influencia es local para el vehículo seleccionado. Valores positivos aumentan el riesgo estimado; valores negativos lo reducen."
        )
        items = selected_row.get("Factores clave", [])
        factors = factors_to_df(items)
        if not factors.empty:
            plot_df = factors.head(5).copy()
            fig2 = px.bar(
                plot_df,
                x="Influencia en riesgo",
                y="Factor",
                orientation="h",
                color="Dirección",
                text="Valor del vehículo",
                title="Influencia local de variables en el vehículo seleccionado",
                color_discrete_map={
                    "Aumenta riesgo": "#dc2626",
                    "Reduce riesgo": "#16a34a",
                    "Factor relevante": "#64748b",
                },
            )
            fig2.add_vline(x=0, line_width=1, line_dash="dash", line_color="#334155")
            fig2.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Influencia sobre score de riesgo",
                yaxis_title="Factor",
            )
            st.plotly_chart(fig2, use_container_width=True)

            table_cols = [
                "Factor",
                "Valor del vehículo",
                "Influencia en riesgo",
                "Dirección",
                "Lectura comercial",
            ]
            table = factors[table_cols].copy()
            table["Influencia en riesgo"] = table["Influencia en riesgo"].map(lambda x: f"{x:+.4f}")
            st.dataframe(table, hide_index=True, use_container_width=True)

            method = selected_row.get("Método explicación")
            if method and "fallback" in str(method):
                st.caption(
                    "Método de explicación fallback: muestra factores relevantes, pero la dirección puede ser aproximada."
                )
            else:
                st.caption(
                    "Método de explicación local compatible con SHAP: la dirección indica si el factor aumenta o reduce el riesgo estimado."
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




# =============================================================================
# Vista 3: Análisis global del modelo para el dashboard comercial
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_final_model() -> Any:
    """Carga el modelo final para análisis global en Streamlit."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo final en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _features_step(model: Any) -> Any:
    return model.named_steps.get("features") if hasattr(model, "named_steps") else None


def _preprocess_step(model: Any) -> Any:
    return model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None


def _classifier_step(model: Any) -> Any:
    return model.named_steps.get("clf") if hasattr(model, "named_steps") else model


def _as_dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _extract_class1_shap_values(values: Any, n_rows: int, n_features: int) -> np.ndarray:
    """Normaliza outputs de SHAP a matriz (n_rows, n_features) para clase 1."""
    if isinstance(values, list):
        values = values[-1]

    arr = np.asarray(values)

    if arr.ndim == 3:
        # shap reciente: (n_rows, n_features, n_classes)
        if arr.shape[0] == n_rows and arr.shape[1] == n_features:
            return arr[:, :, -1].astype(float)
        # alternativa: (n_classes, n_rows, n_features)
        if arr.shape[1] == n_rows and arr.shape[2] == n_features:
            return arr[-1, :, :].astype(float)

    if arr.ndim == 2:
        if arr.shape == (n_rows, n_features):
            return arr.astype(float)
        if arr.shape == (n_features, n_rows):
            return arr.T.astype(float)

    raise ValueError(f"Formato SHAP no esperado: {arr.shape}")


def _base_feature_name(transformed_feature: str) -> str:
    """Agrupa nombres técnicos de ColumnTransformer/OHE a variables de negocio."""
    name = str(transformed_feature)
    if "__" in name:
        name = name.split("__", 1)[1]
    if name in FEATURE_FRIENDLY_NAMES:
        return name
    for prefix in sorted(FEATURE_FRIENDLY_NAMES, key=len, reverse=True):
        if name.startswith(prefix + "_"):
            return prefix
    return name


def _prepare_model_matrix(model: Any, sample: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str], Any]:
    """Devuelve muestra cruda, matriz transformada, nombres transformados y clasificador."""
    metadata = read_json(str(METADATA_PATH))
    raw_features = metadata.get("raw_features") or [c for c in sample.columns if c != TARGET]
    cols = [c for c in raw_features if c in sample.columns]
    X_raw = sample[cols].copy()

    features = _features_step(model)
    preprocess = _preprocess_step(model)
    clf = _classifier_step(model)

    X_features = features.transform(X_raw) if features is not None else X_raw
    if not isinstance(X_features, pd.DataFrame):
        X_features = pd.DataFrame(X_features)

    if preprocess is not None:
        transformed = preprocess.transform(X_features)
        try:
            names = [str(x) for x in preprocess.get_feature_names_out()]
        except Exception:
            names = [f"feature_{i}" for i in range(_as_dense(transformed).shape[1])]
    else:
        transformed = X_features
        names = [str(c) for c in X_features.columns]

    return X_features, _as_dense(transformed), names, clf


def _bagging_global_shap(clf: Any, X_dense: np.ndarray, max_estimators: int) -> tuple[np.ndarray | None, str]:
    """Calcula SHAP global promediando árboles internos de Bagging."""
    if not hasattr(clf, "estimators_"):
        return None, "not_bagging"

    try:
        import shap  # type: ignore
    except Exception as exc:
        return None, f"shap_not_available: {exc}"

    n_rows, n_features = X_dense.shape
    estimators_features = getattr(clf, "estimators_features_", None)
    total = np.zeros((n_rows, n_features), dtype=float)
    valid = 0

    for idx, estimator in enumerate(clf.estimators_[:max_estimators]):
        try:
            selected = (
                np.asarray(estimators_features[idx], dtype=int)
                if estimators_features is not None
                else np.arange(n_features, dtype=int)
            )
            X_sub = X_dense[:, selected]
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(X_sub)
            shap_sub = _extract_class1_shap_values(values, n_rows=n_rows, n_features=len(selected))
            mapped = np.zeros((n_rows, n_features), dtype=float)
            mapped[:, selected] = shap_sub
            total += mapped
            valid += 1
        except Exception:
            continue

    if valid == 0:
        return None, "bagging_shap_failed"
    return total / valid, f"bagging_tree_shap_average_{valid}_estimators"


def _direct_global_shap(clf: Any, X_dense: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Calcula SHAP directo para modelos compatibles con TreeExplainer."""
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(X_dense)
        return _extract_class1_shap_values(values, X_dense.shape[0], X_dense.shape[1]), "shap.TreeExplainer"
    except Exception as exc:
        return None, f"shap_failed: {exc}"


@st.cache_data(show_spinner=False)
def compute_global_shap_report(sample_size: int = 150, max_estimators: int = 30) -> Dict[str, Any]:
    """Calcula un reporte SHAP global sobre una muestra del test final."""
    if not DEFAULT_SAMPLE.exists():
        return {"error": f"No se encontró muestra en {DEFAULT_SAMPLE}"}

    sample = pd.read_csv(DEFAULT_SAMPLE).head(max(sample_size, 1)).copy()
    if sample.empty:
        return {"error": "La muestra de evaluación está vacía."}

    model = load_final_model()
    X_features, X_dense, transformed_names, clf = _prepare_model_matrix(model, sample)

    shap_values, method = _bagging_global_shap(clf, X_dense, max_estimators=max_estimators)
    if shap_values is None:
        shap_values, method = _direct_global_shap(clf, X_dense)

    if shap_values is None:
        # Fallback global: importancias del clasificador. No es SHAP, pero evita vista vacía.
        return {"error": f"No se pudo calcular SHAP global ({method}). Verificar instalación de shap y compatibilidad del modelo."}

    base_names = [_base_feature_name(name) for name in transformed_names]
    unique_bases = list(dict.fromkeys(base_names))

    # Agregación de columnas transformadas al nivel de variable de negocio.
    shap_grouped = np.zeros((shap_values.shape[0], len(unique_bases)), dtype=float)
    value_grouped = np.zeros((shap_values.shape[0], len(unique_bases)), dtype=float)

    for group_idx, base in enumerate(unique_bases):
        indices = [i for i, b in enumerate(base_names) if b == base]
        shap_grouped[:, group_idx] = shap_values[:, indices].sum(axis=1)

        if base in X_features.columns and pd.api.types.is_numeric_dtype(X_features[base]):
            raw_values = pd.to_numeric(X_features[base], errors="coerce").fillna(0).to_numpy(dtype=float)
            value_grouped[:, group_idx] = raw_values
        else:
            # Para categóricas/OHE se usa activación agregada de columnas transformadas.
            value_grouped[:, group_idx] = np.abs(X_dense[:, indices]).sum(axis=1)

    mean_abs = np.mean(np.abs(shap_grouped), axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_order = order[:12]

    importance_rows = []
    for idx in top_order:
        base = unique_bases[idx]
        importance_rows.append(
            {
                "Variable": FEATURE_FRIENDLY_NAMES.get(base, base),
                "Variable técnica": base,
                "Impacto medio absoluto": float(mean_abs[idx]),
                "Impacto promedio": float(np.mean(shap_grouped[:, idx])),
            }
        )

    bees_rows = []
    for idx in top_order:
        base = unique_bases[idx]
        display = FEATURE_FRIENDLY_NAMES.get(base, base)
        values = value_grouped[:, idx]
        v_min, v_max = float(np.nanmin(values)), float(np.nanmax(values))
        if np.isclose(v_min, v_max):
            norm = np.full_like(values, 0.5, dtype=float)
        else:
            norm = (values - v_min) / (v_max - v_min)
        for shap_value, raw_value, norm_value in zip(shap_grouped[:, idx], values, norm):
            bees_rows.append(
                {
                    "Variable": display,
                    "Impacto SHAP": float(shap_value),
                    "Valor normalizado": float(norm_value),
                    "Valor usado": float(raw_value) if np.isfinite(raw_value) else None,
                }
            )

    return {
        "method": method,
        "sample_size": int(len(sample)),
        "importance": importance_rows,
        "beeswarm": bees_rows,
    }


def render_model_analysis_view() -> None:
    """Vista técnica separada con SHAP global del modelo."""
    st.markdown('<div class="main-title">🔎 Análisis global del modelo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Interpretabilidad del modelo final a nivel global. Esta vista explica qué variables influyen más en el comportamiento general del modelo.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="callout">
        <b>Lectura:</b> cada punto representa un vehículo de la muestra de evaluación. La posición horizontal indica cuánto empuja una variable la predicción hacia mayor o menor riesgo de BadBuy. El color resume si el valor de esa variable es bajo o alto dentro de la muestra.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        sample_size = st.slider("Vehículos para análisis", min_value=50, max_value=500, value=150, step=50)
    with c2:
        max_estimators = st.slider("Árboles internos a promediar", min_value=5, max_value=100, value=30, step=5)
    with c3:
        st.caption("A mayor muestra y más árboles, el análisis es más estable pero puede tardar más.")

    with st.spinner("Calculando análisis global del modelo..."):
        report = compute_global_shap_report(sample_size=sample_size, max_estimators=max_estimators)

    if report.get("error"):
        st.warning(report["error"])
        st.info("El dashboard comercial sigue funcionando. Esta sección requiere que `shap` esté instalado y que el modelo sea compatible con TreeExplainer o con explicación promedio de Bagging.")
        return

    st.caption(f"Método de explicación: `{report.get('method')}` · muestra usada: {report.get('sample_size')} vehículos")

    bees_df = pd.DataFrame(report.get("beeswarm", []))
    importance_df = pd.DataFrame(report.get("importance", []))

    if bees_df.empty or importance_df.empty:
        st.info("No se generaron datos suficientes para graficar el análisis global.")
        return

    fig_bees = px.scatter(
        bees_df,
        x="Impacto SHAP",
        y="Variable",
        color="Valor normalizado",
        color_continuous_scale=["#f59e0b", "#7c3aed"],
        opacity=0.72,
        title="SHAP global del modelo final",
        labels={"Valor normalizado": "Valor de variable"},
    )
    fig_bees.add_vline(x=0, line_width=1, line_dash="dash", line_color="#334155")
    fig_bees.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=60, b=20),
        yaxis={"categoryorder": "array", "categoryarray": importance_df["Variable"].iloc[::-1].tolist()},
    )
    st.plotly_chart(fig_bees, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        fig_imp = px.bar(
            importance_df.sort_values("Impacto medio absoluto"),
            x="Impacto medio absoluto",
            y="Variable",
            orientation="h",
            title="Importancia global promedio",
        )
        fig_imp.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)
    with right:
        st.markdown("**Top variables globales**")
        table = importance_df.copy()
        table["Impacto medio absoluto"] = table["Impacto medio absoluto"].map(lambda x: f"{x:.5f}")
        table["Impacto promedio"] = table["Impacto promedio"].map(lambda x: f"{x:+.5f}")
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Diferencia frente a la explicación local del semáforo:**  
        - Esta vista resume el comportamiento general del modelo sobre una muestra.  
        - La explicación local del dashboard comercial muestra por qué un vehículo específico salió verde, ámbar o rojo.
        """
    )


def render_commercial_dashboard(api_error: str | None) -> None:
    """Renderiza el dashboard comercial con una vista separada de análisis del modelo."""
    tab_commercial, tab_model = st.tabs(["🚦 Evaluación comercial", "🔎 Análisis del modelo"])
    with tab_commercial:
        _render_commercial_evaluation(api_error)
    with tab_model:
        render_model_analysis_view()
