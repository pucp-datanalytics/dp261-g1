"""Funciones y componentes compartidos para los dashboards Streamlit DP261.

Este módulo concentra estilos, utilidades, carga de archivos, llamadas a la API
y renderizado de bloques comunes. No define navegación: cada dashboard se corre
por separado para tener links independientes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "IsBadBuy"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "06-kickAutomotriz.csv"
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "processed" / "test_final.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
RANKING_PATH = PROJECT_ROOT / "reports" / "business_model_ranking.csv"
FINAL_METRICS_PATH = PROJECT_ROOT / "reports" / "final_validation_metrics.csv"

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

def render_commercial_dashboard(api_error: str | None) -> None:
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
        st.stop()

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


