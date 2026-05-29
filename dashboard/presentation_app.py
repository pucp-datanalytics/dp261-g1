"""Dashboard de presentación ejecutiva del proyecto DP261.

Aplicación Streamlit independiente para resumir el proceso CRISP-DM,
la estrategia de modelado, la función de negocio y los resultados finales.

Uso:
    streamlit run dashboard/presentation_app.py --server.port 8501
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="DP261 · Presentación ejecutiva",
    page_icon="📊",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
RANKING_PATH = PROJECT_ROOT / "reports" / "business_model_ranking.csv"
FINAL_METRICS_PATH = PROJECT_ROOT / "reports" / "final_validation_metrics.csv"


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Carga un archivo JSON de forma segura."""
    if default is None:
        default = {}
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def load_csv(path: Path) -> pd.DataFrame:
    """Carga un CSV de forma segura."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def fmt_usd(value: float | int | None) -> str:
    """Formatea montos en dólares para visualización ejecutiva."""
    if value is None:
        return "No disponible"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"USD {value / 1_000_000:.2f}MM"
    if abs(value) >= 1_000:
        return f"USD {value / 1_000:.1f}K"
    return f"USD {value:,.0f}"


def fmt_pct(value: float | int | None) -> str:
    """Formatea ratios como porcentajes."""
    if value is None:
        return "No disponible"
    return f"{float(value) * 100:.1f}%"


def get_test_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    """Obtiene las métricas finales almacenadas en model_metadata.json."""
    return metadata.get("test_metrics", {}) if isinstance(metadata, dict) else {}


def get_assumptions(metadata: dict[str, Any]) -> dict[str, Any]:
    """Obtiene los supuestos económicos almacenados en model_metadata.json."""
    return metadata.get("business_assumptions", {}) if isinstance(metadata, dict) else {}


# ---------------------------------------------------------------------
# Componentes visuales
# ---------------------------------------------------------------------

def render_technical_sidebar(metadata: dict[str, Any]) -> None:
    """Muestra el panel técnico en el sidebar."""
    with st.sidebar:
        st.title("Panel técnico")
        st.caption("Información de soporte para revisión técnica.")

        st.subheader("Modelo")
        st.write(f"**Modelo final:** {metadata.get('model_name', 'No disponible')}")
        st.write(f"**Versión:** {metadata.get('model_version', 'No disponible')}")
        st.write(f"**Threshold:** {metadata.get('threshold', 'No disponible')}")

        st.subheader("Artefactos")
        artifacts = {
            "Metadata del modelo": METADATA_PATH,
            "Ranking de modelos": RANKING_PATH,
            "Métricas finales": FINAL_METRICS_PATH,
        }

        for label, path in artifacts.items():
            status = "✅ Disponible" if path.exists() else "⚠️ No encontrado"
            st.write(f"{status} · {label}")

        with st.expander("Metadata completa", expanded=False):
            st.json(metadata)


def render_card(label: str, title: str, text: str) -> None:
    """Renderiza una tarjeta informativa simple."""
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e6e6e6;
            border-radius: 14px;
            padding: 18px 18px;
            min-height: 145px;
            background-color: #ffffff;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        ">
            <div style="font-size: 13px; color: #666;">{label}</div>
            <div style="font-size: 19px; font-weight: 700; margin-top: 4px;">{title}</div>
            <div style="font-size: 15px; color: #333; margin-top: 8px; line-height: 1.35;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_ranking(ranking: pd.DataFrame) -> None:
    """Muestra el ranking de modelos por valor de negocio."""
    if ranking.empty or "model" not in ranking.columns or "business_value" not in ranking.columns:
        st.info("Ranking de modelos no disponible. Ejecutar el flujo de entrenamiento para regenerar los reportes.")
        return

    top = ranking.sort_values("business_value", ascending=False).head(8).copy()

    fig = go.Figure(
        go.Bar(
            x=top["business_value"],
            y=top["model"],
            orientation="h",
            text=[fmt_usd(v) for v in top["business_value"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Ranking de modelos por valor de negocio",
        xaxis_title="Valor económico estimado",
        yaxis_title="Modelo",
        yaxis=dict(autorange="reversed"),
        height=430,
        margin=dict(l=20, r=20, t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_business_breakdown(metrics: dict[str, Any], assumptions: dict[str, Any]) -> None:
    """Muestra cómo cada cuadrante de la matriz de confusión aporta al valor económico."""
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tp = int(metrics.get("tp", 0))

    coefficients = {
        "Verdaderos positivos (TP)": float(assumptions.get("benefit_tp", 2500)),
        "Verdaderos negativos (TN)": float(assumptions.get("benefit_tn", 600)),
        "Falsos positivos (FP)": float(assumptions.get("cost_fp", -900)),
        "Falsos negativos (FN)": float(assumptions.get("cost_fn", -4500)),
    }

    quantities = {
        "Verdaderos positivos (TP)": tp,
        "Verdaderos negativos (TN)": tn,
        "Falsos positivos (FP)": fp,
        "Falsos negativos (FN)": fn,
    }

    rows = []
    for component, quantity in quantities.items():
        coefficient = coefficients[component]
        rows.append(
            {
                "Componente": component,
                "Cantidad": quantity,
                "Coeficiente económico": coefficient,
                "Impacto USD": quantity * coefficient,
            }
        )

    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.format(
            {
                "Cantidad": "{:,.0f}",
                "Coeficiente económico": "USD {:,.0f}",
                "Impacto USD": "USD {:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_metrics_table(metrics: dict[str, Any]) -> None:
    """Muestra las métricas finales más relevantes."""
    rows = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision clase 1", metrics.get("precision")),
        ("Recall clase 1", metrics.get("recall")),
        ("F1", metrics.get("f1")),
        ("F2", metrics.get("f2")),
        ("ROC-AUC", metrics.get("roc_auc")),
        ("Average precision", metrics.get("average_precision")),
    ]

    df = pd.DataFrame(rows, columns=["Métrica", "Valor"])
    df["Valor"] = df["Valor"].apply(lambda x: "No disponible" if x is None else f"{float(x):.3f}")

    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Carga de información del proyecto
# ---------------------------------------------------------------------

metadata = load_json(METADATA_PATH)
ranking = load_csv(RANKING_PATH)
metrics = get_test_metrics(metadata)
assumptions = get_assumptions(metadata)

model_name = metadata.get("model_name", "No disponible")
business_value = metrics.get("business_value")
recall = metrics.get("recall")
precision = metrics.get("precision")
roc_auc = metrics.get("roc_auc")


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

render_technical_sidebar(metadata)

st.title("Presentación ejecutiva · Proyecto DP261")
st.markdown(
    """
    Sistema predictivo para apoyar la decisión de compra de vehículos usados en subastas.
    El objetivo es identificar vehículos con mayor riesgo de mala compra y convertir el
    desempeño del modelo en una recomendación comercial accionable.
    """
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Modelo final", model_name)
kpi2.metric("Impacto económico", fmt_usd(business_value))
kpi3.metric("Recall clase 1", fmt_pct(recall))
kpi4.metric("ROC-AUC", f"{float(roc_auc):.3f}" if roc_auc is not None else "No disponible")

st.divider()


# ---------------------------------------------------------------------
# 1. Contexto de negocio
# ---------------------------------------------------------------------

st.header("1. Contexto de negocio")
c1, c2, c3 = st.columns(3)

with c1:
    render_card(
        "Problema",
        "Riesgo en compras",
        "Comprar un vehículo defectuoso genera costos de reparación, garantías, pérdida comercial y deterioro reputacional.",
    )

with c2:
    render_card(
        "Objetivo",
        "Priorizar malas compras",
        "El modelo estima el riesgo de que un vehículo sea una mala compra antes de aceptar la adquisición.",
    )

with c3:
    render_card(
        "Decisión",
        "Semáforo comercial",
        "La predicción se traduce a una recomendación simple: proceder, revisar manualmente o detener la compra.",
    )


# ---------------------------------------------------------------------
# 2. Preparación de datos
# ---------------------------------------------------------------------

st.header("2. Preparación de datos")
st.markdown(
    """
    La preparación siguió un flujo reproducible: entendimiento del dataset, limpieza,
    creación de variables, separación train/test y encapsulamiento del preprocesamiento
    dentro de pipelines para evitar data leakage.
    """
)

prep1, prep2, prep3, prep4 = st.columns(4)

with prep1:
    render_card(
        "EDA",
        "Entendimiento de variables",
        "Revisión de nulos, duplicados, balance del target, outliers, distribuciones y relación de variables con `IsBadBuy`.",
    )

with prep2:
    render_card(
        "Limpieza",
        "Datos consistentes",
        "Tratamiento de valores faltantes, tipos de datos, variables categóricas y variables numéricas según su naturaleza.",
    )

with prep3:
    render_card(
        "Feature engineering",
        "Variables de negocio",
        "Creación de señales de antigüedad, kilometraje, garantía, depreciación, márgenes frente a MMR y calidad de información.",
    )

with prep4:
    render_card(
        "Pipelines",
        "Flujo reproducible",
        "Cada modelo recibe datos crudos y aplica internamente limpieza, features, encoding, escalado y balanceo cuando corresponde.",
    )

with st.expander("Variables creadas y justificación", expanded=False):
    feature_rows = [
        ("odo_per_year", "Kilometraje anualizado para comparar uso relativo entre autos de distintas edades."),
        ("old_high_mileage_flag", "Señal de riesgo cuando un vehículo combina antigüedad y alto kilometraje."),
        ("cost_to_acq_auction_avg", "Relación entre costo de compra y valor promedio de subasta al momento de adquisición."),
        ("acq_auction_margin", "Diferencia entre costo del vehículo y precio MMR de adquisición."),
        ("cost_to_current_auction_avg", "Relación entre costo y valor actual de subasta."),
        ("current_auction_margin", "Margen frente al precio actual estimado por MMR."),
        ("warranty_to_cost", "Peso relativo del costo de garantía frente al costo del vehículo."),
        ("warranty_per_vehicle_year", "Costo de garantía ajustado por antigüedad del vehículo."),
        ("auction_avg_depreciation", "Cambio entre valor MMR de adquisición y valor MMR actual en subasta."),
        ("retail_avg_depreciation", "Cambio entre valor MMR de adquisición y valor MMR actual en retail."),
        ("acq_clean_avg_spread", "Diferencia entre precio clean y promedio en adquisición."),
        ("current_clean_avg_spread", "Diferencia entre precio clean y promedio actual."),
        ("mmr_missing_count", "Cantidad de precios MMR faltantes como indicador de menor calidad de información."),
    ]

    st.dataframe(
        pd.DataFrame(feature_rows, columns=["Variable", "Justificación"]),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------
# 3. Modelado
# ---------------------------------------------------------------------

st.header("3. Estrategia de modelado")
st.markdown(
    """
    La comparación de modelos se organizó por etapas. Primero se establecieron baselines,
    luego se optimizaron hiperparámetros y finalmente se evaluaron modelos avanzados y
    ensambles.
    """
)

m1, m2, m3 = st.columns(3)

with m1:
    render_card(
        "Sprint 3",
        "Modelos baseline",
        "Logistic Regression, Decision Tree, Random Forest, SVM y KNN se usaron como punto de referencia inicial.",
    )

with m2:
    render_card(
        "Sprint 4",
        "Optimización",
        "RandomizedSearchCV y Optuna/TPE permitieron explorar configuraciones con mejor desempeño técnico y económico.",
    )

with m3:
    render_card(
        "Sprint 4",
        "Modelos avanzados",
        "XGBoost, LightGBM, Bagging, Voting y Stacking se evaluaron para capturar relaciones no lineales y reducir varianza.",
    )

with st.expander("Criterio por familia de modelos", expanded=False):
    model_logic = pd.DataFrame(
        [
            (
                "Lineales / distancia",
                "Logistic Regression, SVM, KNN",
                "Requieren escalado; se evaluó balanceo en train para tratar la clase minoritaria.",
            ),
            (
                "Árboles",
                "Decision Tree, Random Forest",
                "No requieren escalado; capturan relaciones no lineales e interacciones.",
            ),
            (
                "Boosting",
                "XGBoost, LightGBM",
                "Modelos competitivos para datos tabulares; permiten ponderar desbalance y controlar overfitting.",
            ),
            (
                "Ensambles",
                "Bagging, Voting, Stacking",
                "Combinan modelos para reducir varianza o aprovechar señales complementarias.",
            ),
        ],
        columns=["Familia", "Modelos", "Justificación"],
    )

    st.dataframe(model_logic, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# 4. Función de negocio
# ---------------------------------------------------------------------

st.header("4. Función de negocio")
st.markdown(
    """
    La selección final priorizó valor económico. La ecuación considera los cuatro
    componentes de la matriz de confusión, con mayor penalización para falsos negativos,
    porque representan vehículos defectuosos aceptados como buenas compras.
    """
)

st.latex(
    r"\text{Valor} = TP \cdot 2500 + TN \cdot 600 + FP \cdot (-900) + FN \cdot (-4500)"
)

eq1, eq2, eq3, eq4 = st.columns(4)
eq1.metric("Beneficio TP", fmt_usd(assumptions.get("benefit_tp", 2500)))
eq2.metric("Beneficio TN", fmt_usd(assumptions.get("benefit_tn", 600)))
eq3.metric("Costo FP", fmt_usd(assumptions.get("cost_fp", -900)))
eq4.metric("Costo FN", fmt_usd(assumptions.get("cost_fn", -4500)))

st.caption(
    "El falso negativo recibe la mayor penalización por ser el error que deja pasar una mala compra al proceso comercial."
)


# ---------------------------------------------------------------------
# 5. Resultados
# ---------------------------------------------------------------------

st.header("5. Resultados y selección final")
left, right = st.columns([1.1, 1])

with left:
    render_model_ranking(ranking)

with right:
    st.subheader("Modelo final")
    if roc_auc is not None:
        st.markdown(
            f"""
            **Modelo seleccionado:** `{model_name}`  
            **Threshold operativo:** `{metadata.get("threshold", 0.5)}`  
            **Valor económico en test final:** **{fmt_usd(business_value)}**  
            **Recall clase 1:** **{fmt_pct(recall)}**  
            **Precision clase 1:** **{fmt_pct(precision)}**  
            **ROC-AUC:** **{float(roc_auc):.3f}**  
            """
        )
    else:
        st.markdown(
            f"""
            **Modelo seleccionado:** `{model_name}`  
            **Threshold operativo:** `{metadata.get("threshold", 0.5)}`  
            **Valor económico en test final:** **{fmt_usd(business_value)}**
            """
        )

    st.markdown(
        """
        La selección final se basa en el valor económico obtenido bajo la función de negocio,
        no únicamente en una métrica técnica aislada. Esto permite alinear la decisión del
        modelo con el impacto esperado para la empresa.
        """
    )

    st.subheader("Métricas finales")
    render_metrics_table(metrics)

st.subheader("Descomposición económica por matriz de confusión")
render_confusion_business_breakdown(metrics, assumptions)


# ---------------------------------------------------------------------
# 6. Producto final
# ---------------------------------------------------------------------

st.header("6. Producto final")
p1, p2, p3 = st.columns(3)

with p1:
    render_card(
        "API",
        "FastAPI + Docker",
        "El modelo se expone mediante endpoints `/health`, `/version`, `/predict` y `/predict_batch`.",
    )

with p2:
    render_card(
        "Dashboard comercial",
        "Streamlit operativo",
        "El usuario carga vehículos y recibe una decisión visual por semáforo, con detalle del vehículo y factores explicativos.",
    )

with p3:
    render_card(
        "Trazabilidad",
        "MLflow",
        "El entrenamiento registra parámetros, métricas, valor económico y artefactos del modelo final.",
    )

st.success(
    "El proyecto cuenta con modelo final versionado, validación económica, API dockerizada, dashboard comercial y trazabilidad de experimentos."
)
