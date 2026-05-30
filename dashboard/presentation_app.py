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


def render_impact_card(label: str, value: str, detail: str, accent: str = "#1f4e79") -> None:
    """Tarjeta ejecutiva para resaltar resultados de validación."""
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-left: 7px solid {accent};
            border-radius: 16px;
            padding: 18px 20px;
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
            min-height: 132px;
        ">
            <div style="font-size: 14px; color: #64748b; font-weight: 600;">{label}</div>
            <div style="font-size: 30px; color: #111827; font-weight: 850; margin-top: 6px;">{value}</div>
            <div style="font-size: 14px; color: #475569; margin-top: 8px; line-height: 1.35;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confusion_matrix(metrics: dict[str, Any]) -> None:
    """Renderiza la matriz de confusión final del modelo."""
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tp = int(metrics.get("tp", 0))

    z = [[tn, fp], [fn, tp]]
    text = [
        [f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
        [f"FN<br>{fn:,}", f"TP<br>{tp:,}"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicción: Good Buy (0)", "Predicción: Bad Buy (1)"],
            y=["Real: Good Buy (0)", "Real: Bad Buy (1)"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 17},
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Matriz de confusión · Test final",
        height=390,
        margin=dict(l=20, r=20, t=60, b=30),
        xaxis_title="Clase predicha por el modelo",
        yaxis_title="Clase real",
    )
    st.plotly_chart(fig, use_container_width=True)


def build_business_breakdown_df(metrics: dict[str, Any], assumptions: dict[str, Any]) -> pd.DataFrame:
    """Construye la tabla de impacto económico por cuadrante."""
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tp = int(metrics.get("tp", 0))

    rows = [
        {
            "Cuadrante": "TP · Bad Buy detectado",
            "Cantidad": tp,
            "Coeficiente": float(assumptions.get("benefit_tp", 2500)),
            "Interpretación": "Mala compra identificada a tiempo",
        },
        {
            "Cuadrante": "TN · Good Buy aceptado",
            "Cantidad": tn,
            "Coeficiente": float(assumptions.get("benefit_tn", 600)),
            "Interpretación": "Compra correcta que puede continuar",
        },
        {
            "Cuadrante": "FP · Good Buy frenado",
            "Cantidad": fp,
            "Coeficiente": float(assumptions.get("cost_fp", -900)),
            "Interpretación": "Caso sano enviado a revisión/detenido",
        },
        {
            "Cuadrante": "FN · Bad Buy no detectado",
            "Cantidad": fn,
            "Coeficiente": float(assumptions.get("cost_fn", -4500)),
            "Interpretación": "Mala compra que se escapa del filtro",
        },
    ]
    df = pd.DataFrame(rows)
    df["Impacto USD"] = df["Cantidad"] * df["Coeficiente"]
    return df


def render_validation_summary(
    metrics: dict[str, Any],
    assumptions: dict[str, Any],
    model_name: str,
    threshold: Any,
) -> None:
    """Bloque visual de validación final para la presentación ejecutiva."""
    business_value = metrics.get("business_value")
    recall = metrics.get("recall")
    precision = metrics.get("precision")
    roc_auc = metrics.get("roc_auc")
    tp = int(metrics.get("tp", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tn = int(metrics.get("tn", 0))

    st.markdown("### Validación final en test")
    st.markdown(
        """
        La validación final se realiza sobre el conjunto de test reservado. El objetivo de esta sección
        es conectar la matriz de confusión con el impacto económico esperado.
        """
    )

    k1, k2, k3 = st.columns([1.2, 0.9, 0.9])
    with k1:
        render_impact_card(
            "Impacto económico en test final",
            fmt_usd(business_value),
            "Valor calculado desde TP, TN, FP y FN usando la función de negocio definida.",
            "#0f766e",
        )
    with k2:
        render_impact_card(
            "Recall clase 1",
            fmt_pct(recall),
            "Porcentaje de malas compras detectadas por el modelo.",
            "#2563eb",
        )
    with k3:
        render_impact_card(
            "ROC-AUC",
            f"{float(roc_auc):.3f}" if roc_auc is not None else "No disponible",
            "Capacidad general de discriminación entre Good Buy y Bad Buy.",
            "#7c3aed",
        )

    st.markdown("")
    c_left, c_right = st.columns([1.05, 0.95])
    with c_left:
        render_confusion_matrix(metrics)

    with c_right:
        st.markdown("#### Lectura ejecutiva")
        st.markdown(
            f"""
            **Modelo seleccionado:** `{model_name}`  
            **Threshold operativo:** `{threshold}`  
            **Precision clase 1:** **{fmt_pct(precision)}**  
            **Bad Buys detectados (TP):** **{tp:,}**  
            **Bad Buys no detectados (FN):** **{fn:,}**  
            **Good Buys correctamente aceptados (TN):** **{tn:,}**  
            **Good Buys enviados a revisión (FP):** **{fp:,}**
            """
        )

        st.info(
            "La decisión final prioriza valor económico: el modelo elegido no es necesariamente "
            "el de mayor métrica técnica aislada, sino el que entrega la mejor combinación entre "
            "detección de Bad Buys, continuidad de Good Buys y costo de errores."
        )

    st.markdown("#### Impacto económico por cuadrante")
    breakdown = build_business_breakdown_df(metrics, assumptions)
    st.dataframe(
        breakdown.style.format(
            {
                "Cantidad": "{:,.0f}",
                "Coeficiente": "USD {:,.0f}",
                "Impacto USD": "USD {:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    total_impact = breakdown["Impacto USD"].sum()
    st.caption(
        f"Impacto total calculado desde la matriz de confusión: {fmt_usd(total_impact)}. "
        "El costo de FN es el más alto porque representa aceptar un vehículo riesgoso como si fuera una buena compra."
    )


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

st.subheader("Comparación de modelos por valor de negocio")
render_model_ranking(ranking)

render_validation_summary(
    metrics=metrics,
    assumptions=assumptions,
    model_name=model_name,
    threshold=metadata.get("threshold", 0.5),
)

with st.expander("Ver métricas técnicas completas", expanded=False):
    render_metrics_table(metrics)


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
