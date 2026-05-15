from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TARGET
from src.evaluation import evaluate_thresholds, get_scores
from src.preprocessing import prepare_features

st.set_page_config(page_title="Kick Automotriz | Modelo Bad Buy", layout="wide")

MODEL_CANDIDATES = [
    MODELS_DIR / "final_model.pkl",
    MODELS_DIR / "final_LogisticRegression_threshold_0_70.pkl",
    MODELS_DIR / "tuned_LogisticRegression.pkl",
]


@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path), path
    return None, None


@st.cache_data
def load_csv_if_exists(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def get_default_threshold(metrics, comparison):
    if not metrics.empty and "threshold" in metrics.columns:
        return float(metrics.iloc[0]["threshold"])
    if not comparison.empty and "threshold" in comparison.columns:
        if "scenario" in comparison.columns:
            mask = comparison["scenario"].astype(str).str.contains("optimal|business", case=False, na=False)
            if mask.any():
                return float(comparison.loc[mask, "threshold"].iloc[0])
        return float(comparison.sort_values("business_value", ascending=False)["threshold"].iloc[0])
    return 0.70


def format_money(value):
    try:
        return f"USD {float(value):,.0f}"
    except Exception:
        return "-"


def prepare_input(raw_df):
    y_true = None
    if TARGET in raw_df.columns:
        y_true = pd.to_numeric(raw_df[TARGET], errors="coerce").astype("Int64")
        X_raw = raw_df.drop(columns=[TARGET])
    else:
        X_raw = raw_df.copy()
    X = prepare_features(X_raw)
    return X, y_true


def score_dataframe(model, raw_df, threshold):
    X, y_true = prepare_input(raw_df)
    scores = get_scores(model, X)
    out = raw_df.copy()
    out["risk_score"] = scores
    out["pred_IsBadBuy"] = (scores >= threshold).astype(int)
    out["decision"] = np.where(out["pred_IsBadBuy"].eq(1), "Revisar / rechazar", "Continuar evaluación")
    return X, y_true, scores, out


def metric_row(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan,
        "average_precision": average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan,
        "positive_rate": y_pred.mean(),
    }


def plot_confusion_matrix(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicción", y="Real", color="Casos"),
        x=["Good Buy", "Bad Buy"],
        y=["Good Buy", "Bad Buy"],
        title=f"Matriz de confusión | threshold={threshold:.2f}",
    )
    fig.update_layout(height=420)
    return fig


def plot_roc(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_value = roc_auc_score(y_true, scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Modelo AUC={auc_value:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio", line=dict(dash="dash")))
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate / Recall",
        height=420,
    )
    return fig


def plot_business_value(threshold_df, selected_threshold):
    fig = px.line(
        threshold_df.sort_values("threshold"),
        x="threshold",
        y="business_value",
        markers=True,
        title="Business Value por threshold",
    )
    best = threshold_df.loc[threshold_df["business_value"].idxmax()]
    fig.add_vline(x=selected_threshold, line_dash="dash", annotation_text=f"Actual {selected_threshold:.2f}")
    fig.add_vline(x=float(best["threshold"]), line_dash="dot", annotation_text=f"Óptimo {float(best['threshold']):.2f}")
    fig.update_layout(height=430)
    return fig


def model_feature_contributions(model, X, row_idx):
    if not hasattr(model, "named_steps"):
        raise ValueError("El modelo no es un Pipeline de sklearn.")

    preprocess = model.named_steps.get("preprocess")
    clf = model.named_steps.get("clf")
    if preprocess is None or clf is None or not hasattr(clf, "coef_"):
        raise ValueError("No se encontró preprocess + clasificador lineal con coeficientes.")

    X_transformed = preprocess.transform(X)
    feature_names = preprocess.get_feature_names_out()
    row = X_transformed[row_idx]
    row_values = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
    coefs = clf.coef_.ravel()
    contributions = row_values * coefs

    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "value_transformed": row_values,
        "coefficient": coefs,
        "contribution": contributions,
    })
    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    return contrib_df.sort_values("abs_contribution", ascending=False)


model, model_path = load_model()
metrics = load_csv_if_exists(REPORTS_DIR / "final_validation_metrics.csv")
threshold_bv = load_csv_if_exists(REPORTS_DIR / "threshold_business_value.csv")
threshold_comparison = load_csv_if_exists(REPORTS_DIR / "threshold_business_value_comparison.csv")
confusion_impact = load_csv_if_exists(REPORTS_DIR / "business_confusion_matrix_impact.csv")
final_decision = load_csv_if_exists(REPORTS_DIR / "final_decision_summary.csv")

st.title("Dashboard Sprint 5 — Bad Buy Prediction")
st.caption("Dashboard para stakeholders: KPIs, simulador, matriz de confusión, business value y explicabilidad.")

if model is None:
    st.error("No se encontró un modelo final en models/. Ejecuta primero los notebooks de validación final.")
    st.stop()

st.sidebar.header("Configuración")
st.sidebar.write(f"Modelo cargado: `{model_path.relative_to(PROJECT_ROOT)}`")
DEFAULT_THRESHOLD = get_default_threshold(metrics, threshold_comparison)
threshold = st.sidebar.slider("Threshold operativo", 0.01, 0.99, float(DEFAULT_THRESHOLD), 0.01)

sample_path = PROCESSED_DIR / "test_final.csv"
uploaded = st.sidebar.file_uploader("Sube un CSV para simular predicciones", type="csv")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif sample_path.exists():
    raw = pd.read_csv(sample_path)
else:
    raw = None

if raw is not None:
    X, y_true, scores, predictions = score_dataframe(model, raw, threshold)
else:
    X, y_true, scores, predictions = None, None, None, pd.DataFrame()

tab_overview, tab_predictions, tab_evaluation, tab_explainability = st.tabs([
    "Resumen ejecutivo",
    "Simulador de predicciones",
    "Evaluación y business value",
    "Explicabilidad",
])

with tab_overview:
    st.subheader("KPIs del modelo final")

    if not metrics.empty:
        row = metrics.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Threshold", f"{float(row.get('threshold', threshold)):.2f}")
        c2.metric("Recall", f"{float(row.get('recall', np.nan)):.3f}")
        c3.metric("Precision", f"{float(row.get('precision', np.nan)):.3f}")
        c4.metric("F2", f"{float(row.get('f2', np.nan)):.3f}")
        c5.metric("PR-AUC", f"{float(row.get('average_precision', np.nan)):.3f}")
    else:
        st.info("No se encontró reports/final_validation_metrics.csv.")

    if not final_decision.empty:
        st.subheader("Decisión final")
        st.dataframe(final_decision, use_container_width=True)

    if not threshold_comparison.empty:
        st.subheader("Comparación de thresholds")
        st.dataframe(threshold_comparison, use_container_width=True)

    if not threshold_bv.empty and "business_value" in threshold_bv.columns:
        st.plotly_chart(plot_business_value(threshold_bv, threshold), use_container_width=True)

with tab_predictions:
    st.subheader("Simulador de predicciones")
    st.write("Carga un CSV o usa el test set generado por los notebooks. El threshold se controla desde la barra lateral.")

    if predictions.empty:
        st.info("No hay data disponible para simular. Sube un CSV o genera data/processed/test_final.csv.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Autos evaluados", f"{len(predictions):,}")
        c2.metric("Marcados como Bad Buy", f"{int(predictions['pred_IsBadBuy'].sum()):,}")
        c3.metric("Positive rate", f"{predictions['pred_IsBadBuy'].mean():.1%}")

        show_cols = [
            c for c in [
                "risk_score", "pred_IsBadBuy", "decision", "VehBCost", "WarrantyCost",
                "MMRAcquisitionAuctionAveragePrice", "MMRCurrentRetailAveragePrice",
                "Make", "Model", "VehYear", "VehicleAge", TARGET
            ] if c in predictions.columns
        ]
        st.dataframe(
            predictions.sort_values("risk_score", ascending=False)[show_cols].head(100),
            use_container_width=True,
        )

        csv = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar predicciones",
            data=csv,
            file_name="predicciones_bad_buy.csv",
            mime="text/csv",
        )

        fig = px.histogram(predictions, x="risk_score", nbins=30, title="Distribución de scores de riesgo")
        fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"threshold={threshold:.2f}")
        st.plotly_chart(fig, use_container_width=True)

with tab_evaluation:
    st.subheader("Evaluación sobre data con target")

    if y_true is None:
        st.info("El CSV cargado no tiene la columna target. Solo se muestran predicciones, no métricas.")
    else:
        valid_mask = y_true.notna()
        y_eval = y_true[valid_mask].astype(int).to_numpy()
        scores_eval = scores[valid_mask]

        m = metric_row(y_eval, scores_eval, threshold)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recall", f"{m['recall']:.3f}")
        c2.metric("Precision", f"{m['precision']:.3f}")
        c3.metric("F2", f"{m['f2']:.3f}")
        c4.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        c5.metric("Positive rate", f"{m['positive_rate']:.1%}")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_confusion_matrix(y_eval, scores_eval, threshold), use_container_width=True)
        with col2:
            st.plotly_chart(plot_roc(y_eval, scores_eval), use_container_width=True)

        bv_current = evaluate_thresholds(y_eval, scores_eval)
        st.subheader("Business Value recalculado sobre la data cargada")
        st.plotly_chart(plot_business_value(bv_current, threshold), use_container_width=True)
        st.dataframe(
            bv_current.sort_values("business_value", ascending=False).head(10),
            use_container_width=True,
        )

    if not confusion_impact.empty:
        st.subheader("Impacto económico por matriz de confusión")
        st.dataframe(confusion_impact, use_container_width=True)
        if {"case", "business_value"}.issubset(confusion_impact.columns):
            fig = px.bar(confusion_impact, x="case", y="business_value", title="Aporte económico por tipo de predicción")
            st.plotly_chart(fig, use_container_width=True)

with tab_explainability:
    st.subheader("Explicabilidad por instancia")

    if predictions.empty:
        st.info("No hay data disponible para explicar.")
    else:
        sorted_preds = predictions.sort_values("risk_score", ascending=False).reset_index(drop=True)
        max_idx = min(len(sorted_preds) - 1, 200)
        selected_rank = st.slider("Selecciona una instancia ordenada por mayor riesgo", 0, max_idx, 0)
        selected_original_idx = sorted_preds.index[selected_rank]

        st.write("Predicción seleccionada")
        st.dataframe(sorted_preds.iloc[[selected_rank]], use_container_width=True)

        try:
            contrib = model_feature_contributions(model, X, selected_original_idx)
            top = contrib.head(15).copy()
            fig = px.bar(
                top.sort_values("contribution"),
                x="contribution",
                y="feature",
                orientation="h",
                title="Top contribuciones al score de riesgo",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top[["feature", "value_transformed", "coefficient", "contribution"]], use_container_width=True)
        except Exception as exc:
            st.warning(f"No se pudo calcular contribución por coeficientes: {exc}")

        with st.expander("SHAP opcional"):
            st.write("Para LogisticRegression, la explicación por coeficientes es ligera y estable. SHAP queda como opción si la librería está instalada.")
            if st.button("Intentar calcular SHAP para esta instancia"):
                try:
                    import shap
                    preprocess = model.named_steps["preprocess"]
                    clf = model.named_steps["clf"]
                    X_transformed = preprocess.transform(X)
                    feature_names = preprocess.get_feature_names_out()
                    sample_size = min(500, X_transformed.shape[0])
                    background = X_transformed[:sample_size]
                    explainer = shap.LinearExplainer(clf, background)
                    shap_values = explainer(X_transformed[selected_original_idx:selected_original_idx + 1])
                    shap_df = pd.DataFrame({
                        "feature": feature_names,
                        "shap_value": np.asarray(shap_values.values).ravel(),
                    })
                    shap_df["abs_shap"] = shap_df["shap_value"].abs()
                    st.dataframe(shap_df.sort_values("abs_shap", ascending=False).head(15), use_container_width=True)
                except Exception as exc:
                    st.error(f"No se pudo calcular SHAP: {exc}")
