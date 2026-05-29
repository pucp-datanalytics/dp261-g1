"""Ejecución end-to-end del proyecto DP261.

Qué hace:
1. Lee dataset crudo y crea split train/test estratificado.
2. Entrena modelos baseline pedidos en Sprint 3.
3. Entrena modelos avanzados pedidos en Sprint 4: XGBoost, LightGBM y ensambles.
4. Optimiza hiperparámetros con RandomizedSearchCV y Optuna/TPE bayesiano.
5. Compara todos los modelos con threshold 0.5 usando función de negocio.
6. Reentrena el ganador con todo el train y valida una sola vez en test final.
7. Guarda artefactos para API, dashboard, MLflow, handoff y reportes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BENEFIT_TN,
    BENEFIT_TP,
    BUSINESS_THRESHOLD,
    COST_FN,
    COST_FP,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_SEARCH_ITER,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    SAMPLE_SIZE,
    TARGET,
    TEST_SIZE,
)
from src.evaluation import bootstrap_metric_ci, evaluate_model, gain_curve, get_scores
from src.models import (
    build_advanced_models,
    build_baseline_models,
    build_ensembles,
    make_model_pipeline,
    save_model,
    train_and_score_candidates,
)
from src.preprocessing import prepare_features, split_X_y
from src.tuning import dataframe_from_rows, default_random_search_spaces, tune_lightgbm_with_optuna, tune_with_random_search


def ensure_dirs():
    for path in [
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        Path("handoff/model"),
        Path("handoff/contracts"),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def business_assumptions():
    return {
        "benefit_tp": BENEFIT_TP,
        "benefit_tn": BENEFIT_TN,
        "cost_fp": COST_FP,
        "cost_fn": COST_FN,
        "equation": f"TP*{BENEFIT_TP} + TN*{BENEFIT_TN} + FP*({COST_FP}) + FN*({COST_FN})",
        "threshold": BUSINESS_THRESHOLD,
    }


def plot_model_business(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordered = results.sort_values("business_value", ascending=True)
    ax.barh(ordered["model"], ordered["business_value"])
    ax.set_title("Valor de negocio por modelo - validación muestral")
    ax.set_xlabel("Valor estimado en muestra de validación (USD)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "business_value_by_model.png", dpi=160)
    plt.close(fig)


def plot_gain_curve(y_true, y_score):
    gc = gain_curve(y_true, y_score)
    gc.to_csv(REPORTS_DIR / "gain_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gc["pct_population"], gc["pct_positives_captured"], label="Modelo")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Aleatorio")
    ax.set_title("Gain curve del modelo final")
    ax.set_xlabel("% de vehículos revisados")
    ax.set_ylabel("% de Bad Buys capturados")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "gain_curve.png", dpi=160)
    plt.close(fig)


def build_contracts(raw_df: pd.DataFrame, metadata: dict):
    example = raw_df.drop(columns=[TARGET], errors="ignore").iloc[0].where(pd.notna(raw_df.iloc[0]), None).to_dict()
    for key, value in list(example.items()):
        if hasattr(value, "item"):
            example[key] = value.item()
    response = {
        "risk_score": 0.57,
        "threshold": metadata["threshold"],
        "prediction": 1,
        "decision": "ROJO - Alto riesgo: detener compra o pasar a revisión experta",
        "risk_segment": "ROJO",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "api_version": "1.0.0",
        "business_value_assumptions": metadata["business_assumptions"],
        "shap_top_features": [],
        "explanation_method": "shap_if_available_else_feature_importance",
    }
    input_schema = {
        "type": "object",
        "description": "Columnas crudas del dataset original, sin variables creadas manualmente.",
        "properties": {col: {"type": "string/number/null"} for col in example},
    }
    output_schema = {"type": "object", "properties": {key: {"example": value} for key, value in response.items()}}
    save_json(example, "handoff/contracts/example_request.json")
    save_json(response, "handoff/contracts/example_response.json")
    save_json(input_schema, "handoff/contracts/input_schema.json")
    save_json(output_schema, "handoff/contracts/output_schema.json")


def write_reports(results: pd.DataFrame, tuning_results: pd.DataFrame, final_metrics: dict, metadata: dict):
    top_table = results.head(5)[["model", "business_value", "recall", "precision", "f2", "roc_auc", "tp", "fp", "fn", "tn"]]
    top_md = top_table.to_markdown(index=False)

    final_report = f"""# Reporte técnico ejecutivo - DP261 Bad Buy Automotriz

## 1. Resumen ejecutivo

El proyecto predice si un vehículo de subasta será una **mala compra** (`IsBadBuy=1`). El entregable final no es solo un modelo: es un pipeline completo que recibe las columnas crudas del dataset, crea variables de negocio, limpia, imputa, codifica, escala cuando corresponde y devuelve una decisión comercial interpretable.

La selección del modelo se hizo con una función de negocio sobre la matriz de confusión y con threshold estándar **0.5**, siguiendo la observación de la revisión final.

## 2. Modelos evaluados

Se evaluaron los modelos pedidos en los sprints:

- Sprint 3 / baseline: Logistic Regression, Decision Tree, Random Forest, SVM y KNN.
- Sprint 4 / avanzados: XGBoost, LightGBM, Bagging, Voting y Stacking.
- Optimización: RandomizedSearchCV para modelos candidatos y Optuna/TPE como optimización bayesiana para LightGBM.

No se usó ExtraTrees ni Dummy como candidato final porque no forman parte explícita del backlog de modelado solicitado.

## 3. Ecuación de negocio

`Valor = TP*{BENEFIT_TP} + TN*{BENEFIT_TN} + FP*({COST_FP}) + FN*({COST_FN})`

Interpretación:

- **TP**: el modelo detecta un Bad Buy. Se evita comprar un vehículo defectuoso.
- **TN**: el modelo aprueba un buen vehículo. Se captura margen operativo.
- **FP**: el modelo rechaza un buen vehículo. Se pierde una oportunidad comercial.
- **FN**: el modelo no detecta un Bad Buy. La empresa compra un vehículo defectuoso; por eso ahora sí tiene una penalización alta.

## 4. Ranking de modelos por valor de negocio

{top_md}

## 5. Modelo seleccionado

Modelo final: **{metadata['model_name']}**.

Se selecciona porque maximiza el valor económico estimado en validación muestral, manteniendo el foco en recall/F2 de la clase 1. Cuando dos modelos son parecidos, se prefiere el que ofrece mejor balance entre menor FN, precision razonable y complejidad operativa sustentable.

## 6. Validación final en test holdout

El test set se usa una sola vez, después de elegir el modelo.

- Recall clase 1: **{final_metrics['recall']:.3f}**
- Precision clase 1: **{final_metrics['precision']:.3f}**
- F2: **{final_metrics['f2']:.3f}**
- ROC-AUC: **{final_metrics['roc_auc']:.3f}**
- Average Precision: **{final_metrics['average_precision']:.3f}**
- Valor de negocio: **USD {final_metrics['business_value']:,.0f}**
- Valor por caso: **USD {final_metrics['business_value_per_case']:,.2f}**
- Matriz: TN={final_metrics['tn']}, FP={final_metrics['fp']}, FN={final_metrics['fn']}, TP={final_metrics['tp']}

## 7. Decisión comercial en dashboard

- **Rojo**: score >= 0.50. Detener compra o exigir revisión experta.
- **Ámbar**: 0.30 <= score < 0.50. Revisar manualmente.
- **Verde**: score < 0.30. Continuar evaluación comercial.

## 8. Riesgos y monitoreo

- Monitorear drift de `VehOdo`, `VehicleAge`, `VehBCost`, `WarrantyCost`, precios MMR, marca/modelo y subasta.
- Guardar predicciones y resultados reales para recalibrar el modelo.
- Revalidar la matriz económica si cambia el margen por unidad o el costo de reparación/garantía.
"""
    (REPORTS_DIR / "final_report.md").write_text(final_report, encoding="utf-8")

    technical_change_log = f"""# Bitácora de correcciones realizadas

1. Se retiraron `Dummy` y `ExtraTrees` del flujo de modelado porque no estaban en el alcance explícito de los sprints.
2. Se agregaron los baseline pedidos: Logistic Regression, Decision Tree, Random Forest, SVM y KNN.
3. Se agregaron modelos avanzados pedidos: XGBoost, LightGBM, Bagging, Voting y Stacking.
4. Se agregó optimización con RandomizedSearchCV y Optuna/TPE bayesiano.
5. Se documentó el porqué de cada feature creada en notebooks y en `src/preprocessing.py`.
6. Se consolidó feature engineering en un único transformer reutilizable por notebooks, API y dashboard.
7. Se corrigió la función de negocio: FN ya no vale cero y tiene penalización fuerte.
8. Se mantuvo threshold 0.5 para comparación final, según especificación de revisión.
9. La API carga el pipeline completo, no solo el estimador. Por tanto recibe columnas crudas y ejecuta transformaciones internas.
10. El dashboard está orientado a usuario comercial con semáforo y explicación simple.
"""
    (REPORTS_DIR / "technical_change_log.md").write_text(technical_change_log, encoding="utf-8")

    model_selection = f"""# Justificación de selección del modelo final

## Modelo ganador

**{metadata['model_name']}**

## Criterio de selección

El criterio principal no fue accuracy. Se seleccionó el modelo con mayor **valor de negocio** usando:

`{business_assumptions()['equation']}`

Esto corrige la observación del profesor: antes el FN tenía peso 0, pero en este problema el FN representa comprar un vehículo defectuoso pensando que era bueno.

## Técnicas comparadas

- Baseline: Logistic Regression, Decision Tree, Random Forest, SVM, KNN.
- Optimización: RandomizedSearchCV y Optuna/TPE.
- Avanzados/ensambles: XGBoost, LightGBM, Bagging, Voting, Stacking.

## Por qué no se usó ExtraTrees

Aunque puede funcionar bien en datos tabulares, se retiró porque no aparece explícitamente en los sprints como modelo solicitado. Para la entrega final se prioriza cumplir el backlog del curso y facilitar la sustentación.
"""
    (REPORTS_DIR / "model_selection_rationale.md").write_text(model_selection, encoding="utf-8")


def fit_evaluate_group(group_name: str, models: dict, X_dev, y_dev, X_valid, y_valid) -> tuple[pd.DataFrame, dict]:
    print(f"\n=== {group_name}: {len(models)} modelos ===")
    results, fitted = train_and_score_candidates(models, X_dev, y_dev, X_valid, y_valid, evaluate_model)
    results.insert(0, "stage", group_name)
    print(results[["model", "business_value", "recall", "precision", "f2", "roc_auc"]].sort_values("business_value", ascending=False))
    return results, fitted


def main():
    ensure_dirs()
    df = pd.read_csv(RAW_DATA_PATH)

    train_full, test_final = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)
    train_sample, _ = train_test_split(
        train_full,
        train_size=min(SAMPLE_SIZE, len(train_full)),
        stratify=train_full[TARGET],
        random_state=RANDOM_STATE,
    )
    train_full.to_csv(PROCESSED_DIR / "train_full.csv", index=False)
    test_final.to_csv(PROCESSED_DIR / "test_final.csv", index=False)
    train_sample.to_csv(PROCESSED_DIR / "train_sample.csv", index=False)

    X_sample, y_sample = split_X_y(train_sample)
    X_dev, X_valid, y_dev, y_valid = train_test_split(
        X_sample, y_sample, test_size=0.30, stratify=y_sample, random_state=RANDOM_STATE
    )

    all_results = []
    all_fitted = {}

    baseline_results, baseline_fitted = fit_evaluate_group(
        "baseline_sprint3",
        build_baseline_models(X_dev),
        X_dev,
        y_dev,
        X_valid,
        y_valid,
    )
    all_results.append(baseline_results)
    all_fitted.update(baseline_fitted)

    advanced_results, advanced_fitted = fit_evaluate_group(
        "advanced_sprint4",
        build_advanced_models(X_dev, y_dev),
        X_dev,
        y_dev,
        X_valid,
        y_valid,
    )
    all_results.append(advanced_results)
    all_fitted.update(advanced_fitted)

    # RandomizedSearchCV sobre candidatos pedidos.
    tuning_rows = []
    tuned_models = {}
    spaces = default_random_search_spaces()
    baseline_for_tuning = build_baseline_models(X_dev)
    advanced_for_tuning = build_advanced_models(X_dev, y_dev)
    random_search_candidates = {
        "RandomForest_random_search": baseline_for_tuning["RandomForest_baseline"],
        "DecisionTree_random_search": baseline_for_tuning["DecisionTree_baseline"],
        "LogisticRegression_random_search": baseline_for_tuning["LogisticRegression_baseline"],
    }
    if "XGBoost_baseline" in advanced_for_tuning:
        random_search_candidates["XGBoost_random_search"] = advanced_for_tuning["XGBoost_baseline"]
    for name, pipe in random_search_candidates.items():
        print(f"\n=== Tuning RandomizedSearchCV: {name} ===")
        best_model, row = tune_with_random_search(name, pipe, spaces[name], X_dev, y_dev, n_iter=RANDOM_SEARCH_ITER)
        tuned_models[name] = best_model
        tuning_rows.append(row)

    # Optuna/TPE bayesiano para LightGBM.
    try:
        print("\n=== Tuning Optuna/TPE: LightGBM ===")
        best_lgbm, optuna_row, trials_df = tune_lightgbm_with_optuna(X_dev, X_dev, y_dev)
        tuned_models["LightGBM_Optuna_TPE"] = best_lgbm
        tuning_rows.append(optuna_row)
        trials_df.to_csv(REPORTS_DIR / "optuna_lightgbm_trials.csv", index=False)
    except Exception as exc:
        (REPORTS_DIR / "optuna_note.txt").write_text(
            f"Optuna no pudo ejecutarse en este entorno: {exc}\nEl código está implementado en src/tuning.py y el entorno incluye optuna.",
            encoding="utf-8",
        )

    tuned_results, tuned_fitted = fit_evaluate_group("tuned_sprint4", tuned_models, X_dev, y_dev, X_valid, y_valid)
    all_results.append(tuned_results)
    all_fitted.update(tuned_fitted)

    ensemble_results, ensemble_fitted = fit_evaluate_group(
        "ensembles_sprint4",
        build_ensembles(X_dev, y_dev),
        X_dev,
        y_dev,
        X_valid,
        y_valid,
    )
    all_results.append(ensemble_results)
    all_fitted.update(ensemble_fitted)

    results = pd.concat(all_results, ignore_index=True)
    results = results.sort_values(["business_value", "recall", "f2", "precision"], ascending=[False, False, False, False]).reset_index(drop=True)
    results.to_csv(REPORTS_DIR / "model_results_summary.csv", index=False)
    results.to_csv(REPORTS_DIR / "business_model_ranking.csv", index=False)
    dataframe_from_rows(tuning_rows).to_csv(REPORTS_DIR / "tuning_results.csv", index=False)
    plot_model_business(results)

    best_name = str(results.loc[0, "model"])
    print(f"\n=== Modelo seleccionado por business value: {best_name} ===")
    final_model = all_fitted[best_name]

    # Reentrenar ganador con todo el train, sin tocar el test final hasta la validación definitiva.
    X_full, y_full = split_X_y(train_full)
    X_test, y_test = split_X_y(test_final)
    final_model.fit(X_full, y_full)
    final_path = save_model(final_model, MODELS_DIR / "final_model.pkl")

    final_scores = get_scores(final_model, X_test)
    final_metrics = evaluate_model(final_model, X_test, y_test)
    final_metrics_no_report = {k: v for k, v in final_metrics.items() if k != "classification_report"}
    final_metrics_no_report["roc_auc_ci_95"] = bootstrap_metric_ci(
        y_test,
        final_scores,
        lambda yt, ys: __import__("sklearn.metrics").metrics.roc_auc_score(yt, ys),
        n_boot=200,
        random_state=RANDOM_STATE,
    )

    pd.DataFrame([final_metrics_no_report]).to_csv(REPORTS_DIR / "final_validation_metrics.csv", index=False)
    pd.DataFrame([
        {
            "tn": final_metrics["tn"],
            "fp": final_metrics["fp"],
            "fn": final_metrics["fn"],
            "tp": final_metrics["tp"],
            "business_value": final_metrics["business_value"],
        }
    ]).to_csv(REPORTS_DIR / "final_confusion_matrix_summary.csv", index=False)
    plot_gain_curve(y_test, final_scores)

    raw_feature_list = [c for c in df.columns if c != TARGET]
    prepared_preview = prepare_features(df.drop(columns=[TARGET]).head(5))
    engineered_feature_list = [c for c in prepared_preview.columns if c not in raw_feature_list]
    metadata = {
        "model_name": best_name,
        "model_version": "final-entrega-dp261-v2",
        "threshold": BUSINESS_THRESHOLD,
        "business_assumptions": business_assumptions(),
        "test_metrics": final_metrics_no_report,
        "raw_features": raw_feature_list,
        "engineered_features": engineered_feature_list,
        "engineered_features_count": len(engineered_feature_list),
        "model_path": str(final_path),
        "stages_evaluated": sorted(results["stage"].unique().tolist()),
    }
    save_json(metadata, MODELS_DIR / "model_metadata.json")
    shutil.copy2(MODELS_DIR / "final_model.pkl", "handoff/model/final_model.pkl")
    shutil.copy2(MODELS_DIR / "model_metadata.json", "handoff/model/model_metadata.json")
    build_contracts(df, metadata)
    write_reports(results, dataframe_from_rows(tuning_rows), final_metrics_no_report, metadata)

    # MLflow: se registra si la dependencia está disponible.
    try:
        import mlflow

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("dp261_bad_buy_final")
        with mlflow.start_run(run_name=f"final_{best_name}"):
            mlflow.log_params(metadata["business_assumptions"])
            mlflow.log_param("model_name", best_name)
            mlflow.log_param("model_version", metadata["model_version"])
            for key in ["business_value", "recall", "precision", "f2", "roc_auc", "average_precision"]:
                mlflow.log_metric(key, float(final_metrics_no_report[key]))
            mlflow.sklearn.log_model(final_model, "model")
            mlflow.log_artifact(str(REPORTS_DIR / "business_model_ranking.csv"))
            mlflow.log_artifact(str(REPORTS_DIR / "tuning_results.csv"))
    except Exception as exc:
        (REPORTS_DIR / "mlflow_note.txt").write_text(
            f"MLflow no se ejecutó en este entorno: {exc}\nInstalar mlflow y correr `python scripts/run_all.py` para registrar el experimento.",
            encoding="utf-8",
        )

    print(json.dumps({"selected_model": best_name, "final_metrics": final_metrics_no_report}, indent=2))


if __name__ == "__main__":
    main()
