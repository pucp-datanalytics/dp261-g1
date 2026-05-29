# DP261-G1 | Predicción de Bad Buys en subastas automotrices

Proyecto final end-to-end para clasificar vehículos usados con alto riesgo de ser una mala compra (`IsBadBuy=1`). La entrega incluye notebooks detallados, módulos reutilizables, pipeline completo, modelo final, API FastAPI, dashboard Streamlit, contratos y reportes.

## Decisiones principales de la versión final

- Target: `IsBadBuy` (`1 = Bad Buy`, `0 = Good Buy`).
- Dataset original: `data/raw/06-kickAutomotriz.csv`.
- El dataset está desbalanceado; por eso se usan `class_weight`, `scale_pos_weight` y balanceo controlado solo dentro de train/folds.
- Fecha `PurchDate`: se transforma dentro del pipeline, no manualmente antes de entrenar.
- Modelos Sprint 3: Logistic Regression, Decision Tree, Random Forest, SVM y KNN.
- Modelos Sprint 4: XGBoost, LightGBM, Bagging, Voting y Stacking.
- Optimización: RandomizedSearchCV + Optuna/TPE bayesiano.
- Modelo final: se selecciona automáticamente en `scripts/run_all.py` por mayor valor de negocio; en la última corrida incluida quedó registrado en `models/model_metadata.json`.
- Threshold final: `0.5`, según la especificación de entrega.
- Función de negocio:

```text
Valor = TP*2500 + TN*600 + FP*(-900) + FN*(-4500)
```

La penalización de FN es alta porque representa comprar un vehículo defectuoso creyendo que era bueno. Esta corrección responde directamente a la observación del profesor.

## Cómo ejecutar todo

Desde la raíz del repositorio:

```bash
conda env create -f environment.yml
conda activate dp261-g1
PYTHONPATH=. python scripts/run_all.py
```

En Windows PowerShell:

```powershell
$env:PYTHONPATH="."
python scripts/run_all.py
```

El script genera o actualiza:

- `data/processed/train_full.csv`
- `data/processed/test_final.csv`
- `data/processed/train_sample.csv`
- `reports/model_results_summary.csv`
- `reports/business_model_ranking.csv`
- `reports/tuning_results.csv`
- `reports/final_validation_metrics.csv`
- `models/final_model.pkl`
- `models/model_metadata.json`
- `handoff/model/final_model.pkl`
- `handoff/contracts/*.json`

Si `mlflow` está instalado, también registra el experimento en `mlruns/`.

## Estructura

```text
api/                  FastAPI + Docker + smoke test
dashboard/            Streamlit comercial con semáforo
data/raw/             dataset original
data/processed/       splits generados train/test/sample
handoff/contracts/    contratos de entrada/salida para Sprint 6
handoff/model/        copia del modelo final y metadata
models/               final_model.pkl + model_metadata.json
notebooks/            flujo CRISP-DM documentado paso a paso
reports/              resultados, ranking, reporte ejecutivo y figuras
scripts/run_all.py    ejecución end-to-end reproducible
src/                  preprocessing, modelos, tuning y evaluación
tests/                pruebas del contrato API
```

## Notebooks principales

Los notebooks están escritos para explicar el proyecto a alguien que no conoce el código:

1. `01_business_understanding.ipynb`: problema, target, stakeholders, criterios de éxito y costo de errores.
2. `02_data_loading.ipynb`: carga, esquema, tipos y primeras validaciones.
3. `03_eda.ipynb`: EDA completo, balance, nulos, distribuciones, relaciones con target y hallazgos.
4. `04_data_preparation_pipeline.ipynb`: limpieza, feature engineering, justificación de variables y balanceo.
5. `05_baseline_models.ipynb`: modelos baseline pedidos en Sprint 3 y por qué se usan.
6. `06_tuning_optuna.ipynb`: RandomizedSearchCV y Optuna/TPE.
7. `07_advanced_ensembles.ipynb`: XGBoost, LightGBM, Bagging, Voting y Stacking.
8. `08_business_value_final_selection.ipynb`: función económica y selección del modelo.
9. `09_final_validation_mlflow.ipynb`: validación final, MLflow y artefactos.
10. `10_api_dashboard_demo.ipynb`: API, dashboard y contratos para deployment.

## Levantar API

```bash
uvicorn api.main:app --reload
```

Endpoints:

- `GET /health`
- `GET /version`
- `POST /predict`
- `POST /predict_batch`

Probar:

```bash
python api/smoke_test.py
```

## Levantar dashboard

En otra terminal, con la API activa:

```bash
streamlit run dashboard/app.py
```

El dashboard está orientado a usuario comercial: muestra semáforo, decisión, score de riesgo, valor económico del lote y variables explicativas.

## Docker

```bash
docker build -f api/Dockerfile -t dp261-badbuy-api .
docker run --rm -p 8000:8000 dp261-badbuy-api
```

## Archivos clave para sustentar

- `reports/final_report.md`: resumen ejecutivo y resultados finales.
- `reports/technical_change_log.md`: qué se corrigió frente a la versión anterior.
- `reports/model_selection_rationale.md`: por qué se selecciona el modelo ganador.
- `reports/business_model_ranking.csv`: selección del modelo por valor económico.
- `reports/tuning_results.csv`: evidencia de RandomizedSearchCV y Optuna/TPE.
- `reports/final_validation_metrics.csv`: validación final en test.
- `src/preprocessing.py`: limpieza + feature engineering reutilizable.
- `src/models.py`: modelos de Sprint 3 y Sprint 4.
- `src/tuning.py`: optimización de hiperparámetros.
- `api/main.py`: servicio FastAPI listo para Docker/AWS.
- `dashboard/app.py`: interfaz comercial.
