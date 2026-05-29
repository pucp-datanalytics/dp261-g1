# Reporte técnico ejecutivo - DP261 Bad Buy Automotriz

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

`Valor = TP*2500 + TN*600 + FP*(-900) + FN*(-4500)`

Interpretación:

- **TP**: el modelo detecta un Bad Buy. Se evita comprar un vehículo defectuoso.
- **TN**: el modelo aprueba un buen vehículo. Se captura margen operativo.
- **FP**: el modelo rechaza un buen vehículo. Se pierde una oportunidad comercial.
- **FN**: el modelo no detecta un Bad Buy. La empresa compra un vehículo defectuoso; por eso ahora sí tiene una penalización alta.

## 4. Ranking de modelos por valor de negocio

| model                  |   business_value |   recall |   precision |       f2 |   roc_auc |   tp |   fp |   fn |   tn |
|:-----------------------|-----------------:|---------:|------------:|---------:|----------:|-----:|-----:|-----:|-----:|
| Bagging_DecisionTree   |           151900 | 0.468468 |    0.295455 | 0.419355 |  0.714646 |   52 |  124 |   59 |  665 |
| XGBoost_random_search  |           148900 | 0.495495 |    0.282051 | 0.43036  |  0.718962 |   55 |  140 |   56 |  649 |
| LightGBM_baseline      |           148400 | 0.423423 |    0.313333 | 0.395623 |  0.717958 |   47 |  103 |   64 |  686 |
| Voting_soft_LR_RF_LGBM |           142900 | 0.468468 |    0.285714 | 0.415335 |  0.728017 |   52 |  130 |   59 |  659 |
| LightGBM_Optuna_TPE    |           141400 | 0.522523 |    0.267281 | 0.438729 |  0.740223 |   58 |  159 |   53 |  630 |

## 5. Modelo seleccionado

Modelo final: **Bagging_DecisionTree**.

Se selecciona porque maximiza el valor económico estimado en validación muestral, manteniendo el foco en recall/F2 de la clase 1. Cuando dos modelos son parecidos, se prefiere el que ofrece mejor balance entre menor FN, precision razonable y complejidad operativa sustentable.

## 6. Validación final en test holdout

El test set se usa una sola vez, después de elegir el modelo.

- Recall clase 1: **0.609**
- Precision clase 1: **0.270**
- F2: **0.487**
- ROC-AUC: **0.767**
- Average Precision: **0.477**
- Valor de negocio: **USD 2,835,200**
- Valor por caso: **USD 194.23**
- Matriz: TN=9851, FP=2951, FN=701, TP=1094

## 7. Decisión comercial en dashboard

- **Rojo**: score >= 0.50. Detener compra o exigir revisión experta.
- **Ámbar**: 0.30 <= score < 0.50. Revisar manualmente.
- **Verde**: score < 0.30. Continuar evaluación comercial.

## 8. Riesgos y monitoreo

- Monitorear drift de `VehOdo`, `VehicleAge`, `VehBCost`, `WarrantyCost`, precios MMR, marca/modelo y subasta.
- Guardar predicciones y resultados reales para recalibrar el modelo.
- Revalidar la matriz económica si cambia el margen por unidad o el costo de reparación/garantía.
