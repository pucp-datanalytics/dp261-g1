# Justificación de selección del modelo final

## Modelo ganador

**Bagging_DecisionTree**

## Criterio de selección

El criterio principal no fue accuracy. Se seleccionó el modelo con mayor **valor de negocio** usando:

`TP*2500 + TN*600 + FP*(-900) + FN*(-4500)`

Esto corrige la observación del profesor: antes el FN tenía peso 0, pero en este problema el FN representa comprar un vehículo defectuoso pensando que era bueno.

## Técnicas comparadas

- Baseline: Logistic Regression, Decision Tree, Random Forest, SVM, KNN.
- Optimización: RandomizedSearchCV y Optuna/TPE.
- Avanzados/ensambles: XGBoost, LightGBM, Bagging, Voting, Stacking.

## Por qué no se usó ExtraTrees

Aunque puede funcionar bien en datos tabulares, se retiró porque no aparece explícitamente en los sprints como modelo solicitado. Para la entrega final se prioriza cumplir el backlog del curso y facilitar la sustentación.
