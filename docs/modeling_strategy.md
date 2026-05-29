# Estrategia de modelado

## Modelos incluidos

La versión final se limita a los modelos pedidos en los sprints.

### Sprint 3: Baselines

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

### Sprint 4: Avanzados y ensambles

- XGBoost
- LightGBM
- Bagging
- Voting
- Stacking

### Optimización

- RandomizedSearchCV
- Optuna/TPE bayesiano

## Modelos retirados

No se usa `DummyClassifier` como candidato porque no aporta una solución predictiva real para la entrega final. Tampoco se usa `ExtraTrees`, aunque técnicamente puede funcionar bien, porque no aparece explícitamente en el backlog de modelos solicitado y complica la sustentación.

## Balanceo

El dataset tiene clase positiva minoritaria. La estrategia final evita data leakage:

- Modelos lineales/distancia: `RandomUnderSampler` dentro del pipeline + escalado.
- Árboles: `class_weight` o `balanced_subsample`.
- XGBoost/LightGBM: `scale_pos_weight`.

No se balancea el test.

## Selección final

El modelo final se elige por valor de negocio:

`Valor = TP*2500 + TN*600 + FP*(-900) + FN*(-4500)`

El FN se penaliza fuerte porque representa comprar un vehículo defectuoso no detectado.
