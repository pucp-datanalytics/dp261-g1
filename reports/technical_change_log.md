# Bitácora de correcciones realizadas

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
