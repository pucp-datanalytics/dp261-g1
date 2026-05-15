# Reporte ejecutivo — Predicción de Bad Buys

## 1. Resumen ejecutivo

El proyecto desarrolló un modelo para anticipar vehículos con alto riesgo de convertirse en `Bad Buy` en subastas automotrices. El objetivo de negocio es reducir compras riesgosas sin bloquear de forma excesiva el flujo de autos buenos que sostienen las ventas.

El modelo seleccionado es **LogisticRegression tuneado**. Esta alternativa fue elegida porque mantiene un recall competitivo frente a otros modelos, pero ofrece mejor balance operativo en F2, precision e interpretabilidad.

La evaluación económica muestra que el threshold estándar `0.50` no maximiza el valor esperado. Bajo los supuestos definidos, el threshold óptimo económico es **0.70**.

## 2. Decisión recomendada

| Elemento | Recomendación |
|---|---|
| Modelo final | LogisticRegression tuneado |
| Threshold técnico de referencia | 0.50 |
| Threshold operativo recomendado | 0.70 |
| Criterio final | Maximizar valor económico esperado |
| Modo de uso inicial | Soporte a decisión / piloto controlado |
| Decisión Sprint 6 | GO controlado hacia MVP |

## 3. Evidencia técnica

Con threshold `0.50`, el modelo detecta más Bad Buys, pero también rechaza demasiados autos buenos:

- Recall: **61.6%**
- Precision: **26.7%**
- Positive rate: **28.4%**
- TP: **1,105**
- FP: **3,040**
- FN: **690**
- TN: **9,762**

Con threshold `0.70`, el modelo es más conservador y rentable:

- Recall: **36.7%**
- Precision: **51.1%**
- Positive rate: **8.8%**
- TP: **659**
- FP: **631**
- FN: **1,136**
- TN: **12,171**

## 4. Impacto económico

El análisis costo–beneficio considera:

- TP: ahorro por detectar y evitar un Bad Buy.
- FP: margen perdido por rechazar un auto bueno.
- FN: sin valor incremental frente a operar sin modelo.
- TN: flujo normal de operación.

Comparación principal:

| Escenario | Threshold | Business value | Valor promedio por vehículo |
|---|---:|---:|---:|
| Referencia técnica | 0.50 | 26,500 | 1.82 |
| Óptimo económico | 0.70 | 1,079,600 | 73.96 |

El cambio de threshold genera una mejora estimada de **1,053,100** en valor económico respecto al threshold `0.50`.

## 5. Riesgos y limitaciones

1. **Dependencia de supuestos económicos.** El threshold recomendado depende de los valores asignados a TP, FP, FN y TN. Estos supuestos deben ser validados con negocio.
2. **Falsos negativos persistentes.** El threshold económico reduce falsos positivos, pero deja pasar más Bad Buys que el threshold 0.50. Por eso se recomienda uso como soporte a decisión y no automatización total.
3. **Drift de datos.** Cambios en precios, subastas, garantías o mix de vehículos pueden degradar el modelo.
4. **Generalización.** El modelo fue entrenado con datos históricos; debe monitorearse contra datos nuevos.
5. **Interpretabilidad.** LogisticRegression facilita explicación general, pero en producción se recomienda incluir explicabilidad por instancia en dashboard.

## 6. Recomendaciones accionables

- Desplegar el modelo en Sprint 6 como MVP controlado.
- Usar threshold `0.70` como umbral operativo inicial.
- Mantener threshold `0.50` como referencia de cobertura de riesgo para análisis comparativo, no como decisión final automática.
- Implementar monitoreo mensual de recall, precision, positive rate y business value.
- Revisar la matriz costo–beneficio con el sponsor antes de automatizar rechazos.
- Incorporar dashboard con KPIs, matriz de confusión, curva de ganancia y explicación de predicciones.
- Definir un proceso de revisión manual para casos de riesgo intermedio.

## 7. Handoff para Sprint 6

Artefactos que deben quedar disponibles:

- `models/final_model.pkl`
- `reports/final_decision_summary.csv`
- `reports/threshold_business_value.csv`
- `reports/threshold_business_value_comparison.csv`
- `reports/business_confusion_matrix_impact.csv`
- `reports/executive_report.md`
- `dashboard/app.py`
- contrato de entrada/salida para API o dashboard
- definición del threshold operativo y supuestos económicos

## 8. Conclusión

El modelo genera valor cuando el threshold se define como una decisión de negocio y no solo técnica. La recomendación final es avanzar con `LogisticRegression` tuneado y threshold `0.70` porque maximiza el valor económico esperado bajo los supuestos actuales, reduciendo de forma importante los falsos positivos y manteniendo una detección relevante de Bad Buys.