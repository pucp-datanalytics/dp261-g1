# Reporte ejecutivo - Prediccion de Bad Buys

## 1. Resumen ejecutivo

El proyecto desarrollo un modelo para anticipar vehiculos con alto riesgo de convertirse en `Bad Buy` en subastas automotrices. El objetivo de negocio es reducir compras riesgosas sin bloquear excesivamente el flujo de autos buenos.

El modelo recomendado para el MVP es **LogisticRegression tuneado**. La decision operativa final no se basa solo en F1 o AUC, sino en el valor economico esperado. Con los supuestos confirmados por negocio, el threshold recomendado es **0.70**.

**Recomendacion:** avanzar a Sprint 6 con un **GO controlado**, usando el modelo como soporte a decision y no como rechazo automatico hasta validar los supuestos economicos con el sponsor.

## 2. Decision recomendada

| Elemento | Recomendacion |
|---|---|
| Modelo final | LogisticRegression tuneado |
| Threshold tecnico de referencia | 0.50 |
| Threshold operativo recomendado | 0.70 |
| Criterio final | Maximizar business value esperado |
| Modo de uso inicial | Soporte a decision / piloto controlado |
| Decision Sprint 6 | GO controlado hacia MVP |

## 3. Evidencia tecnica

Con threshold `0.50`, el modelo detecta mas Bad Buys, pero tambien rechaza demasiados autos buenos:

- Recall: **61.6%**
- Precision: **26.7%**
- Positive rate: **28.4%**
- TP: **1,105** | FP: **3,040** | FN: **690** | TN: **9,762**

Con threshold `0.70`, el modelo es mas conservador y rentable:

- Recall: **36.7%**
- Precision: **51.1%**
- Positive rate: **8.8%**
- TP: **659** | FP: **631** | FN: **1,136** | TN: **12,171**

## 4. Impacto economico y funcion de optimizacion

### 4.1 Definicion de supuestos de impacto economico

Para evaluar el impacto economico del modelo no basta con revisar metricas tecnicas como `recall`, `precision` o `F2`. Es necesario traducir la matriz de confusion a terminos de negocio.

En este problema, el modelo predice si un auto sera un `Bad Buy` (`IsBadBuy = 1`). Por lo tanto, cada tipo de error tiene una consecuencia economica distinta:

| Caso | Interpretacion | Impacto de negocio |
|---|---|---|
| **TP** | El modelo detecta correctamente un Bad Buy | Se evita comprar un auto riesgoso |
| **FP** | El modelo marca como Bad Buy un auto que era bueno | Se pierde una oportunidad comercial |
| **FN** | El modelo deja pasar un Bad Buy | Se compra un auto riesgoso |
| **TN** | El modelo acepta correctamente un auto bueno | Se mantiene una oportunidad comercial normal |

El objetivo principal del modelo es reducir el riesgo de comprar autos malos.

A partir del analisis de variables economicas del dataset, se observa que:

- Los autos buenos tienen un margen neto esperado positivo.
- Los `Bad Buys` tienden a mostrar menor margen esperado.
- Los `Bad Buys` tienen mayores costos asociados, especialmente por garantia y sobrepago relativo.
- Rechazar un auto bueno tiene un costo, pero dejar pasar un `Bad Buy` representa una perdida mayor para el negocio.

Una estimacion razonable del costo de dejar pasar un `Bad Buy` considera:

```text
WarrantyCost mediano de Bad Buy
+ margen neto esperado perdido de un auto bueno
+ sobrepago adicional estimado
~ USD 2,500
```

Por eso, el beneficio de detectar correctamente un `Bad Buy` se modela como `TP = USD 2,500`. El costo de marcar erroneamente un auto bueno como riesgoso se modela como `FP = USD -900`. Para evitar doble conteo, el costo de un `FN` queda en `USD 0`, porque el impacto economico de ese riesgo ya se captura como beneficio evitado cuando el modelo acierta un `TP`.

Supuestos finales confirmados para Sprint 5:

| Tipo | Valor unitario | Interpretacion |
|---|---:|---|
| TP | USD 2,500 | Beneficio por detectar y evitar un Bad Buy |
| FP | USD -900 | Costo por bloquear un auto bueno |
| FN | USD 0 | Sin valor incremental frente a operar sin modelo |
| TN | USD 0 | Flujo normal de operacion |

### 4.2 Funcion de optimizacion

El threshold final se define como una decision de negocio. Para cada umbral `t`, el modelo convierte la probabilidad estimada en una prediccion:

```text
y_pred_i(t) = 1 si P(IsBadBuy = 1 | x_i) >= t; en caso contrario, 0
```

Luego se calcula el valor economico esperado de ese threshold:

```text
BusinessValue(t) = TP(t) * 2500 + FP(t) * (-900) + FN(t) * 0 + TN(t) * 0
```

La funcion objetivo del Sprint 5 es seleccionar el umbral que maximiza ese valor:

```text
t* = argmax BusinessValue(t), para t en {0.05, 0.06, ..., 0.95}
```

Con esta funcion de optimizacion, el threshold operativo recomendado es **0.70**.

### 4.3 Resultado economico

Comparacion principal:

| Escenario | Threshold | Business value | Valor promedio por vehiculo |
|---|---:|---:|---:|
| Referencia tecnica | 0.50 | USD 26,500 | USD 1.82 |
| Optimo economico | 0.70 | USD 1,079,600 | USD 73.96 |

El cambio de threshold genera una mejora estimada de **USD 1,053,100** respecto al threshold `0.50`.

## 5. Explicabilidad y dashboard

El dashboard de Sprint 5 debe permitir al stakeholder:

1. Ver KPIs principales del modelo final.
2. Ajustar y comparar thresholds, manteniendo `0.70` como recomendado.
3. Cargar un CSV o usar el test set procesado.
4. Visualizar predicciones, matriz de confusion, curva ROC y business value.
5. Revisar explicabilidad SHAP global y por instancia.

SHAP se incluye para responder por que una prediccion salio como alto riesgo o bajo riesgo. Esto es clave para que el modelo no sea una caja negra durante la demo y para preparar la integracion con la API del Sprint 6.

## 6. Riesgos y limitaciones

1. **Dependencia de supuestos economicos.** El threshold final depende de los valores TP, FP, FN y TN. Si negocio ajusta estos valores, debe recalcularse el threshold.
2. **Falsos negativos persistentes.** El threshold `0.70` reduce falsos positivos, pero deja pasar mas Bad Buys que `0.50`. Por eso se recomienda piloto controlado y revision manual.
3. **Drift de datos.** Cambios en precios, subastas, garantias o mix de vehiculos pueden degradar el modelo.
4. **Generalizacion.** El modelo fue entrenado con datos historicos y debe monitorearse con datos nuevos.
5. **Disponibilidad de variables.** Sprint 6 debe validar que las variables del contrato de entrada esten disponibles en produccion.

## 7. Recomendaciones accionables

- Desplegar el modelo en Sprint 6 como MVP controlado.
- Usar threshold `0.70` como umbral operativo inicial.
- Mantener threshold `0.50` solo como referencia tecnica para analisis comparativo.
- No automatizar rechazos hasta validar los costos con el sponsor.
- Implementar monitoreo de recall, precision, positive rate, business value, latencia y drift.
- Usar el paquete `handoff/` como contrato tecnico para construir la API FastAPI del Sprint 6.

## 8. Handoff para Sprint 6

El cierre de Sprint 5 deja preparados los siguientes artefactos:

- `handoff/model/final_model.pkl.dvc`
- `handoff/model/preproc_pipeline.pkl.dvc`
- `handoff/contracts/input_schema.json`
- `handoff/contracts/output_schema.json`
- `handoff/contracts/example_request.json`
- `handoff/contracts/example_response.json`
- `handoff/requirements.txt`
- `handoff/README_handoff.md`
- `handoff/KPIs_target_SLA.md`
- `dashboard/app.py`
- `reports/final_report.pdf`

## 9. Conclusion

El modelo genera valor cuando el threshold se define como una decision de negocio. La recomendacion final es avanzar con `LogisticRegression tuneado` y threshold `0.70` porque maximiza el business value esperado bajo los supuestos actuales, reduce falsos positivos y deja un contrato tecnico claro para iniciar Sprint 6.
