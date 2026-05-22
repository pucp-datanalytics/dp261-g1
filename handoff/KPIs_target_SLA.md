# KPIs target y SLA - Sprint 6 MVP

## KPIs de negocio

| KPI | Target inicial | Frecuencia |
|---|---:|---|
| Threshold operativo | 0.70 | Fijo al inicio del MVP |
| Precision minima esperada | >= 51.1% referencia test | Mensual |
| Recall referencia | 36.7% en test | Mensual |
| Positive rate referencia | 8.8% en test | Semanal |
| Business value referencia | USD 1,079,600 en test | Mensual |

## SLAs tecnicos del MVP

| Servicio | Target |
|---|---:|
| `GET /health` | HTTP 200 y latencia p95 < 500 ms |
| `POST /predict` | HTTP 200 y latencia p95 < 2 s |
| Error rate 5xx | < 1% sostenido |
| Disponibilidad demo | >= 95% durante ventana de Sprint Review |
| Costo AWS diario | Alarma si supera el presupuesto definido por el equipo |

## Monitoreo minimo

- Latencia p50, p95 y p99.
- Throughput: requests/minuto.
- Error rate 4xx y 5xx.
- Distribucion de `risk_score` vs test set.
- Positive rate diario.
- Drift mensual sobre variables numericas principales: `VehOdo`, `VehBCost`, `WarrantyCost`, precios MMR y `VehicleAge`.

## Alarma minima requerida

- Error rate 5xx > 1% por 5 minutos.
- Latencia p95 > 2 segundos por 5 minutos.
- Costo diario AWS > limite acordado.
