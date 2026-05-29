# KPIs y SLA objetivo del MVP

- Latencia p95 objetivo local/AWS MVP: < 2s.
- Error rate 5xx: < 1% sostenido.
- Endpoint `/health`: debe responder 200.
- Cada respuesta de `/predict` debe incluir `model_version`, `threshold` y `risk_segment`.
- Monitoreo de drift: mensual durante piloto; semanal si el volumen crece.
