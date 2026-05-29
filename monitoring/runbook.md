# Runbook operativo MVP

## Señales a revisar
- `/health` debe responder `status=ok`.
- Error rate 5xx sostenido debe permanecer bajo 1%.
- Latencia p95 objetivo: menor a 2 segundos en MVP.
- Distribución de `risk_score` no debe cambiar drásticamente contra el test set.

## Incidente: API caída
1. Verificar logs del contenedor o CloudWatch.
2. Probar `/health`.
3. Confirmar que `MODEL_PATH` existe y que `models/final_model.pkl` está disponible.
4. Hacer rollback a la imagen anterior si el deploy fue reciente.

## Incidente: predicciones fallan
1. Validar que el CSV tenga columnas del dataset original.
2. Probar `handoff/contracts/example_request.json`.
3. Revisar drift o cambios de tipo de datos en columnas críticas.
