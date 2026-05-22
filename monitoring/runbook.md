# Runbook Sprint 6 — API Bad Buy MVP

## Objetivo

Documento operativo para investigar incidentes, validar salud del MVP y realizar rollback.

## Servicios cubiertos

- API FastAPI: `/health`, `/version`, `/predict`
- Contenedor Docker en AWS EC2/ECS/Lambda
- Dashboard Streamlit consumiendo API
- Logs y alarmas CloudWatch

## Métricas objetivo

- `/health` responde HTTP 200.
- Latencia p95 menor a 500 ms en condiciones normales de MVP.
- Error rate 5xx menor a 1% sostenido.
- Endpoint `/predict` protegido con API Key/IAM en producción.
- Respuesta incluye `api_version`, `model_version` y `model_sha` vía `/version`.

## Validación rápida

```bash
curl -i "$API_URL/health"
curl -i "$API_URL/version"
API_KEY="$API_KEY" python api/smoke_test.py --url "$API_URL"
```

## Incidente: API no responde

1. Revisar `/health`.
2. Revisar si el contenedor está arriba:

```bash
docker ps
docker logs badbuy-api --tail 100
```

3. Revisar CloudWatch Logs.
4. Confirmar que `models/final_model.pkl` fue incluido en la imagen o montado correctamente.
5. Reiniciar contenedor si aplica:

```bash
docker restart badbuy-api
```

## Incidente: errores 401

Causa probable: API key ausente o incorrecta.

1. Confirmar variable `REQUIRE_API_KEY=true`.
2. Confirmar que dashboard tiene `API_KEY` configurada.
3. Probar con header:

```bash
curl -H "x-api-key: $API_KEY" "$API_URL/version"
```

## Incidente: errores 400 en /predict

Causa probable: contrato de entrada incumplido o columnas faltantes.

1. Probar con `handoff/contracts/example_request.json`.
2. Revisar campos requeridos: `VehicleAge`, `VehOdo`, `VehBCost`, `WarrantyCost`.
3. Verificar logs de API para detalle del error.

## Rollback

1. Identificar tag anterior en ECR.
2. Detener contenedor actual.
3. Levantar tag anterior:

```bash
docker rm -f badbuy-api
docker run -d --name badbuy-api -p 80:8000 \
  -e REQUIRE_API_KEY=true \
  -e API_KEY="$API_KEY" \
  "$ECR_URL:<previous_tag>"
```

4. Re-ejecutar smoke test.

## Backlog v2

- Registrar predicciones y feedback real para medir performance post-deploy.
- Monitorear drift de variables contra baseline de Sprint 2.
- A/B testing de thresholds o modelos.
- Autenticación más robusta si el endpoint se expone públicamente.
