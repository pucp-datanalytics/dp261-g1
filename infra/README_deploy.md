# Sprint 6 — Guía de despliegue AWS

Esta carpeta deja la base para PB-20: subir la imagen Docker a ECR y desplegarla en AWS.

## Arquitectura recomendada para MVP

Para este proyecto se recomienda **EC2 + Docker** como primera entrega porque es más fácil de demostrar en una semana y evita problemas típicos de Lambda con dependencias pesadas de ML.

Flujo:

1. GitHub contiene `api/`, `Dockerfile`, `dashboard/`, `monitoring/` y documentación.
2. Docker empaqueta la API FastAPI y el modelo materializado con `dvc pull`.
3. AWS ECR guarda la imagen.
4. AWS EC2 ejecuta el contenedor.
5. El dashboard consume el endpoint público/privado de la API.
6. CloudWatch guarda logs y alarmas.

## Pre-requisitos locales

```bash
git pull
dvc pull
pip install -r api/requirements.txt
```

Validar API local:

```bash
uvicorn api.main:app --reload
python api/smoke_test.py --url http://localhost:8000
```

Validar Docker local:

```bash
docker build -t badbuy-api -f api/Dockerfile .
docker run --rm -p 8000:8000 \
  -e REQUIRE_API_KEY=false \
  -e MODEL_THRESHOLD=0.70 \
  badbuy-api
python api/smoke_test.py --url http://localhost:8000
```

## Variables para AWS

```bash
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID="123456789012"
export ECR_REPOSITORY="badbuy-api"
export IMAGE_TAG="$(git rev-parse --short HEAD)"
export API_KEY="<crear-secret>"
```

## Despliegue manual a ECR

```bash
bash infra/deploy_ec2.sh
```

## Secrets esperados en GitHub Actions

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_ACCOUNT_ID`
- `ECR_REPOSITORY`
- `API_KEY`

## Después del deploy

1. Copiar el endpoint real de la API.
2. Configurar el dashboard:

```bash
export API_URL="https://<endpoint-aws>"
export API_KEY="<secret>"
streamlit run dashboard/app.py
```

3. Ejecutar smoke test contra AWS:

```bash
API_KEY="<secret>" python api/smoke_test.py --url "https://<endpoint-aws>"
```

4. Adjuntar evidencia en el issue PB-20: imagen ECR, contenedor corriendo, `/health`, `/version`, `/predict`, logs CloudWatch.
