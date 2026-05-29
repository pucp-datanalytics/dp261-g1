# Guía de despliegue MVP

## Local con Docker

```bash
docker build -f api/Dockerfile -t dp261-badbuy-api .
docker run --rm -p 8000:8000 dp261-badbuy-api
```

## AWS sugerido

Para MVP: EC2 t3.micro o ECS/Fargate simple. La imagen incluye modelo y código, por lo que no necesita S3 para demo.
En producción, mover el modelo a S3 versionado y montar con `MODEL_PATH`.

Variables recomendadas:

```bash
REQUIRE_API_KEY=true
API_KEY=<secreto>
MODEL_VERSION=final-entrega-dp261-v1
```
