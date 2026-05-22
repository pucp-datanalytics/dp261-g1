# Handoff Sprint 5 -> Sprint 6

## Objetivo

Este paquete deja listo el contrato tecnico para que Sprint 6 empaquete el modelo como API, lo despliegue en AWS y conecte el dashboard al endpoint.

## Decision de negocio congelada para el MVP

- Modelo: **LogisticRegression tuneado**
- Threshold operativo final: **0.70**
- Criterio de seleccion: maximizar business value esperado
- Supuestos: TP=2500, FP=-900, FN=0, TN=0
- Decision: GO controlado hacia Sprint 6

## Artefactos incluidos

```text
handoff/
├── model/
│   ├── final_model.pkl.dvc
│   ├── preproc_pipeline.pkl.dvc
│   ├── model_metadata.json
│   └── README_model.md
├── contracts/
│   ├── input_schema.json
│   ├── output_schema.json
│   ├── example_request.json
│   └── example_response.json
├── requirements.txt
├── README_handoff.md
└── KPIs_target_SLA.md
```

## Como usarlo en Sprint 6

1. Sincronizar la rama main y materializar artefactos:

```bash
git pull
dvc pull
```

2. Verificar que existen los binarios:

```bash
ls models/final_model.pkl
ls handoff/model/final_model.pkl
```

Si se trabaja solo con `models/final_model.pkl`, tambien es valido porque el modelo final es un pipeline de sklearn con preprocessing incluido.

3. Construir la API FastAPI con estos endpoints minimos:

- `GET /health`: estado del servicio.
- `GET /version`: version de API, version/hash del modelo y threshold.
- `POST /predict`: recibe un JSON con el contrato de `input_schema.json` y devuelve `output_schema.json`.

4. El dashboard del Sprint 6 debe consumir la API real, no cargar el modelo local.

## Regla operativa

La API debe aplicar threshold `0.70` por defecto. Puede aceptar un threshold alternativo solo para pruebas, pero el dashboard debe marcar `0.70` como recomendado.

## Validaciones antes de cerrar Sprint 6

- `/health` responde 200.
- `/predict` con `example_request.json` devuelve el formato de `example_response.json`.
- No hay API keys ni secretos en el repositorio.
- El dashboard consume `API_URL` y `API_KEY` desde variables de entorno.
- Hay logs estructurados, metrica de latencia p95 y al menos una alarma activa.
