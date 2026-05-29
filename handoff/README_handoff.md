# Handoff Sprint 6

Este paquete contiene lo mínimo para que Deployment use el modelo sin adivinar contratos.

- `model/final_model.pkl`: pipeline completo serializado.
- `model/model_metadata.json`: versión, threshold, función de negocio y métricas.
- `contracts/example_request.json`: ejemplo de input crudo.
- `contracts/example_response.json`: ejemplo de output de API.
- `contracts/input_schema.json`: campos esperados.
- `contracts/output_schema.json`: campos devueltos.

El modelo ya incluye feature engineering y preprocesamiento, por lo que el consumidor solo envía columnas originales.
