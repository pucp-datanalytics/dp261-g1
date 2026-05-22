# Modelo para Sprint 6

Este directorio contiene punteros DVC para los artefactos que Sprint 6 debe usar.

- `final_model.pkl.dvc`: modelo final recomendado. En la practica es el pipeline de sklearn con preprocessing + LogisticRegression.
- `preproc_pipeline.pkl.dvc`: artefacto de preprocessing separado, incluido para cumplir el contrato del Sprint 6. Si `final_model.pkl` ya contiene el pipeline completo, la API puede cargar solo `final_model.pkl`.
- `model_metadata.json`: threshold operativo final y supuestos de negocio.

Para materializar los archivos binarios:

```bash
dvc pull handoff/model/final_model.pkl.dvc
dvc pull handoff/model/preproc_pipeline.pkl.dvc
```

Threshold operativo: `0.70`.
