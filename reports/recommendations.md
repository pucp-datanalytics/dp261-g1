# Recomendaciones finales - Sprint 5

1. Avanzar a Sprint 6 con un MVP controlado del modelo final.
2. Usar **LogisticRegression tuneado** como modelo base por balance entre desempeno, estabilidad e interpretabilidad.
3. Adoptar **threshold 0.70** como umbral operativo inicial porque maximiza el business value bajo los supuestos confirmados.
4. No automatizar rechazos sin validacion de negocio; usar el modelo como soporte a decision.
5. Monitorear recall, precision, positive rate, business value y drift.
6. Validar los supuestos economicos con sponsor antes de produccion.
7. Usar `handoff/` como contrato tecnico para FastAPI, Docker, AWS y dashboard consumiendo API.
