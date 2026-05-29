# Guía UX del dashboard comercial

## Usuario objetivo
El dashboard está diseñado para un agente comercial o comprador de vehículos que no necesita ver métricas técnicas en primera instancia. Su pregunta principal es: **¿debo continuar, revisar o detener la compra de este vehículo?**

## Decisiones de diseño

1. **Semáforo visual**
   - 🔴 ROJO: alto riesgo. Acción sugerida: detener compra o derivar a revisión experta.
   - 🟠 ÁMBAR: riesgo medio. Acción sugerida: revisión manual antes de comprar.
   - 🟢 VERDE: riesgo bajo. Acción sugerida: continuar evaluación comercial.

2. **Datos identificadores del vehículo**
   El resultado no solo muestra el score. También incluye subasta, marca, modelo, año, antigüedad, kilometraje, costo base, garantía y estado. Esto permite que el usuario ubique rápidamente el vehículo dentro del lote.

3. **Explicación de factores**
   La API devuelve los principales factores de la predicción. Si SHAP no está disponible para el estimador final, se usa un fallback transparente basado en importancia de variables. Para Bagging sobre árboles, se promedia la importancia de los árboles internos.

4. **Detalle técnico colapsado**
   La información de API, versión de modelo, JSON crudo y método de explicación está en desplegables. Esto permite sustentar técnicamente sin contaminar la vista comercial.

5. **Flujo simple**
   - Cargar CSV.
   - Presionar “Evaluar vehículos”.
   - Revisar resumen del lote.
   - Abrir detalle de los vehículos rojos o ámbar.
   - Descargar resultados.

## Advertencia sobre explicabilidad
Cuando el método de explicación sea `bagging_tree_importance_fallback`, la lectura debe entenderse como **factores más relevantes**, no como SHAP causal estricto. Se mantiene porque permite al usuario comercial entender qué variables pesaron más sin dejar vacío el dashboard.
