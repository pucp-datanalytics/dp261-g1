# dp261-g1

## Miembros del Proyecto

| Nombre | Rol |
|--------|-----|
| Pedro Shiguihara | Product Owner (P.O) |
| Davida Ponce | Product Manager |
| Freddy Nina | Business Analyst |
|  Alexandra Lozano | Data Engineer |
| Martin Bendezu | Data Analyst |
| Oliver Malqui | Prototype Developer |

## GitHub Project

Tablero del proyecto: [dp261-g1 Project](https://github.com/orgs/pucp-datanalytics/projects/1)

## Labels

Los labels del repositorio siguen una convencion de prefijos para organizar el trabajo:

| Prefijo | Color | Descripcion |
|---------|-------|-------------|
| `scrum/` | `#0052CC` | `scrum/user-story`, `scrum/task`, `scrum/bug`, `scrum/backlog`, `scrum/in-progress`, `scrum/blocked`, `scrum/review`, `scrum/done` |
| `crisp/` | `#006B75` | `crisp/sprint1-business-data-understanding`, `crisp/sprint2-data-preparation`, `crisp/sprint3-modeling-baseline`, `crisp/sprint4-modeling-advanced`, `crisp/sprint5-evaluation`, `crisp/sprint6-deployment` |
| `rol/` | `#E65100` | `rol/pm`, `rol/ba`, `rol/de`, `rol/da`, `rol/pd` |
| `entrega/` | `#5319E7` | `entrega/notebook`, `entrega/dataset`, `entrega/environment`, `entrega/dashboard`, `entrega/model`, `entrega/report`, `entrega/docs`, `entrega/api-rest`, `entrega/aws-deploy` |
| `prioridad/` | `#B60205` | `prioridad/high`, `prioridad/medium`, `prioridad/low` |

### Detalle de Labels

**`scrum/`**

| Label | Descripcion |
|-------|-------------|
| `scrum/user-story` | Historia de usuario |
| `scrum/task` | Tarea técnica del sprint |
| `scrum/bug` | Error o fallo en el entregable |
| `scrum/backlog` | Pendiente en el product backlog |
| `scrum/in-progress` | En desarrollo activo |
| `scrum/blocked` | Bloqueado, requiere atención |
| `scrum/review` | En revisión del equipo |
| `scrum/done` | Completado y aprobado |

**`crisp/`**

| Label | Descripcion |
|-------|-------------|
| `crisp/sprint1-business-data-understanding` | Sprint 1: Business & Data Understanding |
| `crisp/sprint2-data-preparation` | Sprint 2: Data Preparation |
| `crisp/sprint3-modeling-baseline` | Sprint 3: Modeling baseline |
| `crisp/sprint4-modeling-advanced` | Sprint 4: Modeling avanzado + tuning |
| `crisp/sprint5-evaluation` | Sprint 5: Evaluation + Business Value |
| `crisp/sprint6-deployment` | Sprint 6: Deployment MVP en AWS |

**`rol/`**

| Label | Descripcion |
|-------|-------------|
| `rol/pm` | Project Manager |
| `rol/ba` | Business Analyst |
| `rol/de` | Data Engineer |
| `rol/da` | Data Analyst |
| `rol/pd` | Prototype Developer |

**`entrega/`**

| Label | Descripcion |
|-------|-------------|
| `entrega/notebook` | Jupyter Notebook |
| `entrega/dataset` | Conjunto de datos (DVC) |
| `entrega/environment` | environment.yml / entorno Conda |
| `entrega/dashboard` | Visualización / Dashboard interactivo |
| `entrega/model` | Modelo entrenado y serializado |
| `entrega/report` | Informe o presentación ejecutiva |
| `entrega/docs` | Documentación del proyecto |
| `entrega/api-rest` | API REST del modelo desplegada |
| `entrega/aws-deploy` | Despliegue en AWS (EC2/Lambda/SageMaker) |

**`prioridad/`**

| Label | Descripcion |
|-------|-------------|
| `prioridad/high` | Prioridad alta |
| `prioridad/medium` | Prioridad media |
| `prioridad/low` | Prioridad baja |

## Milestones

| Sprint | Titulo | Fecha de entrega |
|--------|--------|------------------|
| 1 | Sprint 1 — Business & Data Understanding | 10/04/2026 |
| 2 | Sprint 2 — Data Preparation | 17/04/2026 |
| 3 | Sprint 3 — Modeling (Baseline) | 24/04/2026 |
| 4 | Sprint 4 — Modeling (Avanzado + Tuning) | 08/05/2026 |
| 5 | Sprint 5 — Evaluation + Business Value | 15/05/2026 |
| 6 | Sprint 6 — Deployment MVP en AWS | 22/05/2026 |

### Detalle de Milestones

**Sprint 1 — Business & Data Understanding** (entrega: 10/04/2026)

Comprender el problema de negocio y explorar los datos.
Entregables:
- Repositorio GitHub configurado con board Kanban, Issues creados y asignados, README (PM)
- 01_business.ipynb: problema de negocio, variable objetivo, KPIs y criterios de éxito (BA)
- environment.yml + 02_data_loading.ipynb: entorno Conda, DVC configurado, carga y verificación de datos (DE)
- 03_eda.ipynb: análisis de calidad, estadísticas descriptivas, visualizaciones y hallazgos (DA)
- 04_prototype.ipynb: prototipo interactivo con ipywidgets (PD)



<details>
<summary><strong>Setup del proyecto</strong></summary>

### Requisitos
- Git
- Miniconda o Anaconda
- Google Drive for desktop
- Acceso a la carpeta compartida de Drive: DVC-G1-Storage

### 1. Clonar el repositorio
git clone https://github.com/pucp-datanalytics/dp261-g1.git
cd dp261-g1

### 2. Crear y activar el entorno Conda
Si el archivo se llama environment.yml:
conda env create -f environment.yml
conda activate dp261-g1

Si el archivo se llama env.yml, usar:
conda env create -f env.yml
conda activate dp261-g1

### 3. Registrar el kernel de Jupyter
Esto ayuda a que VS Code/Jupyter reconozca correctamente el entorno.
python -m ipykernel install --user --name dp261-g1 --display-name "Python (dp261-g1)"

### 4. Configurar Google Drive Desktop
1. Instalar Google Drive for desktop
2. Iniciar sesión con la cuenta que tenga acceso a la carpeta compartida
3. Verificar que la carpeta DVC-G1-Storage aparezca en Drive (link: https://drive.google.com/drive/folders/1plBl9DATtwC6qpNztwBsGbRsbFX8ZPHJ?usp=sharing)
4. Obtener la ruta local de esa carpeta en la computadora

Importante: La ruta será distinta en cada máquina y sistema operativo.

Ejemplo en Mac:
/Users/usuario/Library/CloudStorage/GoogleDrive-correo@gmail.com/My Drive/DVC-G1-Storage

Ejemplo en Windows:
G:\.shortcut-targets-by-id\...\DVC-G1-Storage

### 5. Configurar el remote de DVC
Si es la primera vez en esa computadora:
dvc remote add --local -d teamdrive "RUTA_LOCAL_A_DVC-G1-Storage"

Si el remote teamdrive ya existía:
dvc remote modify --local teamdrive url "RUTA_LOCAL_A_DVC-G1-Storage"
dvc remote default --local teamdrive

Verificar: dvc remote list

Esta configuración se guarda en .dvc/config.local, por lo que es local por computadora y no debe subirse a GitHub.

### 6. Descargar los datos con DVC
dvc pull

Esto materializa los archivos reales en el proyecto, por ejemplo: data/raw/06-kickAutomotriz.csv

### 7. Ejecutar Jupyter Lab
jupyter lab

Si usas VS Code:
- seleccionar el intérprete dp261-g1
- seleccionar el kernel Python (dp261-g1)

### 8. Orden recomendado de ejecución
notebooks/01_business.ipynb
notebooks/02_data_loading.ipynb
notebooks/03_eda.ipynb
notebooks/04_prototype.ipynb

### Notas importantes

1. **No volver a inicializar Git ni DVC**  
   El repositorio ya viene inicializado, así que no se debe ejecutar nuevamente:

   - `git init`
   - `dvc init`

2. **No agregar nuevamente el dataset si ya está rastreado**  
   Esto solo lo hace quien incorpora un dataset nuevo al proyecto. No ejecutar:

   - `dvc add "data/raw/06-kickAutomotriz.csv"`

3. **El archivo CSV real no se sube a GitHub**  
   Git sí versiona:

   - `*.dvc`
   - `dvc.yaml` / `dvc.lock` si existieran
   - notebooks, código y documentación

   Git no debe subir:

   - `data/raw/06-kickAutomotriz.csv`

4. **El remote de DVC es local por máquina**  
   Aunque todos usan la misma carpeta compartida de Drive, cada integrante debe configurar su propia ruta local una sola vez.

5. **Flujo normal de trabajo**

   ```bash
   git pull
   conda activate dp261-g1
   dvc pull
   jupyter lab

</details>
------------------------------------------------------
------------------------------------------------------


**Sprint 2 — Data Preparation** (entrega: 17/04/2026)

Limpiar, transformar y construir features del dataset.
Entregables:
- Pipeline de limpieza y transformación de datos
- Ingeniería de características (feature engineering)
- Dataset final listo para modelado versionado con DVC
- Notebook documentado con cada decisión de preprocesamiento

**Sprint 3 — Modeling (Baseline)** (entrega: 24/04/2026)

Entrenar y comparar modelos de clasificación baseline.
Entregables:
- Notebook de experimentación con modelos baseline (Logistic Regression, Decision Tree, etc.)
- Métricas comparativas: accuracy, precision, recall, F1, ROC-AUC
- Selección del mejor modelo baseline con justificación
- Registro de experimentos documentado

**Sprint 4 — Modeling (Avanzado + Tuning)** (entrega: 08/05/2026)

Optimizar hiperparámetros y aplicar modelos avanzados.
Entregables:
- Notebook con modelos avanzados (Random Forest, XGBoost, etc.)
- Optimización de hiperparámetros (GridSearchCV / Optuna)
- Comparativa final de modelos con métricas detalladas
- Modelo final seleccionado y serializado (pickle/joblib)

**Sprint 5 — Evaluation + Business Value** (entrega: 15/05/2026)

Evaluar el Business Value y documentar resultados finales.
Entregables:
- Evaluación del modelo frente a los KPIs definidos en Sprint 1
- Análisis de errores e interpretabilidad (SHAP, feature importance)
- Dashboard interactivo con resultados para stakeholders
- Informe ejecutivo con conclusiones y recomendaciones

**Sprint 6 — Deployment MVP en AWS** (entrega: 22/05/2026)

Desplegar el mejor modelo como MVP en AWS.
Entregables:
- API REST del modelo desplegada en AWS (EC2, Lambda o SageMaker)
- Dashboard interactivo accesible en producción
- Documentación de arquitectura y guía de uso
- Demo funcional del MVP presentada al stakeholder

