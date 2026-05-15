from pathlib import Path

RANDOM_STATE = 42
TARGET = "IsBadBuy"
SAMPLE_SIZE = 20000          # baja a 8000-12000 si SVM/KNN demora en tu laptop
TEST_SIZE = 0.20
CV_SPLITS = 5                # exigido para comparación robusta; baja a 3 solo para pruebas rápidas

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "06-kickAutomotriz.csv"
DATA_RAW_XLSX = PROJECT_ROOT / "data" / "raw" / "06-kickAutomotriz.xlsx"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# En este proyecto se definió que el falso negativo duele más:
# FN = auto malo marcado como buena compra => se acepta/compra una unidad riesgosa.
# Por ello, la selección de modelos y thresholds prioriza recall y F2.
PRIMARY_METRIC = "recall"
SECONDARY_METRIC = "f2"
CONTROL_METRIC = "precision"

# Supuestos iniciales para Sprint 5. Cámbialos si el negocio define otros costos.
BENEFIT_TP = 2500 #supuesto de pérdida evitada por cada bad buy
COST_FP = -900 #el costo de equivocarte prediciendo como malo un auto que sí era bueno.
COST_FN = 0
BENEFIT_TN = 0
