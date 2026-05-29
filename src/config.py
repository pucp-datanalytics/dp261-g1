"""Configuración central del proyecto DP261 - Bad Buy Automotriz.

La clase positiva es `IsBadBuy = 1`: vehículo defectuoso / mala compra.
La selección final del modelo se hace con una función económica sobre la matriz
 de confusión usando threshold estándar 0.5, según la observación de la entrega final.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "06-kickAutomotriz.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "IsBadBuy"
RANDOM_STATE = 42
TEST_SIZE = 0.20
# Muestra estratificada para experimentación. El modelo final se reentrena con todo el train.
SAMPLE_SIZE = 3000
CV_SPLITS = 2

# Búsqueda controlada para laptops de estudiantes.
RANDOM_SEARCH_ITER = 2
OPTUNA_TRIALS = 2

# Ecuación de valor de negocio, aplicada con threshold estándar 0.5.
# TP: detectamos Bad Buy y evitamos una compra defectuosa.
# TN: aprobamos correctamente un buen vehículo y capturamos margen operativo.
# FP: rechazamos un buen vehículo y perdemos oportunidad comercial.
# FN: compramos un Bad Buy sin detectarlo; es el error más costoso.
BENEFIT_TP = 2500
BENEFIT_TN = 600
COST_FP = -900
COST_FN = -4500
BUSINESS_THRESHOLD = 0.50
