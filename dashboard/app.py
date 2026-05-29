"""Dashboard comercial Streamlit para el MVP DP261.

Incluye dos vistas en el cuerpo principal:
1. Evaluación comercial: carga de vehículos, semáforo y detalle por caso.
2. Análisis del modelo: interpretabilidad global tipo SHAP.

Uso local:
    API_URL=http://localhost:8000 streamlit run dashboard/app.py --server.port 8502
"""
from __future__ import annotations

import streamlit as st

from _shared import render_commercial_dashboard
from _sidebar import render_technical_sidebar

st.set_page_config(page_title="Semáforo Bad Buy", page_icon="🚦", layout="wide")

api_error = render_technical_sidebar("Dashboard comercial · Evaluación y análisis", show_legend=True)
render_commercial_dashboard(api_error)
