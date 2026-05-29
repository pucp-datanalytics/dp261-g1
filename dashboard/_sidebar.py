
from __future__ import annotations

from _shared import (
    RISK_STYLE,
    api_url,
    get_api_status,
    render_commercial_dashboard,
    render_presentation_dashboard,
    st,
)


def render_technical_sidebar(page_label: str, show_legend: bool = True) -> str | None:
    """Panel técnico lateral ocultable de la demo.

    La información técnica queda fuera del cuerpo principal para no distraer a la
    audiencia comercial. Streamlit permite ocultar/mostrar el sidebar con el
    control nativo de la esquina superior izquierda.
    """
    with st.sidebar:
        st.title("DP261 Bad Buy")
        st.caption(page_label)
        st.markdown("---")
        st.markdown("### ⚙️ Panel técnico")
        st.caption("Estado de API, endpoint y versión del modelo. Ocúltalo durante la explicación comercial.")
        st.code(api_url(), language="text")

    health, version, api_error = get_api_status()

    with st.sidebar:
        if api_error:
            st.warning("API no conectada")
            st.caption(api_error)
        else:
            st.success("API conectada")
            st.write(f"**Estado:** `{health.get('status', 'N/A')}`")
            st.write(f"**Modelo:** `{version.get('model_name', 'N/A')}`")
            st.write(f"**Versión modelo:** `{version.get('model_version', 'N/A')}`")
            st.write(f"**Versión API:** `{version.get('api_version', 'N/A')}`")
            with st.expander("Ver JSON técnico de health/version", expanded=False):
                st.json({"health": health, "version": version})

        if show_legend:
            st.markdown("---")
            st.markdown("### Leyenda semáforo")
            for segment, style in RISK_STYLE.items():
                st.markdown(f"{style['emoji']} **{segment}**: {style['action']}")

    return api_error
