# main_dashboard.py
import streamlit as st
from dashboards.facebook.facebook_layout import render_facebook_dashboard
from dashboards.x_predictivo.x_layout import render_x_dashboard
from dashboards.benchmark.benchmark_layout import render_benchmark_dashboard
from dashboards.telegram.telegram_layout import render_telegram_dashboard

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(
    page_title="Dashboard Maestro - Inteligencia Digital",
    layout="wide"
)

# --- LIMPIAR CACHÉ (asegura carga limpia en Render) ---
st.cache_data.clear()

# --- BARRA SUPERIOR PERSONALIZADA ---
st.markdown(
    """
    <style>
        .top-bar {
            background-color: #0E1117;
            padding: 1rem 2rem;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-size: 1.3rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 1rem;
        }
        .top-bar.light {
            background-color: #f3f3f3;
            color: #222;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SELECTOR DE MODO VISUAL ---
modo = st.sidebar.radio("🎨 Modo de visualización", ["Oscuro", "Claro"])
modo_css = "top-bar" if modo == "Oscuro" else "top-bar light"

st.markdown(f'<div class="{modo_css}">Dashboard de Inteligencia Digital</div>', unsafe_allow_html=True)
st.markdown("Visualización unificada de análisis en redes sociales y canales de alerta.")
st.info("Selecciona una plataforma para ver sus métricas.")

# --- MENÚ DE NAVEGACIÓN ---
tabs = st.tabs(["Listening", "Analysis", "Benchmark", "Incidencias"])

# --- CONTENIDO DE CADA PESTAÑA ---
with tabs[0]:
    st.subheader("Dashboard de Listening")
    render_facebook_dashboard()

with tabs[1]:
    st.subheader("Dashboard de Analysis")
    render_x_dashboard()

with tabs[2]:
    st.subheader("Benchmark Institucional")
    render_benchmark_dashboard()

with tabs[3]:
    st.subheader("Alertas e Incidencias")
    render_telegram_dashboard()
