# main_dashboard.py
import streamlit as st

# Importar dashboards individuales
from dashboards.facebook.facebook_layout import render_facebook_dashboard
from dashboards.x_predictivo.x_layout import render_x_dashboard
from dashboards.benchmark.benchmark_layout import render_benchmark_dashboard
from dashboards.telegram.telegram_layout import render_telegram_dashboard

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(
    page_title="Dashboard Maestro - Inteligencia Digital",
    layout="wide"
)

# --- SIDEBAR: Modo Claro/Oscuro ---
modo = st.sidebar.radio("Selecciona un modo de visualización:", ["Claro", "Oscuro"])
if modo == "Claro":
    st.markdown(
        """
        <style>
        .main {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .main {
            background-color: #0E1117;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- ENCABEZADO PRINCIPAL ---
st.title("Dashboard de Inteligencia Digital")
st.markdown("Visualización unificada de análisis en redes sociales y canales de alerta.")

st.info("Selecciona una plataforma para ver sus métricas en las pestañas de abajo.")

# --- MENÚ DE NAVEGACIÓN ---
tabs = st.tabs(["Listening", "Analysis", "Benchmark", "Incidencias"])

# --- CONTENIDO DE CADA PESTAÑA ---
with tabs[0]:
    st.header("Listening")
    st.subheader("Dashboard de métricas de redes sociales")
    render_facebook_dashboard()

with tabs[1]:
    st.header("Analysis")
    st.subheader("Análisis predictivo de sentimiento y hashtags en X (Twitter)")
    render_x_dashboard()

with tabs[2]:
    st.header("Benchmark")
    st.subheader("Comparativo de métricas entre instituciones y redes sociales")
    render_benchmark_dashboard()

with tabs[3]:
    st.header("Incidencias")
    st.subheader("Alertas de riesgo y mensajes críticos")
    render_telegram_dashboard()


