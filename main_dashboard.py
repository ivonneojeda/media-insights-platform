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

# --- ESTILO DEL CONTENEDOR ---
# Se puede cambiar el color de fondo del contenedor principal
st.markdown(
    """
    <style>
    .main > div {
        background-color: #F9F9F9;  /* Fondo claro */
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- ENCABEZADO PRINCIPAL ---
st.title("Dashboard de inteligencia digital")
st.markdown("Visualización unificada de análisis en redes sociales y canales de alerta.")

# --- MENÚ LATERAL ---
st.sidebar.title("Secciones del Dashboard")
st.sidebar.markdown("Selecciona una plataforma para ver sus métricas:")

section = st.sidebar.radio(
    "",
    ["Listening", "Analysis", "Benchmark", "Incidencias"]
)

# --- CONTENIDO DE CADA SECCIÓN ---
if section == "Listening":
    st.subheader("Dashboard Listening")
    st.markdown("Análisis de métricas y sentimiento en Facebook.")
    render_facebook_dashboard()

elif section == "Analysis":
    st.subheader("Dashboard Analysis")
    st.markdown("Análisis predictivo de sentimiento y hashtags en X (Twitter).")
    render_x_dashboard()

elif section == "Benchmark":
    st.subheader("Benchmark Institucional")
    st.markdown("Comparativo de métricas entre diferentes redes sociales.")
    render_benchmark_dashboard()

elif section == "Incidencias":
    st.subheader("Alertas de riesgo")
    st.markdown("Registro de alertas y nivel de riesgo de mensajes en Telegram.")
    render_telegram_dashboard()

