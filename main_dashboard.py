# main_dashboard.py
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.badges import badge

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

# --- Modo claro / oscuro ---
theme = st.sidebar.radio("Selecciona un modo", ["Claro", "Oscuro"])

# Ajuste de estilos según modo
if theme == "Claro":
    bg_color = "#FFFFFF"
    text_color = "#000000"
else:
    bg_color = "#0E1117"
    text_color = "#FFFFFF"

# Limpiar caché de Streamlit para evitar problemas de colores
st.cache_data.clear()
st.cache_resource.clear()

# --- Estilos CSS ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .css-1d391kg {{
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Encabezado principal ---
st.title("Dashboard de inteligencia digital", anchor=None)
st.markdown("Visualización unificada de análisis en redes sociales y canales de alerta.")

# --- Aviso inicial ---
st.info("Selecciona una plataforma para ver sus métricas")

# --- Menú de navegación (pestañas) ---
tabs = st.tabs(["Listening", "Analysis", "Benchmark", "Incidencias"])

# --- Contenido de cada pestaña ---
with tabs[0]:
    st.header("Listening")
    render_facebook_dashboard()

with tabs[1]:
    st.header("Analysis")
    render_x_dashboard()

with tabs[2]:
    st.header("Benchmark")
    render_benchmark_dashboard()

with tabs[3]:
    st.header("Incidencias")
    render_telegram_dashboard()



