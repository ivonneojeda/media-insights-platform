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

# --- Modo claro / oscuro ---
theme = st.sidebar.radio("Personalización de tema", ["Claro", "Oscuro"])

# Ajuste de estilos según modo
if theme == "Claro":
    bg_color = "#FFFFFF"
    text_color = "#000000"
    tab_bg_color = "#E0E0E0"
    tab_text_color = "#000000"
else:
    bg_color = "#0E1117"
    text_color = "#FFFFFF"
    tab_bg_color = "#1B1B2F"
    tab_text_color = "#FFFFFF"

# --- Limpiar caché para evitar problemas de colores ---
st.cache_data.clear()
st.cache_resource.clear()

# --- Estilos CSS generales ---
st.markdown(
    f"""
    <style>
    /* Fondo y texto general */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    /* Sidebar */
    .css-1d391kg, .css-1avcm0n {{
        color: {text_color};
    }}
    /* Ajuste de títulos de pestañas */
    .css-1offfwp {{
        background-color: {tab_bg_color} !important;
        color: {tab_text_color} !important;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Encabezado principal ---
st.title("Dashboard de inteligencia digital")
st.markdown("Visualización unificada de análisis en redes sociales y canales de alerta.")

# --- Mensaje inicial adaptado a tema ---
st.markdown(
    f"""
    <div style="
        background-color: {tab_bg_color};
        color: {text_color};
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 15px;
        ">
        Selecciona una plataforma para ver sus métricas
    </div>
    """,
    unsafe_allow_html=True
)

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
