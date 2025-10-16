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
theme = st.sidebar.radio("Selecciona un modo", ["Claro", "Oscuro"], index=0)

# --- Colores según modo ---
if theme == "Claro":
    bg_color = "#FFFFFF"
    text_color = "#000000"
    sidebar_bg = "#F0F2F6"
    tab_bg = "#E0E0E0"
    tab_active_bg = "#4A90E2"  # azul para pestaña activa
    tab_text_color = "#000000"
    tab_active_text = "#FFFFFF"
else:
    bg_color = "#0E1117"
    text_color = "#FFFFFF"
    sidebar_bg = "#11151C"
    tab_bg = "#1C1F26"
    tab_active_bg = "#1E90FF"  # azul para pestaña activa
    tab_text_color = "#FFFFFF"
    tab_active_text = "#FFFFFF"

# --- Limpiar caché ---
st.cache_data.clear()
st.cache_resource.clear()

# --- Estilos CSS ---
st.markdown(
    f"""
    <style>
    /* Fondo general */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    /* Sidebar */
    .css-1d391kg, .css-1oe6wy1 {{
        background-color: {sidebar_bg};
        color: {text_color};
    }}

    /* Tabs */
    .stTabs [role="tab"] {{
        background-color: {tab_bg};
        color: {tab_text_color};
        padding: 0.5rem 1rem;
        border-radius: 0.5rem 0.5rem 0 0;
        margin-right: 0.25rem;
    }}
    .stTabs [role="tab"]:hover {{
        background-color: {tab_active_bg};
        color: {tab_active_text};
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background-color: {tab_active_bg};
        color: {tab_active_text};
        font-weight: bold;
    }}

    /* Headers dentro de pestañas */
    h1, h2, h3, h4 {{
        color: {text_color};
    }}

    /* Texto general */
    .stMarkdown, .stText {{
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Encabezado principal ---
st.title("Dashboard de inteligencia digital")
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
