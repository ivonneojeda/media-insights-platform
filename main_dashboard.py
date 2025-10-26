# main_dashboard.py
import streamlit as st

# Dashboards individuales
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

if theme == "Claro":
    bg_color = "#FFFFFF"
    text_color = "#2F2F2F"
else:
    bg_color = "#0E1117"
    text_color = "#FFFFFF"

# --- Sidebar con color fijo ---
sidebar_bg = "#1E293B"         # gris azulado oscuro (constante)
sidebar_text_color = "#FFFFFF" # texto blanco siempre

# --- Estilos CSS ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {sidebar_text_color} !important;
    }}
    /* Texto general */
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Encabezado principal ---
st.title("Dashboard de inteligencia digital")
st.markdown("Análisis en redes sociales y canales de alerta.")

# --- Menú lateral como pestañas ---
selected_dashboard = st.sidebar.radio(
    "Selecciona una plataforma",
    ["Listening", "Analysis", "Benchmark", "Incidencias"]
)

# --- Contenido de cada sección ---
if selected_dashboard == "Listening":
    st.header("Listening")
    render_facebook_dashboard()

elif selected_dashboard == "Analysis":
    st.header("Analysis")
    # Llamada al dashboard de X (grafo + heatmap + gauge)
    render_x_dashboard()

elif selected_dashboard == "Benchmark":
    st.header("Benchmark")
    render_benchmark_dashboard()

elif selected_dashboard == "Incidencias":
    st.header("Incidencias")
    render_telegram_dashboard()
