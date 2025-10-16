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

# --- SIDEBAR: selector de modo ---
st.sidebar.title("Personalización de tema")
theme = st.sidebar.radio("Selecciona un modo de vista", ["🌞 Claro", "🌚 Oscuro"], index=0)

# --- Definir colores según el tema seleccionado ---
if "🌞" in theme:
    bg_color = "#FFFFFF"
    text_color = "#1A1A1A"
    header_color = "#0A66C2"
else:
    bg_color = "#0E1117"
    text_color = "#F5F5F5"
    header_color = "#66B2FF"

# --- Limpiar caché para asegurar que los cambios se apliquen ---
st.cache_data.clear()
st.cache_resource.clear()

# --- Estilos CSS globales ---
st.markdown(
    f"""
    <style>
    /* Fondo general */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        transition: background-color 0.5s ease, color 0.5s ease;
    }}

    /* Encabezados */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color};
    }}

    /* Pestañas */
    div[data-baseweb="tab"] {{
        color: {text_color};
        font-weight: 500;
        background-color: rgba(0,0,0,0.03);
        border-radius: 10px 10px 0 0;
        padding: 0.4em 1em;
        margin-right: 0.3em;
    }}
    div[data-baseweb="tab"]:hover {{
        background-color: rgba(100,100,100,0.1);
    }}
    div[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {header_color};
        color: white !important;
    }}

    /* Texto y métricas */
    .stMarkdown, .stText, .stDataFrame, .stMetric {{
        color: {text_color};
    }}

    /* Scrollbar más sutil */
    ::-webkit-scrollbar {{
        width: 6px;
    }}
    ::-webkit-scrollbar-thumb {{
        background-color: #999;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- ENCABEZADO PRINCIPAL ---
colored_header(
    label="Dashboard Maestro de Inteligencia Digital",
    description="Visualización unificada de análisis en redes sociales y canales de alerta.",
    color_name="blue-70",
)

# --- MENSAJE INICIAL ---
st.info("Selecciona una pestaña para explorar las métricas en tiempo real.")

# --- MENÚ DE NAVEGACIÓN ---
tabs = st.tabs(["Listening", "Analysis", "Benchmark", "Incidencias"])

# --- CONTENIDO DE CADA PESTAÑA ---
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
