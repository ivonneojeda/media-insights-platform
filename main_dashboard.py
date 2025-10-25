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
theme = st.sidebar.radio("Selecciona un modo", ["Claro", "Oscuro"])

# --- Colores principales ---
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

    import pandas as pd
    import streamlit.components.v1 as components
    from dashboards.x_predictivo.x_layout import render_interactive_graph

    # --- 1) Cargar CSV ---
    csv_url = "https://raw.githubusercontent.com/ivonneojeda/media-insights-platform/main/data/Conversaci%C3%B3n%20sobre%20UNAM%205-7oct25%20-%20ISO.csv"
    try:
        df_historico = pd.read_csv(csv_url, encoding='utf-8')
    except Exception as e:
        st.error(f"No se pudo cargar el CSV: {e}")
        st.stop()

    # --- 2) Normalizar columnas ---
    for col in ['hashtags', 'keywords', 'mentions']:
        if col in df_historico.columns:
            df_historico[col] = df_historico[col].fillna("").astype(str).str.split(r"[;,]\s*")

    # --- 3) Selección de capas ---
    available_layers = ['hashtags', 'mentions', 'keywords']
    selected_layers = st.multiselect(
        "Selecciona las capas para el grafo:", 
        available_layers, 
        default=available_layers
    )

    if not selected_layers:
        st.warning("Selecciona al menos una capa para visualizar el grafo.")
        st.stop()

    # --- 4) Renderizar grafo con manejo seguro de errores ---
    try:
        html_content, G = render_interactive_graph(df_historico, selected_layers, min_degree=2)
        if isinstance(html_content, str) and html_content.startswith("Error:"):
            st.error(html_content)
        else:
            components.html(html_content, height=700, scrolling=True)
    except Exception as e:
        st.error(f"Ocurrió un error al generar el grafo: {e}")


elif selected_dashboard == "Benchmark":
    st.header("Benchmark")
    render_benchmark_dashboard()

elif selected_dashboard == "Incidencias":
    st.header("Incidencias")
    render_telegram_dashboard()
