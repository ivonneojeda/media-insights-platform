import streamlit as st
from dashboards.facebook.facebook_layout import render_facebook_dashboard
from dashboards.benchmark.benchmark_layout import render_benchmark_dashboard
from dashboards.telegram.telegram_layout import render_telegram_dashboard

# Importamos solo la nueva función del dashboard de X
from dashboards.x_predictivo.x_layout import show_graph_layout

# =============================
# INTERFAZ PRINCIPAL DEL DASHBOARD
# =============================

st.set_page_config(page_title="Dashboard Maestro", layout="wide")

st.sidebar.title("Panel de navegación")
selected_dashboard = st.sidebar.radio(
    "Selecciona una vista:",
    [
        "Análisis X (grafo)",
        "Facebook",
        "Telegram",
        "Benchmark"
    ]
)

# =============================
# RUTEO ENTRE DASHBOARDS
# =============================
if selected_dashboard == "Análisis X (grafo)":
    st.title("Análisis de redes en X (Twitter)")
    show_graph_layout("data/Conversacion_UNAM_limpio.csv")

elif selected_dashboard == "Facebook":
    render_facebook_dashboard()

elif selected_dashboard == "Telegram":
    render_telegram_dashboard()

elif selected_dashboard == "Benchmark":
    render_benchmark_dashboard()
