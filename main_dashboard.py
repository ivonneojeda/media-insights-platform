# main_dashboard.py
import streamlit as st

# Importar los dashboards individuales
from dashboards.facebook.facebook_layout import render_facebook_dashboard
from dashboards.x_predictivo.x_layout import render_x_dashboard
from dashboards.benchmark.benchmark_layout import render_benchmark_dashboard
from dashboards.telegram.telegram_layout import render_telegram_dashboard

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(
    page_title="Dashboard Maestro - Inteligencia Digital",
    layout="wide"
)

# --- ENCABEZADO PRINCIPAL ---
st.title("📊 Dashboard Maestro de Inteligencia Digital")
st.markdown("Visualización unificada de análisis en redes sociales y canales de alerta.")

# --- MENÚ DE NAVEGACIÓN ---
tabs = st.tabs(["Facebook", "X (Twitter)", "Benchmark", "Alertas Telegram"])

# --- CONTENIDO DE CADA PESTAÑA ---
with tabs[0]:
    st.header("📘 Facebook")
    render_facebook_dashboard()

with tabs[1]:
    st.header("🐦 X (Twitter)")
    render_x_dashboard()

with tabs[2]:
    st.header("📈 Benchmark Institucional")
    render_benchmark_dashboard()

with tabs[3]:
    st.header("🚨 Alertas Telegram")
    render_telegram_dashboard()
