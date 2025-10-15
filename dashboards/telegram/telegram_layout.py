# dashboards/telegram/telegram_layout.py
import os
import pandas as pd
import streamlit as st

# -----------------------
# CONFIG: ruta al CSV de alertas de Telegram
# -----------------------
CSV_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\alertas.csv"

# -----------------------
# CARGA CSV
# -----------------------
@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    if not os.path.exists(path):
        st.error(f"No se encontró el archivo CSV en: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        return df
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        return pd.DataFrame()

# -----------------------
# LAYOUT principal
# -----------------------
def render_telegram_dashboard():
    st.title("📊 Telegram — Alertas de riesgo")
    st.markdown(f"📁 Ruta esperada del CSV:<br> `{CSV_PATH}`", unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.info("No hay datos disponibles en el CSV de alertas.")
        return

    st.subheader("Vista previa de alertas")
    st.dataframe(df.head(10))

    # Estadísticas simples por nivel de riesgo
    if "riesgo" in df.columns:
        counts = df["riesgo"].value_counts().sort_index()
        st.subheader("Resumen de niveles de riesgo")
        st.bar_chart(counts)
    else:
        st.warning("⚠️ La columna 'riesgo' no existe en el CSV.")



