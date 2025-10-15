# dashboards/telegram/telegram_layout.py
import os
import pandas as pd
import streamlit as st

def render_telegram_dashboard():
    """
    Dashboard de alertas de Telegram (versión temporal sin PyTorch)
    """

    st.title("📊 Telegram — Alertas de riesgo")

    # -----------------------
    # Ruta al CSV de alertas
    # -----------------------
    CSV_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\alertas.csv"

    # -----------------------
    # Carga del CSV (robusta)
    # -----------------------
    if not os.path.exists(CSV_PATH):
        st.error(f"No se encontró el archivo CSV en: {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        return

    if df.empty:
        st.warning("El archivo CSV está vacío")
        return

    # -----------------------
    # Mostrar vista previa
    # -----------------------
    st.subheader("Vista previa de las alertas cargadas")
    st.dataframe(df.head(10))

    # -----------------------
    # Conteo de niveles de riesgo
    # -----------------------
    if "riesgo" in df.columns:
        st.subheader("Conteo por nivel de riesgo")
        counts = df["riesgo"].value_counts().sort_index()
        st.bar_chart(counts)
    else:
        st.info("No se encontró la columna 'riesgo' en el CSV")

    # -----------------------
    # Lista de alertas
    # -----------------------
    st.subheader("Alertas recientes")
    if "texto" in df.columns:
        st.write(df[["texto", "riesgo"]].head(20))
    else:
        st.info("No se encontró la columna 'texto' en el CSV")


