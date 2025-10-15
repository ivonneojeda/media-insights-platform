# dashboards/telegram/telegram_layout.py

import streamlit as st
import pandas as pd
import os
from datetime import datetime

# -----------------------
# Ruta al CSV de alertas
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "data", "alertas.csv")

# -----------------------
# Función principal del dashboard de Telegram
# -----------------------
def render_telegram_dashboard():
    st.title("Alertas de Ciberseguridad en Tiempo Real")

    # Cargar CSV de forma segura
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            df = pd.DataFrame(columns=["timestamp", "canal", "mensaje", "riesgo"])
    else:
        st.warning(f"No se encontró el archivo CSV en: {CSV_PATH}")
        df = pd.DataFrame(columns=["timestamp", "canal", "mensaje", "riesgo"])

    # Mostrar alertas existentes
    st.subheader("Alertas recibidas")
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("No hay alertas registradas.")

    # Sección para agregar alerta manual
    st.subheader("Agregar alerta manual")
    canal_input = st.text_input("Canal:")
    mensaje_input = st.text_area("Mensaje:")

    if st.button("Agregar alerta"):
        if mensaje_input.strip() != "":
            nueva_alerta = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "canal": canal_input if canal_input else "Manual",
                "mensaje": mensaje_input,
                "riesgo": "Bajo"  # temporal, si no usamos el modelo
            }
            df = pd.concat([df, pd.DataFrame([nueva_alerta])], ignore_index=True)
            try:
                df.to_csv(CSV_PATH, index=False, encoding="utf-8")
                st.success("Alerta agregada correctamente.")
            except Exception as e:
                st.error(f"No se pudo guardar el CSV: {e}")
            st.experimental_rerun()
        else:
            st.warning("Escribe un mensaje para agregar.")

