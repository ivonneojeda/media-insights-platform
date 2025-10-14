# dashboards/telegram/telegram_layout.py
import streamlit as st
import pandas as pd
from datetime import datetime
from dashboards.telegram.model_utils import predict_risk
import os

# --- ELIMINAR cualquier st.set_page_config() ---
# --- Ajustar ruta CSV a la carpeta 'data' del proyecto ---
CSV_PATH = os.path.join("data", "alertas.csv")

# --- Cargar o inicializar el DataFrame ---
if "df" not in st.session_state:
    if os.path.exists(CSV_PATH):
        st.session_state.df = pd.read_csv(CSV_PATH)
    else:
        st.session_state.df = pd.DataFrame(columns=["timestamp", "canal", "mensaje", "riesgo"])

def render_telegram_dashboard():
    st.title("🚨 Alertas de Ciberseguridad en Tiempo Real")

    # Mostrar alertas existentes
    st.subheader("Alertas recibidas")
    st.dataframe(st.session_state.df.sort_values("timestamp", ascending=False))

    # Agregar alerta manual
    st.subheader("Agregar alerta manual")
    canal_input = st.text_input("Canal:")
    mensaje_input = st.text_area("Mensaje:")

    if st.button("Agregar alerta"):
        if mensaje_input.strip() != "":
            try:
                riesgo_predicho, _ = predict_risk(mensaje_input)
            except Exception:
                riesgo_predicho = "Desconocido"

            nueva_alerta = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "canal": canal_input if canal_input else "Manual",
                "mensaje": mensaje_input,
                "riesgo": riesgo_predicho
            }

            st.session_state.df = pd.concat(
                [st.session_state.df, pd.DataFrame([nueva_alerta])], ignore_index=True
            )

            # Guardar CSV
            st.session_state.df.to_csv(CSV_PATH, index=False)
            st.experimental_rerun()
        else:
            st.warning("Escribe un mensaje para agregar.")

