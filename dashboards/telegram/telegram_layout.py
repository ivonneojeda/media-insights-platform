# dashboards/telegram/telegram_layout.py
import streamlit as st
import pandas as pd
import os
import altair as alt

# ----------------------------
# Configuración CSV
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "data", "alertas.csv")  # ruta relativa

# ----------------------------
# Cargar CSV de forma segura
# ----------------------------
@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    if not os.path.exists(path):
        st.warning(f"No se encontró el archivo CSV en: {path}")
        return pd.DataFrame(columns=["texto", "riesgo"])
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        return df
    except Exception as e:
        st.error(f"Error leyendo CSV ({path}): {e}")
        return pd.DataFrame(columns=["texto", "riesgo"])

df = load_data()

# ----------------------------
# Interfaz Streamlit
# ----------------------------
st.title("Alertas de Ciberseguridad")

# Mostrar dataframe
st.subheader("Alertas registradas")
if df.empty:
    st.info("No hay datos disponibles en el CSV de alertas.")
else:
    st.dataframe(df)

    # ----------------------------
    # Gráfico de conteo de riesgos
    # ----------------------------
    st.subheader("Distribución de nivel de riesgo")
    risk_counts = df['riesgo'].value_counts().reset_index()
    risk_counts.columns = ['riesgo', 'conteo']

    chart = alt.Chart(risk_counts).mark_bar(color="#FF5733").encode(
        x='riesgo:N',
        y='conteo:Q',
        tooltip=['riesgo', 'conteo']
    ).properties(width=600, height=400)

    st.altair_chart(chart, use_container_width=True)
