import streamlit as st
import os
import pandas as pd
from dashboards.facebook.facebook_layout import render_facebook_dashboard

# -----------------------
# Depuración de CSV
# -----------------------
CSV_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\sentimiento_2025-09-30_22-00-03.csv"
st.write(f"Buscando CSV en: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    st.error(f"No se encontró el archivo CSV en: {CSV_PATH}")
else:
    df = pd.read_csv(CSV_PATH)
    st.write("Vista previa del CSV:")
    st.dataframe(df.head())

# -----------------------
# Renderizar el dashboard
# -----------------------
render_facebook_dashboard()
