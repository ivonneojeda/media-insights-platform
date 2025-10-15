# facebook_layout.py
import os
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -----------------------
# CONFIG: ruta relativa al CSV
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "data", "sentimiento_2025-09-30_22-00-03.csv")
CSV_PATH = os.path.abspath(CSV_PATH)

# -----------------------
# CARGA CSV (segura)
# -----------------------
@st.cache_data(ttl=600)
def load_facebook_data(path=CSV_PATH):
    if not os.path.exists(path):
        st.error(f"No se encontró el archivo CSV en: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        return df
    except Exception as e:
        st.error(f"Error leyendo CSV ({path}): {e}")
        return pd.DataFrame()

# -----------------------
# RENDER DASHBOARD
# -----------------------
def render_facebook_dashboard():
    st.title("Dashboard de Facebook")
    
    df = load_facebook_data()
    if df.empty:
        st.warning("No hay datos disponibles en el CSV de Facebook.")
        return

    # Mostrar vista previa
    st.subheader("Vista previa del CSV")
    st.dataframe(df.head())

    # -----------------------
    # Analizar sentimiento
    # -----------------------
    if 'Sentimiento' in df.columns:
        st.subheader("Distribución de sentimiento")
        sentiment_counts = df['Sentimiento'].value_counts()
        st.bar_chart(sentiment_counts)
    else:
        st.info("La columna 'Sentimiento' no se encuentra en el CSV.")

    # -----------------------
    # Nube de palabras
    # -----------------------
    if 'Post' in df.columns:
        st.subheader("Nube de palabras")
        text = " ".join(df['Post'].dropna().astype(str))
        if text.strip() == "":
            st.info("No hay texto disponible para generar la nube de palabras.")
        else:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    else:
        st.info("La columna 'Post' no se encuentra en el CSV.")
