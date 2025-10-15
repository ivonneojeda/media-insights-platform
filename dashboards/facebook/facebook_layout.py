import os
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==========================
# CONFIGURACIÓN DEL ARCHIVO
# ==========================
CSV_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\sentimiento_2025-09-30_22-00-03.csv"

# ==========================
# CARGA Y LIMPIEZA DE DATOS
# ==========================
@st.cache_data
def load_facebook_data():
    """Carga el CSV de Facebook, verifica su existencia y renombra columnas para compatibilidad."""
    if not os.path.exists(CSV_PATH):
        st.error(f"No se encontró el archivo CSV en: {CSV_PATH}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")

        # Renombrar columnas a las que espera el dashboard
        df = df.rename(columns={
            "Fecha": "created_at",
            "Post": "text",
            "Likes": "likes",
            "Sentimiento": "sentiment"
        })

        # Asegurar que las columnas clave existan
        required_cols = ["created_at", "text", "likes", "sentiment"]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Falta la columna obligatoria: {col}")
                return pd.DataFrame()

        # Convertir fechas al formato datetime
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        # Eliminar filas sin texto o sentimiento
        df = df.dropna(subset=["text", "sentiment"])

        if df.empty:
            st.warning("El CSV se cargó correctamente, pero no contiene datos válidos.")
        else:
            st.success(f"Se cargaron {len(df)} registros de Facebook correctamente.")

        return df

    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

# ==========================
# VISUALIZACIONES
# ==========================
def render_wordcloud(df):
    """Genera una nube de palabras a partir de los posts."""
    text = " ".join(df["text"].astype(str))
    wordcloud = WordCloud(width=900, height=500, background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def render_sentiment_summary(df):
    """Muestra un conteo de sentimientos."""
    st.subheader("Distribución del Sentimiento")
    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

# ==========================
# INTERFAZ PRINCIPAL
# ==========================
def render_facebook_dashboard():
    st.title("📘 Dashboard de Facebook")

    df = load_facebook_data()
    if df.empty:
        st.warning("No hay datos disponibles para mostrar.")
        return

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Nube de palabras
    st.subheader("Nube de Palabras")
    render_wordcloud(df)

    # Resumen de sentimiento
    render_sentiment_summary(df)
