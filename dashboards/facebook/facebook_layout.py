import os
import pandas as pd
import streamlit as st
from wordcloud import STOPWORDS, WordCloud
import plotly.express as px
import plotly.graph_objects as go
import spacy
import numpy as np
import random

# -----------------------
# CONFIG: ruta relativa al CSV
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "data", "sentimiento_2025-09-30_22-00-03.csv")
CSV_PATH = os.path.abspath(CSV_PATH)

# -----------------------
# SPAcy modelo español
# -----------------------
nlp = spacy.load("es_core_news_sm")

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
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            if pd.api.types.is_datetime64tz_dtype(df['Fecha']):
                df['Fecha'] = df['Fecha'].dt.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error leyendo CSV ({path}): {e}")
        return pd.DataFrame()

# -----------------------
# GENERAR WORDCLOUD INTERACTIVA
# -----------------------
def generate_interactive_wordcloud(text):
    doc = nlp(text)
    words_filtered = [token.text for token in doc if token.pos_ in ("NOUN","PROPN","ADJ")]
    if not words_filtered:
        return None
    wc = WordCloud(width=800, height=400, stopwords=set(STOPWORDS)).generate(" ".join(words_filtered))
    positions = []
    for word, freq in wc.words_.items():
        positions.append({
            'word': word,
            'freq': freq,
            'x': random.uniform(0,1),
            'y': random.uniform(0,1),
            'size': freq*100
        })
    df_wc = pd.DataFrame(positions)
    return df_wc

# -----------------------
# RENDER FACEBOOK DASHBOARD
# -----------------------
def render_facebook_dashboard():
    st.title("Dashboard de Facebook")

    df = load_facebook_data()
    if df.empty:
        st.info("No hay datos disponibles en el CSV de Facebook.")
        return

    st.write("Filas cargadas:", len(df))

    # -----------------------
    # FILTROS EN SIDEBAR
    # -----------------------
    st.sidebar.header("Filtros Facebook")
    if 'Fecha' in df.columns:
        min_date, max_date = df['Fecha'].min(), df['Fecha'].max()
        selected_dates = st.sidebar.date_input("Rango de fechas", [min_date, max_date])
        if len(selected_dates) == 2:
            start, end = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
            df = df[(df['Fecha'] >= start) & (df['Fecha'] <= end)]

    if 'Sentimiento' in df.columns:
        sentimientos = df['Sentimiento'].dropna().unique()
        selected_sentiments = st.sidebar.multiselect(
            "Filtrar sentimiento",
            sentimientos,
            default=list(sentimientos)
        )
        df = df[df['Sentimiento'].isin(selected_sentiments)]

    # -----------------------
    # COLUMNAS
    # -----------------------
    col1, col2 = st.columns(2)

    # NUBE DE PALABRAS
    with col1:
        if 'Post' in df.columns:
            st.subheader("Nube de palabras")
            text = " ".join(df['Post'].dropna().astype(str))

            # Filtrar sustantivos, nombres propios y adjetivos
            doc = nlp(text)
            words_filtered = [token.text for token in doc if token.pos_ in ("NOUN","PROPN","ADJ")]
            if not words_filtered:
                st.info("No hay palabras válidas para mostrar en la nube de palabras.")
            else:
                wc = WordCloud(width=800, height=400, background_color='white', stopwords=set(STOPWORDS)) \
                    .generate(" ".join(words_filtered))

                # Renderizar con matplotlib
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12,6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        else:
            st.info("La columna 'Post' no se encuentra en el CSV.")


    # PASTEL DE SENTIMIENTOS
    with col2:
        if 'Sentimiento' in df.columns:
            st.subheader("Distribución de sentimiento")
            sentiment_counts = df['Sentimiento'].value_counts()
            if sentiment_counts.empty:
                st.info("No hay datos de sentimiento para mostrar.")
            else:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hovertemplate='%{label}: %{value} posts (%{percent})<extra></extra>'
                )])
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("La columna 'Sentimiento' no se encuentra en el CSV.")
