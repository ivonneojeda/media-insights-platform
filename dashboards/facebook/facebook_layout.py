# dashboards/facebook/facebook_layout.py
import os
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ruta CSV (ajusta si quieres más adelante)
CSV_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\sentimiento_2025-09-30_22-00-03.csv"

# -------------------------
# Helpers
# -------------------------
def load_facebook_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"No se encontró el archivo CSV en: {CSV_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    return df

def normalize_and_map_sentiment(df, col_fecha="Fecha", col_sent="Sentimiento"):
    """
    Normaliza nombres y mapea sentimientos a valores numéricos.
    Devuelve df con columnas 'ds' (datetime) y 'y' (float sentiment score) junto al df original.
    """
    df = df.copy()
    # Normalizar nombres de columna (no destructivo)
    # Si viene en minúsculas/otros nombres, intentar reconocerlos
    colmap = {c.lower().strip(): c for c in df.columns}
    # nombres posibles
    fecha_name = None
    sent_name = None
    for k, v in colmap.items():
        if "fecha" in k or "date" in k or "created" in k:
            fecha_name = v
        if "sent" in k or "sentimiento" in k or "sentimiento" in k:
            sent_name = v
    # fallback a parámetros
    if fecha_name is None and col_fecha in df.columns:
        fecha_name = col_fecha
    if sent_name is None and col_sent in df.columns:
        sent_name = col_sent

    if fecha_name is None or sent_name is None:
        return pd.DataFrame(), df  # señalamos ausencia

    # convertir fecha y quitar timezone si existe
    df[fecha_name] = pd.to_datetime(df[fecha_name], errors="coerce", utc=True)
    df[fecha_name] = df[fecha_name].dt.tz_convert("UTC").dt.tz_localize(None)

    # mapear sentimientos a números (soporta english/español y algunas variantes)
    def map_sent(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        mapping = {
            "positive": 1, "positivo": 1, "positiva": 1,
            "neutral": 0, "neutro": 0, "neutralidad": 0,
            "negative": -1, "negativo": -1, "negativa": -1,
            # si ya es numérico
        }
        try:
            return float(s)
        except Exception:
            return mapping.get(s, None)

    df["_sent_score"] = df[sent_name].apply(map_sent)
    return df[[fecha_name, "_sent_score"]].rename(columns={fecha_name: "ds", "_sent_score": "y"}), df

def generate_sentiment_forecast(df, min_points=6, resample_freq="1D", periods=7):
    """
    Construye y entrena Prophet sobre la serie agregada.
    Devuelve tuple (model, forecast_df) o (None, None) si no es posible.
    """
    series_df, original_df = normalize_and_map_sentiment(df)
    if series_df.empty:
        return None, None

    # Drop nans
    series_df = series_df.dropna(subset=["ds", "y"]).copy()
    if series_df.empty:
        return None, None

    # Resample/aggregate: promedio por resample_freq
    series_df = series_df.set_index("ds").sort_index()
    series_agg = series_df["y"].resample(resample_freq).mean().fillna(0).reset_index().rename(columns={"ds": "ds", "y": "y"})

    # Necesitamos suficientes puntos con variación
    if len(series_agg) < min_points or series_agg["y"].nunique() <= 1:
        return None, None

    try:
        m = Prophet()
        m.fit(series_agg)
        future = m.make_future_dataframe(periods=periods, freq=resample_freq)
        forecast = m.predict(future)
        return m, forecast
    except Exception:
        return None, None

def render_wordcloud(df, text_col="Post"):
    if text_col not in df.columns:
        st.info("No se encontró la columna 'Post' para generar la nube de palabras.")
        return
    text = " ".join(df[text_col].dropna().astype(str))
    if not text.strip():
        st.info("No hay texto para generar la nube de palabras.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# -------------------------
# Layout principal
# -------------------------
def render_facebook_dashboard():
    st.title("📘 Dashboard de Facebook - Sentimiento y Nube de Palabras")

    df = load_facebook_data()
    if df.empty:
        st.warning("No hay datos en el CSV de Facebook.")
        return

    # mostrar info básica
    st.write(f"📁 Archivo: {os.path.basename(CSV_PATH)} — Filas: {len(df)}")
    st.write("📄 Columnas:", list(df.columns))

    # vista previa
    st.subheader("Vista previa")
    st.dataframe(df.head(10))

    # Nube de palabras (primero, porque es rápido)
    st.subheader("☁️ Nube de palabras")
    render_wordcloud(df, text_col="Post")

    # Forecast con Prophet
    st.subheader("📈 Pronóstico de sentimiento (Prophet)")
    model, forecast = generate_sentiment_forecast(df, min_points=6, resample_freq="1D", periods=7)
    if model is None or forecast is None:
        st.info("No hay suficientes datos (o suficiente variación) para generar un pronóstico con Prophet.")
    else:
        # mostrar gráfico interactivo con plot_plotly usando el modelo ya entrenado
        try:
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al mostrar el pronóstico: {e}")

    # Distribución simple de sentimiento (si existe columna)
    # intentamos detectar nombre de columna con 'sent' o 'sentimiento'
    col_candidates = [c for c in df.columns if "sent" in c.lower() or "sentimiento" in c.lower()]
    if col_candidates:
        sent_col = col_candidates[0]
        st.subheader("📊 Distribución del campo de sentimiento (raw)")
        st.write(df[sent_col].value_counts().head(20))
    else:
        st.info("No se detectó columna de sentimiento para mostrar su distribución.")

    # Likes si existe
    likes_candidates = [c for c in df.columns if "like" in c.lower()]
    if likes_candidates:
        like_col = likes_candidates[0]
        st.subheader("👍 Likes por publicación (si aplica)")
        if 'Fecha' in df.columns:
            try:
                df_loc = df.copy()
                df_loc['Fecha'] = pd.to_datetime(df_loc['Fecha'], errors='coerce', utc=True).dt.tz_localize(None)
                fig_likes = px.line(df_loc, x='Fecha', y=like_col, title='Likes en el tiempo')
                st.plotly_chart(fig_likes, use_container_width=True)
            except Exception:
                st.write(df[[like_col]].head(10))
        else:
            st.write(df[[like_col]].head(10))



