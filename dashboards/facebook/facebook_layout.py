# dashboards/facebook/facebook_layout.py
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet

# -----------------------
# CONFIG: ruta al CSV de Facebook
# -----------------------
CSV_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\sentimiento_2025-09-30_22-00-03.csv"

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
        # Mapear columnas reales a las que el código espera
        df = df.rename(columns={"Fecha":"created_at","Sentimiento":"sentiment"})
        return df
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        return pd.DataFrame()

# -----------------------
# Preparar serie para Prophet
# -----------------------
def prepare_prophet_series(df, date_col="created_at", sentiment_col="sentiment", resample_freq="1H"):
    df = df.copy()
    if date_col not in df.columns or sentiment_col not in df.columns:
        return pd.DataFrame()
    
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[date_col, sentiment_col])
    if df.empty:
        return pd.DataFrame()
    
    sent_map = {"Positivo":1,"Negativo":-1,"Neutro":0,"positive":1,"negative":-1,"neutral":0}
    def map_sent(x):
        try:
            return float(x)
        except Exception:
            return sent_map.get(str(x).strip(), 0.0)
    df["_sent_score"] = df[sentiment_col].map(map_sent).astype(float)
    
    df = df.set_index(date_col).sort_index()
    df_hour = df["_sent_score"].resample(resample_freq).mean().fillna(0).reset_index()
    df_hour = df_hour.rename(columns={date_col:"ds","_sent_score":"y"})
    return df_hour

# -----------------------
# Generar forecast Prophet
# -----------------------
def build_prophet_forecast(df_hour, periods=8, freq="H"):
    fig = go.Figure()
    if df_hour.empty or df_hour["y"].nunique() <= 1 or len(df_hour) < 6:
        fig.update_layout(title="No hay datos suficientes para pronóstico con Prophet")
        return fig, None
    try:
        model = Prophet()
        model.fit(df_hour)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
    except Exception as e:
        fig.update_layout(title=f"Error entrenando Prophet: {e}")
        return fig, None

    fig.add_trace(go.Scatter(x=df_hour["ds"], y=df_hour["y"], mode="markers", name="Histórico"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Pronóstico"))
    if "yhat_upper" in forecast and "yhat_lower" in forecast:
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill="tonexty", fillcolor="rgba(255,165,0,0.2)", name="Intervalo 95%"))

    fig.update_layout(title=f"Pronóstico de sentimiento ({periods}h)", xaxis_title="Fecha", yaxis_title="Sentimiento promedio")
    return fig, model

# -----------------------
# LAYOUT principal
# -----------------------
def render_facebook_dashboard():
    st.title("📊 Facebook — Análisis de sentimiento")
    st.markdown(f"📁 Ruta esperada del CSV:<br> `{CSV_PATH}`", unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.info("No hay datos disponibles en el CSV de Facebook.")
        return

    st.subheader("Vista previa de datos")
    st.dataframe(df.head(10))

    df_hour = prepare_prophet_series(df)
    st.write("Datos procesados para Prophet:", df_hour.head())

    fig_forecast, model = build_prophet_forecast(df_hour)
    st.plotly_chart(fig_forecast, use_container_width=True)

