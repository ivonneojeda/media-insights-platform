# dashboards/telegram/telegram_layout.py
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet

# -----------------------
# CONFIG: ruta relativa al CSV
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "data", "telegram_alerts.csv")

# -----------------------
# UTIL: cargar datos (cacheado)
# -----------------------
@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    if not os.path.exists(path):
        st.error(f"No se encontró el archivo CSV: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    return df

# -----------------------
# UTIL: forecast con Prophet
# -----------------------
def forecast(df, column):
    if column not in df.columns:
        st.warning(f"⚠️ Columna {column} no existe en el CSV")
        return pd.DataFrame()
    prophet_df = df.rename(columns={'created_at':'ds', column:'y'})
    if prophet_df['y'].nunique() <= 1:
        return prophet_df.assign(yhat=prophet_df['y'])
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast_df = model.predict(future)
    return forecast_df[['ds', 'yhat']]

# -----------------------
# LAYOUT principal
# -----------------------
def render_telegram_dashboard():
    st.title("💬 Telegram Alerts — Análisis de actividad")

    df = load_data()
    if df.empty:
        st.info("No hay datos para mostrar")
        return

    st.subheader("Vista previa de datos")
    st.dataframe(df.head(10))

    metric = st.selectbox("Selecciona métrica", ["likes_count", "retweets_count", "followers_count"])
    forecast_df = forecast(df, metric)

    fig = px.line(df, x='created_at', y=metric, markers=True, title=f"{metric} histórico")
    st.plotly_chart(fig, use_container_width=True)

    if not forecast_df.empty:
        fig2 = px.line(forecast_df, x='ds', y='yhat', title=f"{metric} pronóstico 30 días")
        st.plotly_chart(fig2, use_container_width=True)
