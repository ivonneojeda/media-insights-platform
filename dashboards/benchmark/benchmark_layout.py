# dashboards/benchmark/benchmark_layout.py
import os
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st

# -----------------------
# CONFIG: carpeta de datos y archivos
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "..", "..", "data")
FILES = [
    "ipn_social_growth_jul_sep_2025.csv",
    "tec_social_growth_jul_sep_2025.csv",
    "uam_social_growth_jul_sep_2025.csv",
    "udg_social_growth_jul_sep_2025.csv",
    "unam_social_growth_jul_sep_2025.csv"
]

# -----------------------
# UTIL: cargar todos los CSV
# -----------------------
@st.cache_data(ttl=600)
def load_data():
    data_dict = {}
    for file in FILES:
        file_path = os.path.join(DATA_FOLDER, file)
        if not os.path.exists(file_path):
            st.warning(f"⚠️ No se encontró el archivo: {file}")
            continue
        uni_name = os.path.basename(file).split("_")[0].upper()
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        data_dict[uni_name] = df
    return data_dict

# -----------------------
# UTIL: forecast con Prophet
# -----------------------
def forecast(df, column):
    prophet_df = df.rename(columns={'date':'ds', column:'y'})
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
def render_benchmark_dashboard():
    st.title("📈 Benchmark Institucional: Comparativo de Redes Sociales")

    metric = st.selectbox("Selecciona métrica", ["followers", "likes"])

    data_dict = load_data()
    if not data_dict:
        st.error("No se cargaron datos de ninguna universidad. Verifica la carpeta de CSV.")
        return

    fig = go.Figure()
    for uni, df in data_dict.items():
        if metric not in df.columns:
            st.warning(f"⚠️ La columna '{metric}' no existe en {uni}")
            continue
        forecast_df = forecast(df, metric)
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[metric],
            mode='lines+markers',
            name=f"{uni} real"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['yhat'],
            mode='lines',
            name=f"{uni} proyección",
            line=dict(dash='dot')
        ))

    fig.update_layout(
        title=f"Comparativo de {metric.capitalize()} con proyección 30 días",
        xaxis_title="Fecha",
        yaxis_title=metric.capitalize(),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)



