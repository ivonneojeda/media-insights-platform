import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st
import os

DATA_FOLDER = r"C:\Users\ivonn\Desktop\Proyecto_benchmark\datos"
FILES = [
    "ipn_social_growth_jul_sep_2025.csv",
    "tec_social_growth_jul_sep_2025.csv",
    "uam_social_growth_jul_sep_2025.csv",
    "udg_social_growth_jul_sep_2025.csv",
    "unam_social_growth_jul_sep_2025.csv"
]

def load_data():
    data_dict = {}
    for file in FILES:
        uni_name = os.path.basename(file).split("_")[0].upper()
        df = pd.read_csv(os.path.join(DATA_FOLDER, file))
        df['date'] = pd.to_datetime(df['date'])
        data_dict[uni_name] = df
    return data_dict

def forecast(df, column):
    prophet_df = df.rename(columns={'date':'ds', column:'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast_df = model.predict(future)
    return forecast_df[['ds', 'yhat']]

def render_benchmark_dashboard():
    st.header("Comparativo de crecimiento de redes sociales")
    
    metric = st.selectbox("Selecciona métrica", ["followers", "likes"])
    
    data_dict = load_data()
    
    fig = go.Figure()
    
    for uni, df in data_dict.items():
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

