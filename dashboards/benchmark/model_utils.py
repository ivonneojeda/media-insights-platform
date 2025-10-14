# dashboards/benchmark/model_utils.py
import pandas as pd
from prophet import Prophet

def forecast(df: pd.DataFrame, column: str, periods: int = 30) -> pd.DataFrame:
    """
    Genera predicciones con Prophet para la columna especificada.
    Si no hay suficiente variación, devuelve la serie original.
    """
    prophet_df = df.rename(columns={'date': 'ds', column: 'y'})
    if prophet_df['y'].nunique() <= 1:
        return prophet_df.assign(yhat=prophet_df['y'])
    
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast_df = model.predict(future)
    return forecast_df[['ds', 'yhat']]


