# model_utils.py
import pandas as pd
from prophet import Prophet
import requests

# -----------------------------
# Prophet
# -----------------------------
def train_prophet(df, date_col='created_at', target_col='sentiment_numeric', freq='H'):
    """
    Entrena un modelo Prophet usando un DataFrame con columna de fecha y columna de sentimiento.
    Devuelve el modelo entrenado y el DataFrame agregado por hora.
    """
    if df.empty or date_col not in df.columns or target_col not in df.columns:
        return None, pd.DataFrame()
    
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce').dt.tz_localize(None)
    df_copy = df_copy.dropna(subset=[date_col, target_col])
    if df_copy.empty:
        return None, pd.DataFrame()
    
    # Agrupar por frecuencia
    df_hourly = df_copy.groupby(df_copy[date_col].dt.floor(freq))[target_col].mean().reset_index()
    df_hourly.rename(columns={date_col:'ds', target_col:'y'}, inplace=True)
    
    # Entrenar Prophet
    model = Prophet()
    model.fit(df_hourly)
    
    return model, df_hourly

def forecast_prophet(model, periods=8, freq='H'):
    """
    Genera un pronóstico a partir del modelo Prophet entrenado.
    Devuelve un DataFrame con la predicción.
    """
    if not model:
        return pd.DataFrame()
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

# -----------------------------
# API X
# -----------------------------
def fetch_x_data(url, bearer_token, params=None):
    """
    Llama a la API de X con bearer token y devuelve un DataFrame.
    """
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Error en API X: {response.status_code} - {response.text}")
    
    data = response.json()
    return pd.DataFrame(data)
