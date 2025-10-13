# dashboards/x_predictivo/x_layout.py
import os
import tempfile
import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

# -----------------------
# CONFIG (ruta confirmada)
# -----------------------
CSV_PATH = r"C:\Users\ivonn\Desktop\dashboard_maestro\data\Conversación sobre UNAM 5-7oct25 - ISO.csv"

# -----------------------
# UTIL: cargar datos (cacheado)
# -----------------------
@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {path}")
    # lectura robusta
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    return df

# -----------------------
# UTIL: preparar serie para Prophet
# -----------------------
def prepare_prophet_series(df, date_col="created_at", sentiment_col="sentiment", resample_freq="1H"):
    # Convertir fecha y normalizar
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[date_col, sentiment_col]).copy()
    if df.empty:
        return pd.DataFrame()

    # Mapear sentimiento a numérico si es texto
    sent_map = {
        "Positivo": 1, "PositivoNeutroPositiva": 0.5, "Positiva": 1,
        "Negativo": -1, "NegativoNeutroNegativa": -0.5, "Negativa": -1,
        "Neutro": 0, "Neutral": 0
    }
    # Si ya es numérico no cambia
    def map_sent(x):
        try:
            return float(x)
        except Exception:
            return sent_map.get(str(x).strip(), 0.0)

    df["_sent_score"] = df[sentiment_col].map(map_sent).astype(float)

    # Resample por frecuencia (promedio)
    df = df.set_index(date_col).sort_index()
    df_hour = df["_sent_score"].resample(resample_freq).mean().fillna(0).reset_index()
    df_hour = df_hour.rename(columns={date_col: "ds", "_sent_score": "y"})
    return df_hour

# -----------------------
# PROPHE T: generar forecast (con chequeos)
# -----------------------
def build_prophet_forecast(df_hour, periods=8, freq="H"):
    fig = go.Figure()
    if df_hour.empty or df_hour["y"].nunique() <= 1 or len(df_hour) < 6:
        # No hay suficiente info
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

    # Construir figura: histórico + pronóstico + banda
    fig.add_trace(go.Scatter(x=df_hour["ds"], y=df_hour["y"], mode="markers", name="Histórico"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Pronóstico"))
    if "yhat_upper" in forecast and "yhat_lower" in forecast:
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill="tonexty", fillcolor="rgba(255,165,0,0.2)", name="Intervalo 95%"))

    fig.update_layout(title=f"Pronóstico de sentimiento ({periods}h)", xaxis_title="Fecha", yaxis_title="Sentimiento promedio")
    return fig, model

# -----------------------
# GRAFO: construir coocurrencias desde 'keywords' o 'hashtags' o desde 'text'
# -----------------------
def build_hashtag_graph(df, keywords_col="keywords", text_col="text"):
    G = nx.Graph()
    # Preferir columna keywords si existe y no vacía
    if keywords_col in df.columns and df[keywords_col].notna().any():
        for row in df[keywords_col].dropna():
            # Asumimos separador por comas o espacios; normalizamos
            if isinstance(row, str):
                parts = [p.strip() for p in (row.split(",") if "," in row else row.split()) if p.strip()]
                tags = [p.lstrip("#").lower() for p in parts if p]
            else:
                tags = []
            for i, a in enumerate(tags):
                G.add_node(a, size=1)
                for b in tags[i+1:]:
                    if G.has_edge(a, b):
                        G[a][b]['weight'] += 1
                    else:
                        G.add_edge(a, b, weight=1)
    else:
        # Fallback: extraer hashtags desde text
        if text_col in df.columns:
            for text in df[text_col].dropna():
                tags = [w.lstrip("#").lower() for w in str(text).split() if w.startswith("#")]
                for i, a in enumerate(tags):
                    G.add_node(a, size=1)
                    for b in tags[i+1:]:
                        if G.has_edge(a, b):
                            G[a][b]['weight'] += 1
                        else:
                            G.add_edge(a, b, weight=1)
    return G

# -----------------------
# Mostrar grafo con Pyvis (guardando temporal y mostrando html)
# -----------------------
def render_pyvis_graph(G, height=600):
    if G.number_of_nodes() == 0:
        st.info("No se encontraron hashtags / coocurrencias para graficar.")
        return

    net = Network(height=f"{height}px", width="100%", bgcolor="#0E1117", font_color="white", notebook=False)
    # Ajustes estéticos
    for n, d in G.nodes(data=True):
        # tamaño por degree (o por atributo size si existe)
        size = max(8, int(G.degree(n) * 4))
        net.add_node(n, label=n, title=f"{n} (degree {G.degree(n)})", size=size)
    for u, v, attrs in G.edges(data=True):
        weight = attrs.get("weight", 1)
        net.add_edge(u, v, value=weight, title=f"weight: {weight}")
    net.force_atlas_2based()
    # Guardar temporal
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tmp_path = tmp.name
    tmp.close()
    net.save_graph(tmp_path)
    # Mostrar en streamlit
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height, scrolling=True)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

# -----------------------
# LAYOUT principal
# -----------------------
def render_x_dashboard():
    st.title("📊 X — Análisis predictivo de sentimiento y hashtags")

    st.markdown(f"📁 Ruta esperada del CSV:<br> `{CSV_PATH}`", unsafe_allow_html=True)

    # Intentar cargar el CSV
    try:
        df = load_data()
        st.success(f"✅ Archivo cargado correctamente. Filas: {len(df)}")
        st.write("Columnas:", list(df.columns))
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Error al leer CSV: {e}")
        return

    # Mostrar primeras filas
    st.subheader("Vista previa de datos")
    st.dataframe(df.head(10))

    # Confirmar columnas clave
    missing_cols = [c for c in ["created_at", "sentiment", "text", "keywords"] if c not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Faltan las columnas requeridas: {missing_cols}")
    else:
        st.info("✅ Todas las columnas requeridas están presentes.")

    # Ahora seguir con el flujo normal
    st.header("Pronóstico de sentimiento (Prophet)")
    df_hour = prepare_prophet_series(df, date_col="created_at", sentiment_col="sentiment", resample_freq="1H")
    st.write("Datos procesados para Prophet:", df_hour.head())

    fig_forecast, model = build_prophet_forecast(df_hour, periods=8, freq="H")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.header("Grafo de hashtags")
    G = build_hashtag_graph(df, keywords_col="keywords", text_col="text")
    st.write(f"Nodos del grafo: {len(G.nodes())}, Enlaces: {len(G.edges())}")
    render_pyvis_graph(G, height=600)

    # ----------------- Estadísticas -----------------
    st.header("Top hashtags")
    # contar frecuencia simple
    counts = {}
    if "keywords" in df.columns and df["keywords"].notna().any():
        iterable = df["keywords"].dropna()
        for row in iterable:
            if isinstance(row, str):
                parts = [p.strip() for p in (row.split(",") if "," in row else row.split()) if p.strip()]
                for p in parts:
                    tag = p.lstrip("#").lower()
                    counts[tag] = counts.get(tag, 0) + 1
    else:
        # fallback extract from text
        for text in df["text"].dropna():
            for w in str(text).split():
                if w.startswith("#"):
                    tag = w.lstrip("#").lower()
                    counts[tag] = counts.get(tag, 0) + 1

    if counts:
        top = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True), columns=["hashtag", "count"])
        st.table(top.head(20))
    else:
        st.info("No se encontraron hashtags para listar.")
