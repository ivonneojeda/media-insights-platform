# model_utils.py
import pandas as pd
import datetime
import re
import itertools
import collections
import networkx as nx
from prophet import Prophet

# -----------------------------
# Configuración
# -----------------------------
TOP_WORDS = 25
FORECAST_HOURS = 8
RESAMPLE_FREQ = "1H"

stopwords_es = {
    "a","ante","bajo","cabe","con","contra","de","del","desde","durante","en","entre",
    "hacia","hasta","mediante","para","por","según","sin","so","sobre","tras","versus","vía",
    "el","la","los","las","un","una","unos","unas","lo","al","su","sus","mi","mis","tu","tus",
    "nuestro","nuestra","nuestros","nuestras","vosotros","vosotras","vuestro","vuestra","vuestros",
    "vuestras","ellos","ellas","nosotros","nosotras","yo","tú","usted","ustedes","él","ella",
    "me","te","se","nos","os","les","le","y","o","que","qué","como","cómo","para","porque","pero",
    "si","ya","tan","muy","más","menos","también","cuando","donde","dónde","ser","estar","haber"
}

_punct_re = re.compile(r'^[\W_]+|[\W_]+$')
def clean_token(tok: str) -> str:
    return _punct_re.sub("", tok.lower())

# -----------------------------
# Grafo de palabras
# -----------------------------
def generar_grafo_palabras(df: pd.DataFrame, top_n: int = TOP_WORDS):
    if df.empty or "Post" not in df.columns:
        return []

    word_counts = collections.Counter()
    word_sent_map = collections.defaultdict(list)
    posts_tokens = []

    for _, row in df.iterrows():
        text = str(row.get("Post", ""))
        tokens = [clean_token(t) for t in re.split(r"\s+", text) if t and len(t) > 0]
        tokens = [t for t in tokens if t not in stopwords_es and len(t) > 2]
        unique_tokens = list(dict.fromkeys(tokens))
        posts_tokens.append(unique_tokens)
        for t in unique_tokens:
            word_counts[t] += 1
            word_sent_map[t].append(str(row.get("Sentimiento", "")).lower())

    if not word_counts:
        return []

    top_words = [w for w, _ in word_counts.most_common(top_n)]
    G = nx.Graph()
    color_map = {"positivo":"#2ca02c", "positivo.":"#2ca02c", "positive":"#2ca02c",
                 "negativo":"#d62728", "negative":"#d62728",
                 "neutro":"#7f7f7f", "neutral":"#7f7f7f"}

    for w in top_words:
        freq = word_counts[w]
        sents = [s for s in word_sent_map.get(w, []) if s]
        sent_mode = collections.Counter(sents).most_common(1)[0][0] if sents else None
        color = color_map.get(sent_mode, "#7f7f7f")
        size = max(20, 8 + freq * 7)
        G.add_node(w, size=size, color=color, freq=int(freq), sentiment=sent_mode)

    edge_counts = collections.Counter()
    for tokens in posts_tokens:
        present = [t for t in tokens if t in top_words]
        for a, b in itertools.combinations(sorted(set(present)), 2):
            edge_counts[(a,b)] += 1

    for (a,b), w in edge_counts.items():
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b, weight=int(w))

    elements = []
    for node, attrs in G.nodes(data=True):
        elements.append({
            "data": {"id": node, "label": node},
            "style": {"width": attrs["size"], "height": attrs["size"], "background-color": attrs["color"]}
        })
    for source, target, attrs in G.edges(data=True):
        elements.append({"data": {"id": f"e-{source}-{target}", "source": source, "target": target}})
    return elements

# -----------------------------
# Forecast con Prophet
# -----------------------------
def build_forecast_figure(df: pd.DataFrame, hours_ahead: int = FORECAST_HOURS, resample_freq: str = RESAMPLE_FREQ):
    import plotly.graph_objects as go
    fig = go.Figure()
    if df.empty or "Fecha" not in df.columns or "Sentimiento" not in df.columns:
        fig.update_layout(title="Sin datos para pronóstico")
        return fig

    dfc = df.copy()
    dfc["Fecha_parsed"] = pd.to_datetime(dfc["Fecha"], errors="coerce")
    dfc = dfc.dropna(subset=["Fecha_parsed", "Sentimiento"])
    if dfc.empty:
        fig.update_layout(title="Sin datos para pronóstico")
        return fig

    sent_map = {"Positivo": 1, "Negativo": -1, "Neutro": 0,
                "positivo": 1, "negativo": -1, "neutro": 0,
                "positive": 1, "negative": -1, "neutral": 0}
    dfc["y"] = dfc["Sentimiento"].map(sent_map).fillna(0)

    df_hour = dfc.set_index("Fecha_parsed")["y"].resample(resample_freq).mean().fillna(0).reset_index().rename(columns={"Fecha_parsed":"ds"})

    if df_hour.empty or df_hour["y"].nunique() <= 1:
        fig.update_layout(title="No hay suficiente variación de datos para pronóstico")
        return fig

    df_prophet = df_hour[["ds","y"]].copy()
    if pd.api.types.is_datetime64_any_dtype(df_prophet["ds"]):
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)

    try:
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=hours_ahead, freq=resample_freq)
        forecast = model.predict(future)
    except Exception as e:
        fig.update_layout(title=f"Error entrenando Prophet: {e}")
        return fig

    fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="markers", name="Histórico", marker=dict(color="blue", size=6)))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name=f"Pronóstico ({hours_ahead}h)", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill="tonexty", fillcolor="rgba(255,165,0,0.2)", name="Confianza 95%"))

    fig.update_layout(title=f"Pronóstico de sentimiento ({hours_ahead}h)", xaxis_title="Hora", yaxis_title="Sentimiento promedio")
    return fig
