# dashboards/x_predictivo/x_layout.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import ast
import re
import tempfile
import math
import json
import networkx as nx
from pyvis.network import Network
import plotly.express as px

# Algunas instalaciones usan "community" o "python-louvain"
try:
    import community as community_louvain
except Exception:
    community_louvain = None

# ------------------------
# CONFIG
# ------------------------
CSV_PATH = "data/Conversacion_UNAM_limpio.csv"
DATE_COL = "created_at"
SENTIMENT_COL = "sentiment"
FIG_BG = '#262730'
MODERN_COLOR_SCALE = ["#FF6347", "#D3D3D3", "#4CAF50"]
COLOR_NEG = "#FF6347"
COLOR_NEUTRO = "#D3D3D3"
COLOR_POS = "#4CAF50"

# ------------------------
# DATA LOAD & PREP
# ------------------------
@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        return df
    except Exception as e:
        st.error(f"Error cargando CSV: {e}")
        return pd.DataFrame()

def prepare_data(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Fecha
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df.get(DATE_COL), errors="coerce")
        df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    else:
        df[DATE_COL] = pd.NaT

    # Columnas de entidades: hashtags, keywords, mentions
    for col in ["hashtags", "keywords", "mentions"]:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        else:
            df[col] = df[col].apply(lambda x: safe_list(x))

    # date_str / hour_str para heatmap
    try:
        df["date_str"] = df[DATE_COL].dt.strftime("%b %d")
        df["hour_str"] = df[DATE_COL].dt.strftime("%H:00")
    except Exception:
        df["date_str"] = ""
        df["hour_str"] = ""

    return df

def safe_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if x in ["", "[]"]:
            return []
        try:
            return ast.literal_eval(x)
        except:
            return [x]
    return []
# dashboards/x_predictivo/x_layout.py (añadir o reemplazar render_interactive_graph)
import math
import json
import networkx as nx
from pyvis.network import Network
import plotly.express as px

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

FIG_BG = '#262730'

def render_interactive_graph(df_historico, selected_layers):
    """
    Construye el HTML del grafo y devuelve (html, G_filtered)
    - selected_layers: lista con alguno de ['hashtags','mentions','keywords']
    """

    if df_historico is None or df_historico.empty:
        return "Error: DataFrame vacío o no proporcionado.", nx.Graph()

    df_historico = df_historico.copy()
    df_historico.columns = [c.lower().strip() for c in df_historico.columns]

    # Construir grafo
    G = nx.Graph()
    for _, row in df_historico.iterrows():
        row_entities = []
        if 'hashtags' in selected_layers and 'hashtags' in row:
            row_entities.extend([h.lower() for h in row.get('hashtags', []) if isinstance(h, str) and h.strip()])
        if 'mentions' in selected_layers and 'mentions' in row:
            row_entities.extend([m.lower() for m in row.get('mentions', []) if isinstance(m, str) and m.strip()])
        if 'keywords' in selected_layers and 'keywords' in row:
            row_entities.extend([k.lower() for k in row.get('keywords', []) if isinstance(k, str) and k.strip()])

        # Crear aristas entre todas las entidades de la misma fila
        for i, a in enumerate(row_entities):
            if not G.has_node(a):
                G.add_node(a)
            for b in row_entities[i+1:]:
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)

    if G.number_of_nodes() == 0:
        return "Error: No hay nodos para mostrar con la selección actual.", nx.Graph()

    # Clustering con Louvain
    partition = {}
    if community_louvain is not None and G.number_of_edges() > 0:
        try:
            partition = community_louvain.best_partition(G, weight='weight')
        except Exception:
            partition = {n: 0 for n in G.nodes()}
    else:
        partition = {n: 0 for n in G.nodes()}

    # Asignar colores por cluster
    base_colors = px.colors.qualitative.Dark24
    for node in G.nodes():
        cluster_id = partition.get(node, 0)
        G.nodes[node]['cluster_id'] = cluster_id
        G.nodes[node]['color'] = base_colors[cluster_id % len(base_colors)]

    # Crear visualización PyVis
    net = Network(height="700px", width="100%", bgcolor=FIG_BG, font_color="white", notebook=False)
    for n, data in G.nodes(data=True):
        deg = G.degree(n)
        size = max(12, min(int(math.log(deg + 1) * 20), 60))
        net.add_node(
            n,
            label=n,
            title=f"{n} (Grado: {deg}, Cluster: {data.get('cluster_id', 'n/a')})",
            size=size,
            color=data.get('color', '#888888'),
            font={'size': 14, 'face': 'Arial', 'color': 'white'}
        )

    for u, v, attrs in G.edges(data=True):
        net.add_edge(u, v, value=attrs.get('weight', 1), title=f"Co-ocurrencia: {attrs.get('weight', 1)}")

    net.toggle_physics(True)

    # Opciones de PyVis
    options_dict = {
        "nodes": {"font": {"size": 14}, "scaling": {"min": 10, "max": 60}},
        "edges": {"color": {"inherit": True}, "smooth": {"enabled": True, "type": "dynamic"}},
        "physics": {"enabled": True, "solver": "forceAtlas2Based"},
        "interaction": {"hover": True, "zoomView": True, "dragNodes": True}
    }
    net.set_options(json.dumps(options_dict))

    # Generar HTML
    try:
        html = net.generate_html()
        html = html.replace("\ufeff", "").replace("\x00", "")
    except Exception as e:
        return f"Error al generar el grafo: {e}", nx.Graph()

    return html, G


# ------------------------
# HEATMAP
# ------------------------
def render_heatmap(df):
    if df.empty or not all(c in df.columns for c in ["sentiment", "date_str", "hour_str"]):
        st.warning("Datos inválidos o faltantes para el Heatmap.")
        return

    # Mapear sentimiento a numérico solo para cálculo, sin tocar la columna original
    SENTIMENT_MAP = {"Positivo": 1.0, "Negativo": -1.0, "Neutro": 0.0}
    df["_sentiment_num"] = df["sentiment"].map(SENTIMENT_MAP)

    pivot = df.pivot_table(
        index="hour_str",
        columns="date_str",
        values="_sentiment_num",
        aggfunc="mean"
    ).fillna(0).T

    hour_order = [f"{h:02d}:00" for h in range(24)]
    date_order = sorted(df["date_str"].unique(), key=lambda x: pd.to_datetime(x, format="%b %d"))
    full_index = pd.MultiIndex.from_product([date_order, hour_order], names=['date_str', 'hour_str'])
    df_plot = pivot.stack().reindex(full_index).fillna(0).reset_index(name='sentiment_mean')

    fig = px.density_heatmap(
        df_plot,
        x="date_str",
        y="hour_str",
        z="sentiment_mean",
        color_continuous_scale=MODERN_COLOR_SCALE,
        category_orders={"hour_str": hour_order, "date_str": date_order}
    )
    fig.update_layout(
        title=dict(text="Distribución de Sentimiento Promedio por Hora (Histórico)", x=0.0, xanchor='left'),
        height=600,
        xaxis=dict(tickangle=-45),
        plot_bgcolor=FIG_BG,
        paper_bgcolor=FIG_BG,
        font=dict(color="white")
    )
    fig.update_traces(xgap=1, ygap=1)

    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# GAUGE
# ------------------------
def render_gauge(df):
    if df.empty or "sentiment" not in df.columns:
        st.warning("Datos inválidos para el Gauge.")
        return

    SENTIMENT_MAP = {"Positivo": 1.0, "Negativo": -1.0, "Neutro": 0.0}
    last_pred = df["sentiment"].map(SENTIMENT_MAP).mean()

    if last_pred >= 0.33:
        INDICATOR_COLOR, sentiment_label = COLOR_POS, "POSITIVO"
    elif last_pred <= -0.33:
        INDICATOR_COLOR, sentiment_label = COLOR_NEG, "NEGATIVO"
    else:
        INDICATOR_COLOR, sentiment_label = COLOR_NEUTRO, "NEUTRO"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=np.clip(last_pred, -1, 1),
        number={"valueformat": ".2f", "font": dict(color='white', size=32)},
        gauge={"axis": {"range": [-1, 1], "showticklabels": False}, "bar": {"color": INDICATOR_COLOR}}
    ))
    fig_gauge.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=FIG_BG, font=dict(color="white"))
    fig_gauge.add_annotation(text=sentiment_label, x=0.5, y=0.25, showarrow=False, font=dict(size=18, color="white"))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------
# GRAFO: BUILD + LOUVAIN + PYVIS (una capa a la vez, min_degree fijo a 1)
# ------------------------
def render_x_dashboard():
    st.title("Análisis de Sentimiento y Conversación de X")
    st.markdown("---")

    # Cargar y preparar datos
    df_raw = load_data()
    if df_raw.empty:
        st.info("No hay datos cargados.")
        return

    df_historico = prepare_data(df_raw)
    if df_historico.empty:
        st.info("No hay datos válidos después de la limpieza.")
        return

    # Visualizaciones superiores: heatmap + gauge
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Histórico de Sentimiento")
        render_heatmap(df_historico)
    with col2:
        st.subheader("Pronóstico (Promedio de Sentimiento)")
        render_gauge(df_historico)

    st.markdown("---")

    # Grafo - multiselect por capas (hashtags, mentions, keywords)
    st.subheader("Grafo de Relaciones y Temas")
    default_layers = ['hashtags', 'mentions', 'keywords']
    selected_layers = st.multiselect(
        "Selecciona capas de datos:",
        options=default_layers,
        default=default_layers
    )

    if not selected_layers:
        st.info("Selecciona al menos una capa para construir el grafo.")
        st.stop()

    # Render del grafo (min_degree interno fijo a 1)
    html_content, G_full = render_interactive_graph(df_historico, selected_layers)

    if isinstance(html_content, str) and html_content.startswith("Error"):
        st.error(html_content)
    elif G_full is None or G_full.number_of_nodes() == 0:
        st.info("No hay suficientes nodos para mostrar con el filtro actual.")
    else:
        st.write(f"Nodos: **{len(G_full.nodes())}**, Enlaces: **{len(G_full.edges())}**")
        st.components.v1.html(html_content, height=700, scrolling=True)

