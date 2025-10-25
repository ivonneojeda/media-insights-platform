# dashboards/x_predictivo/x_layout.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import math
import json
from prophet import Prophet

# ------------------------
# CONFIGURACIÓN
# ------------------------
CSV_PATH = "data/Conversacion_UNAM_limpio.csv"
DATE_COL = "created_at"
SENTIMENT_COL = "sentiment"
SENTIMENT_MAP = {"Positivo": 1.0, "Negativo": -1.0, "Neutro": 0.0}
FIG_BG = '#262730'
MODERN_COLOR_SCALE = ["#FF6347", "#D3D3D3", "#4CAF50"]
COLOR_NEG = "#FF6347"
COLOR_NEUTRO = "#D3D3D3"
COLOR_POS = "#4CAF50"

# ------------------------
# CARGA DE DATOS
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
    df[DATE_COL] = pd.to_datetime(df.get(DATE_COL), errors="coerce")
    df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)

    # Sentimiento
    if SENTIMENT_COL in df.columns:
        df[SENTIMENT_COL] = df[SENTIMENT_COL].map(SENTIMENT_MAP).astype(float)
    else:
        df[SENTIMENT_COL] = np.nan

    if df[SENTIMENT_COL].isna().all():
        st.warning("⚠️ No se detectaron datos de sentimiento válidos en el CSV.")

    df = df.rename(columns={SENTIMENT_COL: "sentiment_score"})

    # Columnas obligatorias
    for col in ["hashtags", "keywords", "mentions"]:
        if col not in df.columns:
            df[col] = np.nan
        # Convertir a listas si no lo son
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else str(x).split() if pd.notna(x) else [])

    # Para heatmap
    df["date_str"] = df[DATE_COL].dt.strftime("%b %d")
    df["hour_str"] = df[DATE_COL].dt.strftime("%H:00")
    return df

# ------------------------
# GRAFO INTERACTIVO
# ------------------------
def render_interactive_graph(df_historico, selected_layers, min_degree=2):
    """
    Construye un grafo interactivo para Streamlit, usando columnas separadas:
    hashtags, keywords, mentions.
    """
    if df_historico is None or df_historico.empty:
        return "Error: DataFrame vacío o no proporcionado.", nx.Graph()

    G = nx.Graph()
    for idx, row in df_historico.iterrows():
        row_nodes = []
        if 'hashtags' in selected_layers:
            row_nodes.extend([h.lower() for h in row.get('hashtags', []) if isinstance(h, str) and h])
        if 'mentions' in selected_layers:
            row_nodes.extend([m.lower() for m in row.get('mentions', []) if isinstance(m, str) and m])
        if 'keywords' in selected_layers:
            row_nodes.extend([k.lower() for k in row.get('keywords', []) if isinstance(k, str) and k])

        for i, a in enumerate(row_nodes):
            if not G.has_node(a):
                G.add_node(a, size=1)
            for b in row_nodes[i+1:]:
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)

    visible_nodes = [n for n in G.nodes if G.degree(n) >= min_degree]
    G_filtered = G.subgraph(visible_nodes).copy()

    net = Network(height="700px", width="100%", bgcolor="#262730", font_color="white", notebook=False)
    for n in G_filtered.nodes:
        deg = G_filtered.degree(n)
        node_size = max(12, min(int(math.log(deg + 1) * 20), 60))
        font_size = max(10, min(node_size * 2 // 3, 28))
        net.add_node(n, label=n, title=f"{n} (Grado: {deg})",
                     size=node_size, color="#888888",
                     font={'size': font_size, 'face': 'Arial', 'color': '#FFFFFF'})

    for u, v, attrs in G_filtered.edges(data=True):
        net.add_edge(u, v, value=attrs.get('weight', 1), title=f"Co-ocurrencia: {attrs.get('weight', 1)}")

    net.toggle_physics(True)
    try:
        net.force_atlas_2based()
    except Exception:
        pass

    options_dict = {
        "nodes": {"font": {"size": 18, "strokeWidth": 3}, "scaling": {"min": 10, "max": 60}},
        "edges": {"color": {"inherit": True}, "smooth": {"enabled": True, "type": "dynamic"}, "width": 1},
        "physics": {"enabled": True, "solver": "forceAtlas2Based"},
        "interaction": {"hover": True, "tooltipDelay": 100, "zoomView": True, "dragNodes": True}
    }
    net.set_options(json.dumps(options_dict))

    try:
        html = net.generate_html()
        html = html.replace("\ufeff", "").replace("\x00", "")
    except Exception as e:
        return f"Error al generar el grafo: {e}", nx.Graph()

    return html, G_filtered

# ------------------------
# HEATMAP
# ------------------------
def render_heatmap(df):
    if df.empty or not all(c in df.columns for c in ["sentiment_score","date_str","hour_str"]):
        st.warning("Datos inválidos para el Heatmap.")
        return

    pivot = df.pivot_table(index="hour_str", columns="date_str", values="sentiment_score", aggfunc="mean").fillna(0).T
    hour_order = [f"{h:02d}:00" for h in range(24)]
    date_order = sorted(df["date_str"].unique(), key=lambda x: pd.to_datetime(x, format="%b %d"))
    full_index = pd.MultiIndex.from_product([date_order, hour_order], names=['date_str', 'hour_str'])
    df_plot = pivot.stack().reindex(full_index).fillna(0).reset_index(name='sentiment_mean')

    fig = px.density_heatmap(df_plot, x="date_str", y="hour_str", z="sentiment_mean",
                             histfunc="avg", color_continuous_scale=MODERN_COLOR_SCALE,
                             category_orders={"hour_str": hour_order, "date_str": date_order})
    fig.update_layout(title=dict(text="Distribución de Sentimiento Promedio por Hora", x=0.0, xanchor='left'),
                      height=600,
                      xaxis=dict(tickangle=-45),
                      plot_bgcolor=FIG_BG, paper_bgcolor=FIG_BG, font=dict(color="white"))
    fig.update_traces(xgap=1, ygap=1)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# GAUGE
# ------------------------
def render_gauge(df):
    if df.empty or "sentiment_score" not in df.columns:
        st.warning("Datos inválidos para el Gauge.")
        return
    last_score = df["sentiment_score"].mean()
    if last_score >= 0.33:
        color, label = COLOR_POS, "POSITIVO"
    elif last_score <= -0.33:
        color, label = COLOR_NEG, "NEGATIVO"
    else:
        color, label = COLOR_NEUTRO, "NEUTRO"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=np.clip(last_score, -1, 1),
        number={"valueformat":".2f", "font":dict(color='white',size=32)},
        gauge={"axis":{"range":[-1,1], "showticklabels": False}, "bar":{"color": color}}
    ))
    fig_gauge.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=FIG_BG, font=dict(color="white"))
    fig_gauge.add_annotation(text=label, x=0.5, y=0.25, showarrow=False, font=dict(size=18,color="white"))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------
# DASHBOARD COMPLETO
# ------------------------
def render_x_dashboard():
    st.title("🗣️ Análisis de Sentimiento y Conversación de X")
    st.markdown("---")

    df_raw = load_data()
    if df_raw.empty:
        st.info("No hay datos cargados.")
        return
    df_historico = prepare_data(df_raw)
    if df_historico.empty:
        st.info("No hay datos válidos después de la limpieza.")
        return

    # Visualizaciones
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("🔥 Histórico de Sentimiento")
        render_heatmap(df_historico)
    with col2:
        st.subheader("🔮 Pronóstico (Próxima Hora)")
        render_gauge(df_historico)

    st.markdown("---")
    st.subheader("🌐 Grafo de Relaciones y Temas")
    default_layers = ['hashtags','mentions','keywords']
    selected_layers = st.multiselect("Selecciona capas de datos:", options=default_layers, default=default_layers)
    if not selected_layers:
        st.info("Selecciona al menos una capa para construir el grafo.")
        st.stop()

    min_deg = st.slider("Mínimo de conexiones (Grado) a mostrar:", 1, 5, 1)
    html_content, G_full = render_interactive_graph(df_historico, selected_layers, min_degree=min_deg)

    if isinstance(html_content, str) and html_content.startswith("Error"):
        st.error(html_content)
    elif G_full.number_of_nodes() == 0:
        st.info("No hay suficientes nodos para mostrar con el filtro actual.")
    else:
        st.write(f"Nodos: **{len(G_full.nodes)}**, Conexiones: **{len(G_full.edges)}**")
        st.components.v1.html(html_content, height=700, scrolling=True)
