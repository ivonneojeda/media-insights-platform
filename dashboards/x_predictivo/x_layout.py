# x_layout.py
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import math
import json

# ------------------------
# CONFIG
# ------------------------
CSV_PATH = "data/Conversación sobre UNAM 5-7oct25 - ISO.csv"
DATE_COL = "created_at"
SENTIMENT_COL = "sentiment"
SENTIMENT_MAP = {"Positivo": 1.0, "Negativo": -1.0, "Neutro": 0.0}
NODE_COLORS = {'hashtag': '#FF5733', 'mention': '#33C4FF', 'keyword': '#4CAF50'}

# ------------------------
# DATA LOAD & PREP
# ------------------------
@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        return df
    except Exception as e:
        st.error(f"No se pudo cargar CSV: {e}")
        return pd.DataFrame()

def prepare_data(df):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df.get(DATE_COL), errors="coerce").dt.tz_localize(None)
    df[SENTIMENT_COL] = df[SENTIMENT_COL].map(SENTIMENT_MAP) if SENTIMENT_COL in df.columns else np.nan
    df = df.rename(columns={SENTIMENT_COL: "sentiment_score"})
    df["date_str"] = df[DATE_COL].dt.strftime("%b %d")
    df["hour_str"] = df[DATE_COL].dt.strftime("%H:00")
    for col in ["keywords", "mentions"]:
        if col not in df.columns: df[col] = np.nan
        df[col] = df[col].fillna("").astype(str).str.split(r"[;,]\s*")
    return df

# ------------------------
# GRAFO INTERACTIVO
# ------------------------
def render_interactive_graph(df, selected_layers, min_degree=2):
    if df.empty: return "Error: DataFrame vacío", nx.Graph()

    G = nx.Graph()
    for _, row in df.iterrows():
        row_entities = []
        if "Hashtags" in selected_layers:
            row_entities.extend([(h.lower(), 'hashtag') for h in row.get("keywords", []) if h.startswith("#")])
        if "Menciones (@)" in selected_layers:
            row_entities.extend([(m.lower(), 'mention') for m in row.get("mentions", []) if m.startswith("@")])
        if "Keywords" in selected_layers:
            row_entities.extend([(k.lower(), 'keyword') for k in row.get("keywords", []) if not k.startswith("#")])

        names = [n for n, _ in row_entities]
        types = {n: t for n, t in row_entities}

        for i, a in enumerate(names):
            if not G.has_node(a):
                G.add_node(a, type=types[a], color=NODE_COLORS.get(types[a], "#FFFFFF"), size=1)
            for b in names[i+1:]:
                if G.has_edge(a,b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a,b, weight=1)

    # Clustering Louvain
    try:
        import community as community_louvain
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G, weight='weight')
            base_colors = ["#FF5733","#33C4FF","#4CAF50","#FF33A1","#A133FF"]
            for node, cluster_id in partition.items():
                G.nodes[node]['cluster_id'] = int(cluster_id)
                G.nodes[node]['color'] = base_colors[cluster_id % len(base_colors)]
    except Exception:
        pass

    # PyVis
    net = Network(height="700px", width="100%", bgcolor="#262730", font_color="white", notebook=False)
    visible_nodes = [n for n in G.nodes if G.degree(n) >= min_degree]
    for n in visible_nodes:
        deg = G.degree(n)
        size = max(12, min(int(math.log(deg+1)*20), 60))
        font_size = max(10, min(size*2//3, 28))
        net.add_node(n, label=n, title=f"{n} (Grado: {deg})", size=size, color=G.nodes[n].get('color','#888888'),
                     font={'size': font_size,'face':'Arial','color':'#FFFFFF'})

    for u, v, attrs in G.edges(data=True):
        if u in visible_nodes and v in visible_nodes:
            net.add_edge(u, v, value=attrs.get('weight',1), title=f"Co-ocurrencia: {attrs.get('weight',1)}")

    net.toggle_physics(True)
    try:
        net.force_atlas_2based(gravity=-40, central_gravity=0.02, spring_length=150, spring_strength=0.05, damping=0.6, overlap=0.5)
    except Exception:
        pass

    try:
        html = net.generate_html().replace("\ufeff","").replace("\x00","")
    except Exception as e:
        return f"Error al generar grafo: {e}", nx.Graph()

    return html, G

# ------------------------
# DASHBOARD X
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

    st.subheader("🌐 Grafo de Relaciones y Temas")
    default_layers = ['Hashtags', 'Menciones (@)', 'Keywords']
    selected_layers = st.multiselect("Selecciona capas:", default_layers, default=default_layers)
    if not selected_layers:
        st.info("Selecciona al menos una capa.")
        return
    min_deg = st.slider("Mínimo de conexiones (grado):", 1, 5, 1)

    html, G = render_interactive_graph(df_historico, selected_layers, min_degree=min_deg)
    if isinstance(html, str) and html.startswith("Error"):
        st.error(html)
    elif G.number_of_nodes() == 0:
        st.info("No hay suficientes nodos para mostrar.")
    else:
        st.write(f"Nodos: **{len(G.nodes())}**, Enlaces: **{len(G.edges())}**")
        st.components.v1.html(html, height=700, scrolling=True)
