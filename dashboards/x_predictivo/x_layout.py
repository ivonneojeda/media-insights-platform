# dashboards/x_predictivo/x_layout.py
import os
import json
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network

# Algunas instalaciones usan "community" o "python-louvain"
# Intentamos importar de forma segura.
try:
    import community as community_louvain
except Exception:
    community_louvain = None

# ------------------------
# CONFIG
# ------------------------
CSV_PATH = "data/Conversacion_UNAM_limpio.csv"  # ruta relativa dentro del repo
DATE_COL = "created_at"
SENTIMENT_COL = "sentiment"
SENTIMENT_MAP = {"Positivo": 1.0, "Negativo": -1.0, "Neutro": 0.0}
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
    """
    Normaliza columnas esperadas:
    - created_at -> datetime
    - sentiment -> mapeado a sentiment_score
    - hashtags, keywords, mentions -> listas (si vienen como strings 'a, b, c' se transforman)
    - date_str, hour_str para heatmap
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Fecha
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df.get(DATE_COL), errors="coerce")
        df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    else:
        df[DATE_COL] = pd.NaT

    # Sentimiento
    # detectar columna de sentimiento o crearla si no existe
    sentiment_col = next((c for c in df.columns if "sentiment" in c.lower()), None)
    if sentiment_col:
        df[sentiment_col] = df[sentiment_col].map(SENTIMENT_MAP).astype(float)
        df = df.rename(columns={sentiment_col: "sentiment_score"})
    else:
        df["sentiment_score"] = np.nan

    if df["sentiment_score"].isna().all():
        st.warning("⚠️ No se detectaron datos de sentimiento válidos en el CSV.")

    # Columnas de entidades: aseguramos que existan y que sean listas de strings
    for col in ["hashtags", "keywords", "mentions"]:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        else:
            # Si vienen como string con comas o espacios, convertir a listas
            def to_list_cell(x):
                if pd.isna(x):
                    return []
                if isinstance(x, list):
                    # Aplanar listas internas de strings si hay
                    out = []
                    for el in x:
                        if isinstance(el, list):
                            out.extend([str(s).strip() for s in el if str(s).strip()])
                        else:
                            if str(el).strip():
                                out.append(str(el).strip())
                    return out
                # string
                s = str(x).strip()
                if s == "":
                    return []
                # split por comas, punto y coma o pipes o dobles espacios
                parts = [p.strip() for p in re_split_delims(s) if p.strip()]
                return parts

            df[col] = df[col].apply(to_list_cell)

    # date_str / hour_str para heatmap
    try:
        df["date_str"] = df[DATE_COL].dt.strftime("%b %d")
        df["hour_str"] = df[DATE_COL].dt.strftime("%H:00")
    except Exception:
        df["date_str"] = ""
        df["hour_str"] = ""

    return df

# helper para split flexible
def re_split_delims(s):
    import re
    # separadores comunes: coma, punto y coma, pipe, barra, barra inversa, espacios múltiples
    return re.split(r"[,\;\|\:/\\\t]+|\s{2,}|\s", s)

# ------------------------
# GRAFO: BUILD + LOUVAIN + PYVIS
# ------------------------
def render_interactive_graph(df_historico, selected_layers, min_degree=1):
    """
    Construye el HTML del grafo y devuelve (html, G_filtered).
    - selected_layers: lista con alguno de ['hashtags','mentions','keywords']
    - min_degree: grado mínimo (por defecto 1, como pediste sin slider)
    """
    try:
        if df_historico is None or df_historico.empty:
            return "Error: DataFrame vacío o no proporcionado.", nx.Graph()

        # Asegurarse columnas en minúscula (consistencia)
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

            # Crear aristas entre todas las entidades de la misma fila (co-ocurrencia)
            for i, a in enumerate(row_entities):
                if not G.has_node(a):
                    G.add_node(a)
                for b in row_entities[i+1:]:
                    if G.has_edge(a, b):
                        G[a][b]['weight'] += 1
                    else:
                        G.add_edge(a, b, weight=1)

        # Filtrar por grado mínimo
        if G.number_of_nodes() == 0:
            return "Error: El grafo no contiene nodos con el filtrado actual.", nx.Graph()

        # aplicamos min_degree: si min_degree > 1, quitamos nodos con grado < min_degree
        nodes_to_keep = [n for n, d in G.degree() if d >= max(1, int(min_degree))]
        G_filtered = G.subgraph(nodes_to_keep).copy()
        if G_filtered.number_of_nodes() == 0:
            return "Error: No quedan nodos tras aplicar el filtrado por grado.", nx.Graph()

        # Louvain clustering (si está disponible)
        if community_louvain is not None and G_filtered.number_of_edges() > 0:
            try:
                partition = community_louvain.best_partition(G_filtered, weight='weight')
                # Asignar color por cluster
                base_colors = px.colors.qualitative.Dark24
                for node, cluster_id in partition.items():
                    G_filtered.nodes[node]['cluster_id'] = int(cluster_id)
                    G_filtered.nodes[node]['color'] = base_colors[int(cluster_id) % len(base_colors)]
            except Exception as e:
                # no fallar si Louvain da problemas
                print("Louvain falló:", e)
        else:
            # Si no hay community lib, dejamos color por defecto
            for n in G_filtered.nodes():
                G_filtered.nodes[n]['color'] = "#888888"

        # Construir PyVis
        net = Network(height="700px", width="100%", bgcolor=FIG_BG, font_color="white", notebook=False)
        for n, data in G_filtered.nodes(data=True):
            deg = G_filtered.degree(n)
            size = max(12, min(int(math.log(deg + 1) * 20), 60))
            net.add_node(
                n,
                label=n,
                title=f"{n} (Grado: {deg}, Cluster: {data.get('cluster_id', 'n/a')})",
                size=size,
                color=data.get('color', '#888888'),
                font={'size': 14, 'face': 'Arial', 'color': '#FFFFFF'}
            )

        for u, v, attrs in G_filtered.edges(data=True):
            net.add_edge(u, v, value=attrs.get('weight', 1), title=f"Co-ocurrencia: {attrs.get('weight', 1)}")

        # Física (suave)
        net.toggle_physics(True)
        try:
            net.force_atlas_2based(gravity=-40, central_gravity=0.02, spring_length=120, spring_strength=0.05,
                                   damping=0.6, overlap=0.5)
        except Exception:
            pass

        # Opciones JSON compactas
        options_dict = {
            "nodes": {"font": {"size": 14}, "scaling": {"min": 10, "max": 60}},
            "edges": {"color": {"inherit": True}, "smooth": {"enabled": True, "type": "dynamic"}},
            "physics": {"enabled": True, "solver": "forceAtlas2Based"},
            "interaction": {"hover": True, "zoomView": True, "dragNodes": True}
        }
        net.set_options(json.dumps(options_dict))

        # Generar HTML, limpiar BOM/nulos
        try:
            html = net.generate_html()
            html = html.replace("\ufeff", "").replace("\x00", "")
        except Exception as e:
            return f"Error al generar el grafo: {e}", nx.Graph()

        return html, G_filtered

    except Exception as e:
        return f"Error inesperado en render_interactive_graph: {e}", nx.Graph()


# ------------------------
# VISUALIZACIONES: HEATMAP Y GAUGE
# ------------------------
def render_heatmap(df):
    if df.empty or not all(c in df.columns for c in ["sentiment_score", "date_str", "hour_str"]):
        st.warning("Datos inválidos o faltantes para el Heatmap.")
        return

    pivot = df.pivot_table(index="hour_str", columns="date_str", values="sentiment_score", aggfunc="mean").fillna(0).T
    hour_order = [f"{h:02d}:00" for h in range(24)]
    date_order = sorted(df["date_str"].unique(), key=lambda x: pd.to_datetime(x, format="%b %d"))
    full_index = pd.MultiIndex.from_product([date_order, hour_order], names=['date_str', 'hour_str'])
    df_plot = pivot.stack().reindex(full_index).fillna(0).reset_index(name='sentiment_mean')
    fig = px.density_heatmap(df_plot, x="date_str", y="hour_str", z="sentiment_mean",
                             color_continuous_scale=MODERN_COLOR_SCALE,
                             category_orders={"hour_str": hour_order, "date_str": date_order})
    fig.update_layout(title=dict(text="Distribución de Sentimiento Promedio por Hora (Histórico)", x=0.0, xanchor='left'),
                      height=600, xaxis=dict(tickangle=-45),
                      plot_bgcolor=FIG_BG, paper_bgcolor=FIG_BG, font=dict(color="white"))
    fig.update_traces(xgap=1, ygap=1)
    st.plotly_chart(fig, use_container_width=True)

def render_gauge(df):
    if df.empty or "sentiment_score" not in df.columns:
        st.warning("Datos inválidos para el Gauge.")
        return
    last_pred = df["sentiment_score"].mean()
    if last_pred >= 0.33:
        INDICATOR_COLOR, sentiment_label = COLOR_POS, "POSITIVO"
    elif last_pred <= -0.33:
        INDICATOR_COLOR, sentiment_label = COLOR_NEG, "NEGATIVO"
    else:
        INDICATOR_COLOR, sentiment_label = COLOR_NEUTRO, "NEUTRO"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=np.clip(last_pred, -1, 1),
        number={"valueformat":".2f", "font":dict(color='white', size=32)},
        gauge={"axis": {"range": [-1, 1], "showticklabels": False}, "bar": {"color": INDICATOR_COLOR}}
    ))
    fig_gauge.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=FIG_BG, font=dict(color="white"))
    fig_gauge.add_annotation(text=sentiment_label, x=0.5, y=0.25, showarrow=False, font=dict(size=18, color="white"))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------
# DASHBOARD (RENDER PRINCIPAL)
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

    # Visualizaciones superiores: heatmap + gauge
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔥 Histórico de Sentimiento")
        render_heatmap(df_historico)
    with col2:
        st.subheader("🔮 Pronóstico (Promedio de Sentimiento)")
        render_gauge(df_historico)

    st.markdown("---")

    # Grafo - multiselect por capas (sin slider)
    st.subheader("🌐 Grafo de Relaciones y Temas")
    default_layers = ['hashtags', 'mentions', 'keywords']
    selected_layers = st.multiselect("Selecciona capas de datos:", options=default_layers, default=default_layers)
    if not selected_layers:
        st.info("Selecciona al menos una capa para construir el grafo.")
        st.stop()

    # Render del grafo (min_degree fijo a 1)
    html_content, G_full = render_interactive_graph(df_historico, selected_layers, min_degree=1)

    if isinstance(html_content, str) and html_content.startswith("Error"):
        st.error(html_content)
    elif G_full is None or G_full.number_of_nodes() == 0:
        st.info("No hay suficientes nodos para mostrar con el filtro actual.")
    else:
        st.write(f"Nodos: **{len(G_full.nodes())}**, Enlaces: **{len(G_full.edges())}**")
        st.components.v1.html(html_content, height=700, scrolling=True)

