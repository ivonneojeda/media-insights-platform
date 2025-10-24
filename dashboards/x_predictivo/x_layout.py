# app.py
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import networkx as nx
from pyvis.network import Network
import community as community_louvain

# ------------------------
# CONFIG
# ------------------------
CSV_PATH = "data/Conversación sobre UNAM 5-7oct25 - ISO.csv"
DATE_COL = "created_at"
SENTIMENT_COL = "sentiment"
SENTIMENT_MAP = {"Positivo": 1.0, "Negativo": -1.0, "Neutro": 0.0}
FIG_BG = '#262730'
MODERN_COLOR_SCALE = ["#FF6347", "#D3D3D3", "#4CAF50"]
COLOR_NEG = "#FF6347"
COLOR_NEUTRO = "#D3D3D3"
COLOR_POS = "#4CAF50"
NODE_COLORS = {'hashtag': '#FF5733', 'mention': '#33C4FF', 'keyword': '#4CAF50'}

# ------------------------
# DATA LOAD & PREP
# ------------------------
@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        return df
    except FileNotFoundError:
        st.error(f"❌ Archivo no encontrado en: `{path}`")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
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
    df = df.rename(columns={SENTIMENT_COL: "sentiment_score"})
    # Asegurar columnas
    for col in ["keywords", "mentions"]:
        if col not in df.columns:
            df[col] = np.nan
    # Filtrar
    df = df.dropna(subset=[DATE_COL, "sentiment_score"])
    if df.empty:
        return pd.DataFrame()
    df["date_str"] = df[DATE_COL].dt.strftime("%b %d")
    df["hour_str"] = df[DATE_COL].dt.strftime("%H:00")
    return df

# ------------------------
# PROPHET FORECAST
# ------------------------
@st.cache_data(ttl=86400)
def generate_forecast(df_historico):
    df_prophet_input = df_historico.rename(columns={'created_at': 'ds', 'sentiment_score': 'y'})
    df_prophet_input = df_prophet_input[['ds', 'y']].set_index('ds').resample('H').mean().reset_index().dropna()
    m = Prophet(seasonality_mode='multiplicative', daily_seasonality=True)
    m.fit(df_prophet_input)
    future = m.make_future_dataframe(periods=24, freq='H', include_history=False)
    forecast = m.predict(future)
    df_pronostico = forecast[['ds', 'yhat']].rename(columns={'ds':'created_at','yhat':'sentiment_score'})
    df_pronostico['sentiment_score'] = np.clip(df_pronostico['sentiment_score'], -1, 1)
    return df_pronostico

# ------------------------
# GRAFO: EXTRACCIÓN, BUILD, CLUSTER
# ------------------------
def extract_hashtags_from_keywords(keywords_str):
    if pd.isna(keywords_str) or not isinstance(keywords_str, str): return []
    return [p.lstrip("#").lower() for p in keywords_str.split() if p.startswith("#")]

def extract_mentions_from_mentions_col(mentions_str):
    if pd.isna(mentions_str) or not isinstance(mentions_str, str): return []
    return [p.lstrip("@").lower() for p in mentions_str.split() if p.startswith("@")]

def extract_pure_keywords(keywords_str):
    if pd.isna(keywords_str) or not isinstance(keywords_str, str): return []
    return [p.strip().lower() for p in keywords_str.split() if not p.startswith("#") and not p.startswith("@")]

def build_filtered_graph(df, selected_layers):
    G = nx.Graph()
    for _, row in df.iterrows():
        row_entities = []
        if 'Hashtags' in selected_layers:
            row_entities.extend([(t,'hashtag') for t in extract_hashtags_from_keywords(row.get('keywords',''))])
        if 'Menciones (@)' in selected_layers:
            row_entities.extend([(m,'mention') for m in extract_mentions_from_mentions_col(row.get('mentions',''))])
        if 'Keywords' in selected_layers:
            row_entities.extend([(k,'keyword') for k in extract_pure_keywords(row.get('keywords',''))])
        names = [n for n,_ in row_entities]
        types = {n:t for n,t in row_entities}
        for i,a in enumerate(names):
            if not G.has_node(a):
                G.add_node(a, type=types.get(a), color=NODE_COLORS.get(types.get(a),'#FFFFFF'), size=1)
            for b in names[i+1:]:
                if G.has_edge(a,b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a,b, weight=1)
    return G

def apply_clustering_and_coloring(G):
    if G.number_of_edges() == 0: 
        return G
    try:
        partition = community_louvain.best_partition(G, weight='weight')
        base_colors = px.colors.qualitative.Dark24
        for node, cluster_id in partition.items():
            G.nodes[node]['cluster_id'] = int(cluster_id)
            G.nodes[node]['color'] = base_colors[cluster_id % len(base_colors)]
    except Exception:
        # no fallar si Louvain no funciona
        pass
    return G

# ------------------------
# VISUALIZACIONES
# ------------------------
def render_heatmap(df):
    if df.empty or not all(c in df.columns for c in ["sentiment_score","date_str","hour_str"]):
        st.warning("Datos inválidos o faltantes para el Heatmap.")
        return
    pivot = df.pivot_table(index="hour_str", columns="date_str", values="sentiment_score", aggfunc="mean").fillna(0).T
    hour_order = [f"{h:02d}:00" for h in range(24)]
    date_order = sorted(df["date_str"].unique(), key=lambda x: pd.to_datetime(x, format="%b %d"))
    full_index = pd.MultiIndex.from_product([date_order, hour_order], names=['date_str', 'hour_str'])
    df_plot = pivot.stack().reindex(full_index).fillna(0).reset_index(name='sentiment_mean')
    fig = px.density_heatmap(df_plot, x="date_str", y="hour_str", z="sentiment_mean",
                             histfunc="avg", color_continuous_scale=MODERN_COLOR_SCALE,
                             category_orders={"hour_str": hour_order, "date_str": date_order})
    fig.update_layout(title=dict(text="Distribución de Sentimiento Promedio por Hora (Histórico)", x=0.0, xanchor='left'),
                      height=600,
                      xaxis=dict(tickangle=-45),
                      plot_bgcolor=FIG_BG, paper_bgcolor=FIG_BG, font=dict(color="white"))
    fig.update_traces(xgap=1, ygap=1)
    st.plotly_chart(fig, use_container_width=True)

def render_gauge(df):
    if df.empty or "sentiment_score" not in df.columns:
        st.warning("Datos inválidos para el Gauge.")
        return
    last_pred = df["sentiment_score"].mean()
    if last_pred >= 0.33: INDICATOR_COLOR, sentiment_label = COLOR_POS, "POSITIVO"
    elif last_pred <= -0.33: INDICATOR_COLOR, sentiment_label = COLOR_NEG, "NEGATIVO"
    else: INDICATOR_COLOR, sentiment_label = COLOR_NEUTRO, "NEUTRO"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=np.clip(last_pred,-1,1),
        number={"valueformat":".2f", "font":dict(color='white',size=32)},
        gauge={"axis":{"range":[-1,1], "showticklabels": False}, "bar":{"color": INDICATOR_COLOR}}
    ))
    fig_gauge.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=FIG_BG, font=dict(color="white"))
    fig_gauge.add_annotation(text=sentiment_label, x=0.5, y=0.25, showarrow=False, font=dict(size=18,color="white"))
    st.plotly_chart(fig_gauge, use_container_width=True)

# =======================================================
# RENDER INTERACTIVE GRAPH
# =======================================================
def render_interactive_graph(df_historico, selected_layers, min_degree=1):
    """
    Construye y renderiza un grafo interactivo con PyVis.
    - df_historico: DataFrame preparado
    - selected_layers: Lista de capas ['Hashtags', 'Menciones (@)', 'Keywords']
    - min_degree: Grado mínimo de conexiones para mostrar un nodo
    """
    import networkx as nx
    from pyvis.network import Network
    import os
    import numpy as np
    import tempfile

    try:
        # === 1. Construcción del grafo base ===
        G = build_filtered_graph(df_historico, selected_layers)
        G = apply_clustering_and_coloring(G)

        if G.number_of_nodes() == 0:
            return "No hay suficientes nodos para graficar.", nx.Graph()

        # === 2. Configuración PyVis ===
        net = Network(height="600px", width="100%", bgcolor=FIG_BG, font_color="white", notebook=False)
        
        # Añadimos nodos filtrando por grado mínimo
        for n, d in G.nodes(data=True):
            degree = G.degree(n)
            if degree >= min_degree:
                size = max(10, int(np.log(degree + 1) * 15))  # Tamaño proporcional al grado
                net.add_node(
                    n,
                    label=n,
                    title=f"{n} (Grado: {degree}, Cluster: {d.get('cluster_id')})",
                    size=size,
                    color=d.get('color'),
                    font={'size': min(size + 2, 32)},  # Fuente proporcional al nodo
                    group=d.get('cluster_id')
                )

        # Filtrar aristas visibles
        visible_nodes = [n for n, d in G.nodes(data=True) if G.degree(n) >= min_degree]
        for u, v, attrs in G.edges(data=True):
            if u in visible_nodes and v in visible_nodes:
                weight = attrs.get("weight", 1)
                net.add_edge(u, v, value=weight, title=f"Co-ocurrencia: {weight}")

        # === 3. Física y opciones ===
        net.toggle_physics(True)
        try:
            net.force_atlas_2based(
                gravity=-30000,        # Ajuste más estable
                central_gravity=0.05,
                spring_length=100,
                spring_strength=0.05,
                damping=0.5
            )
        except Exception:
            pass
        net.options.edges.smooth = {'enabled': True}
        net.options.interaction = {'hover': True}

        # === 4. Renderizado seguro en memoria sin dejar archivos ===
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            net.save_graph(tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return html_content, G
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        return f"Error al generar el grafo: {e}", nx.Graph()



# ------------------------
# DASHBOARD (UN SOLO RENDER)
# ------------------------
def render_x_dashboard():
   
    st.title("🗣️ Análisis de Sentimiento y Conversación de X")
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

    # Pronóstico
    try:
        df_pronostico = generate_forecast(df_historico)
        df_viz_gauge = pd.concat([df_historico, df_pronostico], ignore_index=True)
    except Exception:
        df_viz_gauge = df_historico

    # Visualizaciones superiores
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔥 Histórico de Sentimiento")
        render_heatmap(df_historico)
    with col2:
        st.subheader("🔮 Pronóstico (Próxima Hora)")
        render_gauge(df_viz_gauge)

    st.markdown("---")

    st.subheader("🌐 Grafo de Relaciones y Temas")

    # Selección de capas
    default_layers = ['Hashtags', 'Menciones (@)', 'Keywords']
    selected_layers = st.multiselect("Selecciona capas de datos:", options=default_layers, default=default_layers)
    if not selected_layers:
        st.info("Selecciona al menos una capa para construir el grafo.")
        st.stop()

    # Slider para filtrar por grado mínimo
    min_deg = st.slider("Mínimo de conexiones (Grado) a mostrar:", 1, 5, 1)

    # Llamada al render
    html_content, G_full = render_interactive_graph(df_historico, selected_layers, min_degree=min_deg)

    if isinstance(html_content, str) and html_content.startswith("Error"):
        st.error(html_content)
    elif G_full.number_of_nodes() == 0:
        st.info("No hay suficientes nodos para mostrar con el filtro actual.")
    else:
        st.write(f"Nodos: **{len(G_full.nodes())}**, Enlaces: **{len(G_full.edges())}**")
        st.components.v1.html(html_content, height=600, scrolling=True)


# ------------------------
# RUN
# ------------------------
if __name__ == "__main__":
    render_x_dashboard()
