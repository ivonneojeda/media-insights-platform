import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
# Librerías para el grafo
import networkx as nx
from pyvis.network import Network
import community as community_louvain # Requiere 'python-louvain'
import json 

# =======================================================
# 1. CONFIGURACIÓN Y CONSTANTES
# =======================================================

# CONFIG: ruta del CSV (Ajusta esto si vas a desplegar en la nube)
# RECUERDA: Cambiar esta ruta a una relativa si despliegas en la nube.
CSV_PATH = r"C:\Users\ivonn\Desktop\x_sentiment_analysis\datos\Conversación sobre UNAM 5-7oct25 - ISO.csv" 

# Definiciones de columnas y mapeos
DATE_COL = "created_at"
SENTIMENT_COL = "sentiment"
SENTIMENT_MAP = {"Positivo": 1.0, "Negativo": -1.0, "Neutro": 0.0}

# Colores y estilos unificados
FIG_BG = '#262730'
MODERN_COLOR_SCALE = ["#FF6347", "#F0F0F0", "#4CAF50"] # Rojo, Gris, Verde
COLOR_NEG = "#FF6347"
COLOR_NEUTRO = "#D3D3D3"
COLOR_POS = "#4CAF50"
NODE_COLORS = {
    'hashtag': '#FF5733',   # Naranja/Rojo
    'mention': '#33C4FF',   # Azul claro
    'keyword': '#4CAF50',   # Verde
}

# =======================================================
# 2. CARGA Y PREPARACIÓN DE DATOS
# =======================================================

@st.cache_data(ttl=600)
def load_data(path=CSV_PATH):
    """Carga el DataFrame desde la ruta especificada."""
    if not os.path.exists(path):
        st.error(f"¡Error! No se encontró el archivo CSV en: {path}")
        return pd.DataFrame() 
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        return df
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return pd.DataFrame()

def prepare_data(df):
    """Limpia y prepara el DataFrame."""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.tz_localize(None)
    df[SENTIMENT_COL] = df[SENTIMENT_COL].map(SENTIMENT_MAP).astype(float)
    df = df.rename(columns={SENTIMENT_COL: "sentiment_score"})
    
    if "keywords" not in df.columns:
        df["keywords"] = np.nan
    if "mentions" not in df.columns:
        df["mentions"] = np.nan
    
    df = df.dropna(subset=[DATE_COL, "sentiment_score"]) 

    if df.empty:
        st.warning("No hay datos válidos después de la limpieza y mapeo de sentimiento.")
        return pd.DataFrame()
        
    df["date_str"] = df[DATE_COL].dt.strftime("%b %d")
    df["hour_str"] = df[DATE_COL].dt.strftime("%H:00")
    
    return df

# =======================================================
# 3. FUNCIONES DE PROPHET (MODELADO)
# =======================================================

@st.cache_data(ttl=86400)
def generate_forecast(df_historico):
    """Entrena el modelo Prophet y genera la predicción para 24 horas."""
    st.info("Calculando pronóstico de sentimiento (puede tardar unos segundos)...")
    
    df_prophet_input = df_historico.rename(columns={'created_at': 'ds', 'sentiment_score': 'y'})
    df_prophet_input = df_prophet_input[['ds', 'y']].set_index('ds').resample('H').mean().reset_index().dropna()

    m = Prophet(seasonality_mode='multiplicative', daily_seasonality=True)
    m.fit(df_prophet_input)
    
    future = m.make_future_dataframe(periods=24, freq='H', include_history=False)
    forecast = m.predict(future)
    
    df_pronostico = forecast[['ds', 'yhat']].copy()
    df_pronostico = df_pronostico.rename(columns={'ds': 'created_at', 'yhat': 'sentiment_score'})
    df_pronostico['sentiment_score'] = np.clip(df_pronostico['sentiment_score'], -1, 1)
    
    return df_pronostico

# =======================================================
# 4. FUNCIONES DEL GRAFO (EXTRACCIÓN, CLUSTERING Y CONSTRUCCIÓN)
# =======================================================

# --- A. Funciones de Extracción (Ajustadas) ---

def extract_hashtags_from_keywords(keywords_str):
    """Extrae hashtags de la columna 'keywords', eliminando el #."""
    if pd.isna(keywords_str) or not isinstance(keywords_str, str):
        return []
    parts = keywords_str.split()
    return [p.lstrip("#").lower() for p in parts if p.startswith("#")]

def extract_mentions_from_mentions_col(mentions_str):
    """Extrae menciones de la columna 'mentions', eliminando el @."""
    if pd.isna(mentions_str) or not isinstance(mentions_str, str):
        return []
    parts = mentions_str.split()
    return [p.lstrip("@").lower() for p in parts if p.startswith("@")]

def extract_pure_keywords(keywords_str):
    """Extrae keywords puras de la columna 'keywords', filtrando # y @."""
    if pd.isna(keywords_str) or not isinstance(keywords_str, str):
        return []
    
    parts = keywords_str.split()
    keywords_limpias = []
    
    for p in parts:
        p_strip = p.strip()
        if p_strip and not p_strip.startswith('#') and not p_strip.startswith('@'):
            keywords_limpias.append(p_strip.lower())
            
    return keywords_limpias

# --- B. Construcción del Grafo ---

def build_filtered_graph(df, selected_layers):
    """Construye un grafo de co-ocurrencia para las capas seleccionadas."""
    G = nx.Graph()
    all_entities = []
    
    for _, row in df.iterrows():
        row_entities = []
        if 'Hashtags' in selected_layers:
            tags = extract_hashtags_from_keywords(row['keywords'])
            row_entities.extend([(tag, 'hashtag') for tag in tags])
        if 'Menciones (@)' in selected_layers:
            ments = extract_mentions_from_mentions_col(row['mentions'])
            row_entities.extend([(ment, 'mention') for ment in ments])
        if 'Keywords' in selected_layers:
            kws = extract_pure_keywords(row['keywords'])
            row_entities.extend([(kw, 'keyword') for kw in kws])
        
        all_entities.append(row_entities)

    for entities_in_tweet in all_entities:
        names = [name for name, _ in entities_in_tweet]
        types = {name: type_ for name, type_ in entities_in_tweet} 
        
        for i, a in enumerate(names):
            G.add_node(a, type=types.get(a), color=NODE_COLORS.get(types.get(a), '#FFFFFF'), size=1)
            
            for b in names[i+1:]:
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)
                    
    return G

# --- C. Clustering y Coloreado ---

def apply_clustering_and_coloring(G):
    """Detecta clusters (comunidades) usando Louvain y asigna un color único."""
    if G.number_of_edges() == 0:
        return G

    try:
        partition = community_louvain.best_partition(G, weight='weight')
        num_clusters = max(partition.values()) + 1
        base_colors = px.colors.qualitative.Dark24
        
        for node, cluster_id in partition.items():
            cluster_color = base_colors[cluster_id % len(base_colors)]
            G.nodes[node]['cluster_id'] = cluster_id
            G.nodes[node]['color'] = cluster_color
            
    except Exception as e:
        st.warning(f"No se pudo ejecutar la detección de Louvain. {e}")
        
    return G

# =======================================================
# 5. FUNCIONES DE VISUALIZACIÓN (Heatmap, Gauge, Grafo)
# =======================================================

# --- Heatmap (render_heatmap) ---
def render_heatmap(df):
    """Renderiza el Heatmap solo con datos históricos."""
    if df.empty or "sentiment_score" not in df.columns or "date_str" not in df.columns or "hour_str" not in df.columns:
        st.warning("Datos inválidos o faltantes para calcular el Heatmap.")
        return
    
    pivot = df.pivot_table(
        index="hour_str", columns="date_str", values="sentiment_score", aggfunc="mean"
    ).fillna(0).T 

    hour_order = [f"{h:02d}:00" for h in range(24)] 
    date_order = sorted(df["date_str"].unique(), key=lambda x: pd.to_datetime(x, format="%b %d"))

    full_index = pd.MultiIndex.from_product([date_order, hour_order], names=['date_str', 'hour_str'])
    df_plot = pivot.stack().reindex(full_index).fillna(0).reset_index(name='sentiment_mean')
    
    fig = px.density_heatmap(
        df_plot, x="date_str", y="hour_str", z="sentiment_mean", histfunc="avg", 
        color_continuous_scale=MODERN_COLOR_SCALE, 
        category_orders={"hour_str": hour_order, "date_str": date_order} 
    )

    fig.update_layout(
        title=dict(text="Distribución de Sentimiento Promedio por Hora (Histórico)", x=0.0, font=dict(size=14, color='white'), xanchor='left'),
        height=600,
        xaxis=dict(title='Fecha', tickangle=-45, tickfont=dict(size=10, color='white'), showgrid=False, zeroline=False, categoryorder='array', categoryarray=date_order ),
        yaxis=dict(title='Hora', tickfont=dict(size=10, color='white'), autorange="reversed", showgrid=False, zeroline=False, type='category', categoryorder='array', categoryarray=hour_order ),
        plot_bgcolor=FIG_BG, paper_bgcolor=FIG_BG, font=dict(color='white'),
        coloraxis=dict(cmin=-1, cmax=1,  colorbar=dict(title="Sentimiento", title_font=dict(color='white', size=10), tickfont=dict(color='white', size=9), tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["Negativo", "Leve -", "Neutro", "Leve +", "Positivo"], orientation="v", y=0.5, len=0.75, x=1.05)),
        margin=dict(l=50, r=120, t=60, b=50)
    )

    fig.update_traces(xgap=1, ygap=1)
    st.plotly_chart(fig, use_container_width=True)

# --- Gauge (render_gauge) ---
def render_gauge(df):
    """Renderiza el PRONÓSTICO de Sentimiento para la fecha más reciente en el DF."""
    if df.empty or "sentiment_score" not in df.columns or "created_at" not in df.columns:
        st.warning("Datos inválidos o faltantes para calcular el Pronóstico.")
        return
    
    forecast_datetime = df[DATE_COL].max()
    forecast_date_only = forecast_datetime.date()
    
    forecast_df = df[df[DATE_COL].dt.date == forecast_date_only]
    last_pred = forecast_df["sentiment_score"].mean()
    
    if last_pred >= 0.33:
        INDICATOR_COLOR = COLOR_POS
        sentiment_label = "POSITIVO"
    elif last_pred <= -0.33:
        INDICATOR_COLOR = COLOR_NEG
        sentiment_label = "NEGATIVO"
    else:
        INDICATOR_COLOR = COLOR_NEUTRO
        sentiment_label = "NEUTRO"

    forecast_date_str = forecast_datetime.strftime('%d %b')
    title_text = f"Pronóstico<br><sub>{forecast_date_str}</sub>"
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=np.clip(last_pred, -1, 1), 
        number={"valueformat": ".2f", "font": dict(color='white', size=32)}, 
        gauge={
            "axis": {"range": [-1, 1], "showticklabels": False, "tickcolor": "rgba(255,255,255,0.2)"},
            "steps": [{"range": [-1, 1], "color": "rgba(100, 100, 100, 0.4)"}],
            "bar": {"color": INDICATOR_COLOR, "thickness": 0.8}, 
            "threshold": {"line": {"color": INDICATOR_COLOR, "width": 4}, "thickness": 0.75, "value": last_pred,}
        },
        title={"text": title_text, "font": dict(color='white', size=12)} 
    ))
    
    fig_gauge.add_annotation(
        text=sentiment_label, x=0.5, y=0.3, showarrow=False,
        font=dict(size=18, color="white", weight='bold')
    )
    
    fig_gauge.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor=FIG_BG, font=dict(color="white"))
    st.plotly_chart(fig_gauge, use_container_width=True)


# --- Grafo Interactivo (render_interactive_graph) ---
# =======================================================
# 5. FUNCIONES DE VISUALIZACIÓN (GRAFO FINAL)
# =======================================================


def render_interactive_graph(df_historico, selected_layers):
    """
    Construye, clusteriza y renderiza el grafo. 
    """
    if not selected_layers:
        return "No hay suficientes datos para construir un grafo.", nx.Graph()
    
    # 1. Construir y Clusterizar
    G = build_filtered_graph(df_historico, selected_layers)
    G = apply_clustering_and_coloring(G)
    
    if G.number_of_nodes() == 0:
        return "No hay suficientes datos para construir un grafo.", G

    # 2. Renderizado del Grafo con Pyvis
    min_degree_default = 2 
    
    net = Network(height="600px", width="100%", bgcolor='#262730', font_color="white", notebook=False)
    
    # Construcción de nodos y aristas
    for n, d in G.nodes(data=True):
        degree = G.degree(n)
        if degree >= min_degree_default:
            # 🚨 TAMAÑO: Factor 40 y Mínimo 18
            size = max(18, int(np.log(degree + 1) * 40)) 
            net.add_node(
                n, 
                label=n, 
                title=f"{n} (Grado: {degree}, Cluster: {d.get('cluster_id')})", 
                size=size,
                color=d.get('color'),
                group=d.get('cluster_id')
            )
    
    visible_nodes = [n for n, d in G.nodes(data=True) if G.degree(n) >= min_degree_default]
    for u, v, attrs in G.edges(data=True):
        if u in visible_nodes and v in visible_nodes:
             weight = attrs.get("weight", 1)
             net.add_edge(u, v, value=weight, title=f"Co-ocurrencia: {weight}")
             
    # Configuración de pyvis para movimiento natural
    net.force_atlas_2based()
    net.options.edges.smooth.enabled = True
    net.options.interaction.hover = True
    
    # 🚨 SOLUCIÓN DEFINITIVA DEL ERROR DE SINTAXIS JSON (USANDO REEMPLAZO FUERTE) 🚨
    options_dict = {
        "physics": {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -40000, 
                "centralGravity": 0.05,
                "springLength": 100,
                "springConstant": 0.05,
                "damping": 0.5,   # Movimiento natural
                "avoidOverlap": 0
            },
            "stabilization": {
                "enabled": True,
                "iterations": 3000, 
                "fit": True
            },
            "timestep": 0.5,
            "minVelocity": 0.001 
        }
    }
    
    # Convertimos el diccionario a una cadena JSON válida.
    json_part = json.dumps(options_dict)
    
    # Construimos la cadena JS, eliminando cualquier caracter invisible que pueda causar 'Extra data'.
    json_options_string = "var options = " + json_part + ";" 
    json_options_string = json_options_string.replace('\n', '').replace('\t', '').strip() # Limpieza Forzada

    net.set_options(json_options_string)
    
    # Guardar, leer y borrar el archivo temporal
    # Usamos os.path.dirname(os.path.abspath(__file__)) para asegurar la ruta del archivo temporal
    temp_filename = f"tmp_pyvis_{hash(tuple(selected_layers))}.html" 
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_filename)

    try:
        net.save_graph(tmp_path)
        html = open(tmp_path, "r", encoding="utf-8").read()
    except Exception as e:
        st.error(f"Error al guardar/leer el archivo temporal: {e}")
        return "Error de archivo.", nx.Graph()
    finally:
        # Aseguramos que el archivo se borre, incluso si hay un error de lectura.
        if os.path.exists(tmp_path):
            os.remove(tmp_path) 
    
    return html, G
# =======================================================
# 6. FUNCIÓN PRINCIPAL DEL DASHBOARD
# =======================================================

def render_x_dashboard():
    st.title("🗣️ Análisis de Sentimiento y Conversación de X")
    st.markdown("---")
    
    # A. Cargar y Preparar datos
    df_raw = load_data()
    if df_raw.empty:
        return

    df_historico = prepare_data(df_raw.copy()) 
    if df_historico.empty:
        return
    
    # B. Generar Pronóstico y concatenar
    try:
        df_pronostico = generate_forecast(df_historico.copy())
        df_viz_gauge = pd.concat([df_historico, df_pronostico], ignore_index=True)
    except Exception as e:
        st.warning(f"Error al generar/concatenar el pronóstico: {e}. Mostrando solo datos históricos.")
        df_viz_gauge = df_historico # Fallback

    # -------------------------------------
    # 1. VISUALIZACIONES DE TIEMPO
    # -------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔥 Histórico de Sentimiento")
        render_heatmap(df_historico.copy()) 
        
    with col2:
        st.subheader("🔮 Pronóstico (Próxima Hora)")
        render_gauge(df_viz_gauge) 
        
    st.markdown("---")
    
    # -------------------------------------
    # 2. GRAFO DE RELACIONES (CÓDIGO LIMPIO Y ESTABLE)
    # -------------------------------------
    st.header("🌐 Grafo de Relaciones y Temas")
    
    st.subheader("Configuración del Grafo")
    
    # === Botones de Checkbox para la estabilidad ===
    col_h, col_m, col_k = st.columns(3)
    selected_layers = []

    with col_h:
        show_hashtags = st.checkbox("Hashtags (#)", value=True, help="Activar/Desactivar la capa de hashtags.")
        if show_hashtags:
            selected_layers.append('Hashtags')

    with col_m:
        show_mentions = st.checkbox("Menciones (@)", value=True, help="Activar/Desactivar la capa de menciones (arrobamientos).")
        if show_mentions:
            selected_layers.append('Menciones (@)')

    with col_k:
        show_keywords = st.checkbox("Keywords", value=True, help="Activar/Desactivar la capa de palabras clave puras.")
        if show_keywords:
            selected_layers.append('Keywords')
    # ===============================================

    if not selected_layers:
        st.info("Por favor, selecciona al menos una capa de datos para construir el grafo.")
        return

    # 1. LLAMADA A LA FUNCIÓN (SIN CACHÉ, SIN COMA EXTRA)
    try:
        # Se asegura la indentación correcta de la llamada dentro del try
        html_content, G_full = render_interactive_graph(
            df_historico, 
            selected_layers 
        )
    except Exception as e:
        # Se asegura que el manejo de la excepción se haga correctamente
        st.error(f"Error fatal al generar el grafo: {e}")
        return

    # Si no se pudo generar el grafo (ej. si devuelve un mensaje de error o grafo vacío)
    if not isinstance(G_full, nx.Graph) or G_full.number_of_nodes() == 0:
        st.info("No hay suficientes datos o conexiones para mostrar el grafo con las capas seleccionadas.")
        return

    # 2. RENDERIZADO FINAL
    
    max_degree = max([G_full.degree(n) for n in G_full.nodes()]) if G_full.number_of_nodes() > 0 else 5
    
    st.write(f"Nodos: **{len(G_full.nodes())}**, Enlaces: **{len(G_full.edges())}**")

    # Slider de Grado (Ahora solo informativo, no recalcula el grafo)
    min_degree = st.slider(
        "Mínimo de conexiones (Grado) a mostrar:", 
        1, max(3, max_degree), 2, 
        help="Este filtro ayuda a visualizar solo los nodos más relevantes. Los nodos se calculan con un grado mínimo de 2 para el caché."
    )
    
    # El HTML se renderiza
    st.components.v1.html(html_content, height=600, scrolling=True)

# Ejecución 
if __name__ == "__main__":
    render_x_dashboard()