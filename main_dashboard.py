# main_dashboard.py
import streamlit as st
from dashboards.x_predictivo.x_layout import load_data, prepare_data, render_heatmap, render_gauge, render_interactive_graph
from dashboards.facebook.facebook_layout import render_facebook_dashboard
from dashboards.benchmark.benchmark_layout import render_benchmark_dashboard
from dashboards.telegram.telegram_layout import render_telegram_dashboard

# ------------------------
# CONFIGURACIÓN GENERAL
# ------------------------
st.set_page_config(
    page_title="Dashboard Analítico",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🗣️"
)

# Modo oscuro consistente
st.markdown(
    """
    <style>
    .main { background-color: #262730; color: white; }
    .st-bk { background-color: #262730; }
    .css-1d391kg { color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------
# SIDEBAR
# ------------------------
st.sidebar.title(" Navegación")
dashboard_option = st.sidebar.radio(
    "Selecciona un dashboard:",
    ("X", "Facebook", "Benchmark", "Alertas")
)

st.sidebar.markdown("---")
st.sidebar.write("Filtrado y configuración del grafo y visualizaciones se aplican solo en 'X'.")

# ------------------------
# RENDER PRINCIPAL
# ------------------------
if dashboard_option == "X":
    st.title(" Análisis de Sentimiento y Conversación de X")
    st.markdown("---")

    # Cargar y preparar datos
    df_raw = load_data()
    if df_raw.empty:
        st.info("No hay datos cargados.")
        st.stop()
    df_historico = prepare_data(df_raw)
    if df_historico.empty:
        st.info("No hay datos válidos después de la limpieza.")
        st.stop()

    # Visualizaciones superiores: Heatmap + Gauge
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Histórico de Sentimiento")
        render_heatmap(df_historico)
    with col2:
        st.subheader("Pronóstico (Promedio de Sentimiento)")
        render_gauge(df_historico)

    st.markdown("---")

    # Grafo interactivo
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

    html_content, G_full = render_interactive_graph(df_historico, selected_layers)

    if isinstance(html_content, str) and html_content.startswith("Error"):
        st.error(html_content)
    elif G_full is None or G_full.number_of_nodes() == 0:
        st.info("No hay suficientes nodos para mostrar con la selección actual.")
    else:
        st.write(f"Nodos: **{len(G_full.nodes())}**, Enlaces: **{len(G_full.edges())}**")
        st.components.v1.html(html_content, height=700, scrolling=True)

# ------------------------
# Otros dashboards
# ------------------------
elif dashboard_option == "Facebook":
    render_facebook_dashboard()
elif dashboard_option == "Benchmark":
    render_benchmark_dashboard()
elif dashboard_option == "Alertas":
    render_telegram_dashboard()
