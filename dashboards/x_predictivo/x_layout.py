import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from community import community_louvain

# ==============================
# FUNCIÓN PRINCIPAL
# ==============================

def show_graph_layout(csv_path):
    st.title("Mapa de hashtags, menciones y palabras")

    # --- Cargar datos ---
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return

    # --- Construir grafo ---
    G = nx.Graph()

    for _, row in df.iterrows():
        palabras = str(row.get("keywords", "")).split()
        for palabra in palabras:
            if palabra not in G:
                G.add_node(palabra)
            for otra in palabras:
                if palabra != otra:
                    G.add_edge(palabra, otra)

    # --- Calcular comunidades con Louvain ---
    if len(G.nodes) > 0:
        partition = community_louvain.best_partition(G)
        nx.set_node_attributes(G, partition, 'community')
    else:
        st.warning("No se encontraron nodos en el grafo.")
        return

    # --- Filtro: grado mínimo de conexión ---
    st.subheader("Filtrar hashtags, menciones o palabras poco conectadas")

    min_degree = st.slider(
        "Selecciona el nivel mínimo de conexión:",
        min_value=1,
        max_value=10,
        value=2,
        help="Ajusta cuántas conexiones debe tener un hashtag, mención o palabra para mostrarse. "
             "Los valores bajos muestran más elementos; los altos muestran solo los más conectados."
    )

    # --- Filtrar nodos por grado ---
    nodes_to_keep = [n for n, d in dict(G.degree()).items() if d >= min_degree]
    G_filtered = G.subgraph(nodes_to_keep).copy()

    st.caption(
        "Este grafo muestra solo los hashtags, menciones o palabras con más conexiones para facilitar la visualización. "
        "Al reducir el filtro, se incluirán los que tienen menos relación con los demás."
    )

    # --- Mensaje si se ocultan elementos ---
    if len(G_filtered) < len(G):
        st.info(
            f"Se muestran {len(G_filtered)} de {len(G)} elementos totales. "
            "Algunos hashtags, menciones o palabras con pocas conexiones no aparecen."
        )

    # --- Crear visualización interactiva con PyVis ---
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black")

    # Definir colores por comunidad
    communities = set(nx.get_node_attributes(G_filtered, 'community').values())
    palette = [
        "#E4572E", "#76B041", "#4C78A8", "#F2AF29", "#B279A2", "#FF8C00", "#54A24B",
        "#9467BD", "#17BECF", "#D62728"
    ]

    color_map = {
        c: palette[i % len(palette)] for i, c in enumerate(sorted(communities))
    }

    for node, data in G_filtered.nodes(data=True):
        net.add_node(
            node,
            label=node,
            title=f"{node} — Conexiones: {G_filtered.degree(node)}",
            color=color_map.get(data.get("community"), "#CCCCCC")
        )

    for source, target in G_filtered.edges():
        net.add_edge(source, target)

    # --- Exportar y mostrar ---
    net.repulsion(
        node_distance=120,
        central_gravity=0.33,
        spring_length=120,
        spring_strength=0.10,
        damping=0.95
    )

    html_path = "graph.html"
    net.save_graph(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=680, scrolling=True)
