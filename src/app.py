"""Streamlit app for transaction network visualization."""

from datetime import timedelta
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Lazy GPU imports (RAPIDS cuGraph for Linux/WSL GPU support)
cudf = None
cugraph = None
nxcg = None
cuml_kmeans = None
GPU_AVAILABLE = None


def check_gpu_available():
    """Lazy check for GPU availability using RAPIDS cuGraph."""
    global cudf, cugraph, nxcg, cuml_kmeans, GPU_AVAILABLE
    if GPU_AVAILABLE is None:
        try:
            import cudf as _cudf
            import cugraph as _cugraph
            import nx_cugraph as _nxcg
            from cuml.cluster import KMeans as _KMeans

            cudf = _cudf
            cugraph = _cugraph
            nxcg = _nxcg
            cuml_kmeans = _KMeans
            GPU_AVAILABLE = True
        except (ImportError, Exception):
            GPU_AVAILABLE = False
    return GPU_AVAILABLE


# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
TRANSACTIONS_DIR = DATA_DIR / "transactions"

TIME_FILTERS = {
    "1 mois": 30,
    "2 mois": 60,
    "3 mois": 90,
    "6 mois": 180,
    "1 an": 365,
    "Tout": None,
}

DEGREE_FILTERS = {
    "Tous": (0, 999999),
    "1-2 liens": (1, 2),
    "2-5 liens": (2, 5),
    "5-10 liens": (5, 10),
    "10-20 liens": (10, 20),
    "20-50 liens": (20, 50),
    "50-100 liens": (50, 100),
    ">100 liens": (100, 999999),
}


@st.cache_data
def load_transactions() -> pd.DataFrame:
    """Load all transactions."""
    first_batch = TRANSACTIONS_DIR / "transactions_batch_000.parquet"
    df = pd.read_parquet(first_batch)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def filter_by_period(df: pd.DataFrame, days) -> pd.DataFrame:
    """Filter transactions by time period."""
    if days is None:
        return df
    max_date = df["transaction_date"].max()
    min_date = max_date - timedelta(days=days)
    return df[df["transaction_date"] >= min_date]


def aggregate_by_client(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions at client pair level."""
    agg = (
        df.groupby(["emetteur_rib", "destinataire_rib"])
        .agg(
            montant_total=("montant", "sum"),
            nb_transactions=("transaction_id", "count"),
        )
        .reset_index()
    )
    return agg


def create_networkx_graph(agg_df: pd.DataFrame) -> nx.DiGraph:
    """Create NetworkX DiGraph."""
    G = nx.DiGraph()
    for _, row in agg_df.iterrows():
        G.add_edge(
            row["emetteur_rib"],
            row["destinataire_rib"],
            weight=row["montant_total"],
            count=row["nb_transactions"],
        )
    return G


def detect_clusters_networkx(G: nx.DiGraph, n_clusters: int = 3) -> dict:
    """Detect clusters using K-means on combined node features (CPU)."""
    if len(G.nodes()) == 0:
        return {}
    node_features = {}
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes()}
    clustering_coef = nx.clustering(G.to_undirected())
    pagerank = nx.pagerank(G)

    for node in G.nodes():
        out_edges = G.out_edges(node, data=True)
        in_edges = G.in_edges(node, data=True)
        montant_emis = sum(d.get("weight", 0) for _, _, d in out_edges)
        nb_tx_emis = sum(d.get("count", 0) for _, _, d in out_edges)
        montant_recu = sum(d.get("weight", 0) for _, _, d in in_edges)
        nb_tx_recu = sum(d.get("count", 0) for _, _, d in in_edges)
        node_features[node] = [
            total_deg[node],
            clustering_coef.get(node, 0),
            pagerank.get(node, 0),
            montant_emis,
            montant_recu,
            nb_tx_emis,
            nb_tx_recu,
        ]
    nodes = list(node_features.keys())
    X = pd.DataFrame(
        [node_features[n] for n in nodes],
        columns=[
            "degree",
            "clustering",
            "pagerank",
            "montant_emis",
            "montant_recu",
            "nb_tx_emis",
            "nb_tx_recu",
        ],
        index=nodes,
    )
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=min(n_clusters, len(G.nodes())), random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return {node: int(label) for node, label in zip(nodes, labels)}


def detect_clusters_gpu(G: nx.DiGraph, n_clusters: int = 3) -> dict:
    """Detect clusters using GPU-accelerated KMeans with RAPIDS cuML."""
    global cudf, cuml_kmeans, nxcg
    if cudf is None or cuml_kmeans is None or len(G.nodes()) == 0:
        return {}

    # Use nx_cugraph for GPU-accelerated graph metrics if available
    try:
        G_undirected = G.to_undirected()
        # GPU-accelerated PageRank via nx_cugraph backend
        pagerank = nx.pagerank(G, backend="cugraph")
        clustering_coef = nx.clustering(G_undirected, backend="cugraph")
    except Exception:
        # Fallback to CPU for graph metrics
        pagerank = nx.pagerank(G)
        clustering_coef = nx.clustering(G.to_undirected())

    # Build node features
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes()}

    nodes = list(G.nodes())
    features = []
    for node in nodes:
        out_edges = G.out_edges(node, data=True)
        in_edges = G.in_edges(node, data=True)
        montant_emis = sum(d.get("weight", 0) for _, _, d in out_edges)
        nb_tx_emis = sum(d.get("count", 0) for _, _, d in out_edges)
        montant_recu = sum(d.get("weight", 0) for _, _, d in in_edges)
        nb_tx_recu = sum(d.get("count", 0) for _, _, d in in_edges)
        features.append(
            [
                total_deg[node],
                clustering_coef.get(node, 0),
                pagerank.get(node, 0),
                montant_emis,
                montant_recu,
                nb_tx_emis,
                nb_tx_recu,
            ]
        )

    # GPU-accelerated standardization and KMeans with cuDF/cuML
    df = cudf.DataFrame(
        features,
        columns=[
            "degree",
            "clustering",
            "pagerank",
            "montant_emis",
            "montant_recu",
            "nb_tx_emis",
            "nb_tx_recu",
        ],
    )

    # Standardize on GPU
    mean = df.mean()
    std = df.std() + 1e-8
    df_scaled = (df - mean) / std

    # cuML KMeans clustering on GPU
    n_clusters = min(n_clusters, len(nodes))
    kmeans = cuml_kmeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df_scaled)

    # Convert cuDF Series to list
    labels_list = labels.to_pandas().tolist()

    return {node: int(label) for node, label in zip(nodes, labels_list)}


def filter_by_degree(G: nx.DiGraph, min_deg: int, max_deg: int) -> nx.DiGraph:
    G_undirected = G.to_undirected()
    degrees = dict(G_undirected.degree())
    nodes_to_keep = [n for n, d in degrees.items() if min_deg <= d <= max_deg]
    return G.subgraph(nodes_to_keep).copy()


def visualize_graph_plotly(G: nx.DiGraph, cluster_map: dict) -> go.Figure:
    """Plotly visualization (identique pour CPU/GPU)."""
    if len(G.nodes()) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucun client à afficher",
            showarrow=False,
            font=dict(size=20, color="white"),
        )
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        return fig

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    total_degree = {n: G.degree(n) for n in G.nodes()}

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scattergl(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="skip",
        mode="lines",
        opacity=0.3,
    )

    node_x, node_y, node_colors, node_sizes, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        cluster = cluster_map.get(node, 0)
        node_colors.append(cluster)
        degree = total_degree[node]
        node_sizes.append(5 + min(degree * 2, 30))

        out_edges = G.out_edges(node, data=True)
        in_edges = G.in_edges(node, data=True)
        montant_emis = sum(d.get("weight", 0) for _, _, d in out_edges)
        nb_tx_emis = sum(d.get("count", 0) for _, _, d in out_edges)
        montant_recu = sum(d.get("weight", 0) for _, _, d in in_edges)
        nb_tx_recu = sum(d.get("count", 0) for _, _, d in in_edges)
        solde = montant_recu - montant_emis
        solde_str = f"+{solde:,.0f}" if solde >= 0 else f"{solde:,.0f}"

        node_text.append(
            f"<b>RIB:</b> {node}<br>"
            f"<b>Cluster:</b> {cluster}<br>"
            f"<b>Liens:</b> {degree}<br>---<br>"
            f"<b>Émis:</b> {montant_emis:,.0f} € ({nb_tx_emis} tx)<br>"
            f"<b>Reçu:</b> {montant_recu:,.0f} € ({nb_tx_recu} tx)<br>"
            f"<b>Solde:</b> {solde_str} €"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertemplate="%{text}",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_colors,
            size=node_sizes,
            colorbar=dict(thickness=15, title="Cluster", xanchor="left"),
            line=dict(width=1, color="#fff"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Réseau de Transactions",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
        ),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Transaction Network", page_icon="🔗", layout="wide")
    st.title("🔗 Transaction Network Visualization")

    st.sidebar.markdown("---")
    engine_options = ["CPU", "GPU (RAPIDS cuGraph)"]
    cluster_engine = st.sidebar.radio("Moteur clustering", engine_options, index=0)

    # Only check GPU availability when GPU mode is selected
    use_gpu = False
    if cluster_engine == "GPU (RAPIDS cuGraph)":
        if check_gpu_available():
            use_gpu = True
        else:
            st.sidebar.error("RAPIDS non disponible, utilisation CPU")
            cluster_engine = "CPU"

    st.sidebar.header("Filtres")
    time_period = st.sidebar.selectbox("Période", list(TIME_FILTERS.keys()), index=5)
    days = TIME_FILTERS[time_period]

    with st.spinner("Chargement..."):
        df_raw = load_transactions()
        agg_df_full = aggregate_by_client(df_raw)
        G_full = create_networkx_graph(agg_df_full)

        max_clusters = st.sidebar.slider(
            "Nb clusters max", min_value=2, max_value=3, value=3
        )

        # Clustering selon moteur choisi
        if use_gpu:
            try:
                cluster_map = detect_clusters_gpu(G_full, n_clusters=max_clusters)
                st.sidebar.success("✓ GPU clustering (RAPIDS cuGraph/cuML)")
            except Exception as e:
                st.sidebar.error(f"GPU error: {e}")
                cluster_map = detect_clusters_networkx(G_full, n_clusters=max_clusters)
        else:
            cluster_map = detect_clusters_networkx(G_full, n_clusters=max_clusters)

        # Filtrer par période
        df_filtered = filter_by_period(df_raw, days)
        agg_df_filtered = aggregate_by_client(df_filtered)
        G = create_networkx_graph(agg_df_filtered)

    # Nombre de clients
    st.sidebar.markdown("---")
    max_clients = st.sidebar.slider(
        "Nombre de clients max",
        min_value=50,
        max_value=min(5000, len(G_full.nodes())),
        value=min(500, len(G_full.nodes())),
        step=50,
    )

    # Degree filter
    degree_filter = st.sidebar.selectbox(
        "Filtrer par nb de liens", list(DEGREE_FILTERS.keys())
    )
    min_deg, max_deg = DEGREE_FILTERS[degree_filter]
    G = filter_by_degree(G, min_deg, max_deg)

    # Limiter le nombre de clients
    if len(G.nodes()) > max_clients:
        degrees = dict(G.to_undirected().degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[
            :max_clients
        ]
        G = G.subgraph(top_nodes).copy()

    # Cluster filter
    unique_clusters = sorted(set(cluster_map.values()))
    cluster_options = ["Tous"] + [f"Cluster {c}" for c in unique_clusters]
    selected_cluster = st.sidebar.selectbox("Filtrer cluster", cluster_options)
    if selected_cluster != "Tous":
        cluster_id = int(selected_cluster.split()[-1])
        nodes_in_cluster = [n for n, c in cluster_map.items() if c == cluster_id]
        G = G.subgraph(nodes_in_cluster).copy()

    # Recherche
    st.sidebar.markdown("---")
    search_rib = st.sidebar.text_input("Rechercher un RIB", "")
    if search_rib:
        matching = [r for r in G.nodes() if search_rib.upper() in r.upper()]
        if matching:
            st.sidebar.success(f"{len(matching)} résultat(s)")
            neighbors = set()
            for rib in matching[:5]:
                neighbors.add(rib)
                neighbors.update(G.predecessors(rib))
                neighbors.update(G.successors(rib))
            G = G.subgraph(neighbors).copy()
        else:
            st.sidebar.warning("Aucun résultat")

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{len(df_filtered):,}")
    col2.metric("Montant total", f"{df_filtered['montant'].sum():,.0f} €")
    col3.metric("Clients affichés", f"{len(G.nodes()):,}")
    col4.metric("Relations", f"{len(G.edges()):,}")

    # Visualize
    with st.spinner("Génération du graphe..."):
        fig = visualize_graph_plotly(G, cluster_map)
        st.plotly_chart(fig, use_container_width=True)

    # Tableau relations
    st.subheader("Relations (filtrées)")
    filtered_edges = []
    for u, v, data in G.edges(data=True):
        filtered_edges.append(
            {
                "Émetteur": u,
                "Destinataire": v,
                "Montant (€)": data.get("weight", 0),
                "Nb transactions": data.get("count", 0),
            }
        )
    if filtered_edges:
        df_edges = pd.DataFrame(filtered_edges).sort_values(
            "Montant (€)", ascending=False
        )
        df_edges["Montant (€)"] = df_edges["Montant (€)"].apply(lambda x: f"{x:,.2f}")
        st.dataframe(df_edges, height=500, width=2000)
        st.caption(f"{len(df_edges)} relations affichées")
    else:
        st.info("Aucune relation à afficher avec les filtres actuels")


if __name__ == "__main__":
    main()
