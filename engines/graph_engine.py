"""
graph_engine.py — Force Graph Network Visualization
Uses NetworkX + Plotly (Streamlit Cloud compatible)
No pyvis CDN dependency issues — pure Plotly rendering
"""
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from engines.data_engine import LAYER_CONFIG, SYMBOL_TO_LAYER, display_name

LAYER_COLORS = {layer: cfg["color"] for layer, cfg in LAYER_CONFIG.items()}

# ── Build Network Graph ────────────────────────────────────────────────────────

def build_correlation_graph(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3,
    centrality: pd.DataFrame = None,
    lead_lag: pd.DataFrame = None,
) -> nx.DiGraph:
    """
    Build directed weighted graph from correlation matrix.
    Edge direction from lead-lag if available.
    """
    G = nx.DiGraph()

    # Add nodes
    for sym in corr_matrix.columns:
        layer = SYMBOL_TO_LAYER.get(sym, "")
        # Reverse-lookup display name to original
        for orig, disp in {s: display_name(s) for s in SYMBOL_TO_LAYER}.items():
            if disp == sym:
                layer = SYMBOL_TO_LAYER.get(orig, layer)
                break

        size = 20
        if centrality is not None and not centrality.empty:
            row = centrality[centrality["symbol"] == sym]
            if not row.empty:
                size = 15 + row["eigenvector"].values[0] * 60

        is_tsla = sym == "TSLA"
        G.add_node(
            sym,
            layer=layer,
            color=LAYER_COLORS.get(layer, "#888888"),
            size=50 if is_tsla else size,
            is_tsla=is_tsla,
        )

    # Build lead-lag direction map
    direction_map = {}
    if lead_lag is not None and not lead_lag.empty:
        for _, row in lead_lag.iterrows():
            sym = row["symbol"]
            lag = row.get("best_lag_bars", 0)
            if lag > 0:
                direction_map[sym] = "leads"   # sym leads TSLA
            elif lag < 0:
                direction_map[sym] = "follows"

    # Add edges
    symbols = corr_matrix.columns.tolist()
    for i, sym_a in enumerate(symbols):
        for j, sym_b in enumerate(symbols):
            if i >= j:
                continue
            corr_val = corr_matrix.loc[sym_a, sym_b]
            if abs(corr_val) < threshold:
                continue

            # Determine edge direction
            dir_a = direction_map.get(sym_a, "coincident")
            dir_b = direction_map.get(sym_b, "coincident")

            if "TSLA" in (sym_a, sym_b):
                leader = sym_a if dir_a == "leads" else sym_b
                follower = sym_b if leader == sym_a else sym_a
                src, tgt = leader, follower
            else:
                src, tgt = sym_a, sym_b

            # Edge color: positive = teal, negative = red
            edge_color = "#00D4AA" if corr_val > 0 else "#FF4455"
            G.add_edge(
                src, tgt,
                weight=abs(corr_val),
                corr=corr_val,
                color=edge_color,
                width=1 + abs(corr_val) * 5,
            )

    return G


def _spring_layout_deterministic(G: nx.DiGraph, seed: int = 42) -> dict:
    """Spring layout with TSLA pinned at center."""
    pos = nx.spring_layout(G, k=2.5, seed=seed, weight="weight", iterations=80)

    # Force TSLA to center
    if "TSLA" in pos:
        center = np.array([0.0, 0.0])
        old_tsla = pos["TSLA"]
        shift = center - old_tsla
        # Shift everything relative
        for node in pos:
            pos[node] = pos[node] + shift

    return pos


def _layer_radial_layout(G: nx.DiGraph) -> dict:
    """Layered radial layout: TSLA center, layers in rings."""
    layer_radii = {"Mega Cap": 0.35, "ETF": 0.6, "Macro": 0.82, "Crypto": 1.0}
    pos = {}

    for layer, radius in layer_radii.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == layer]
        if not nodes:
            continue
        n = len(nodes)
        for i, node in enumerate(nodes):
            if node == "TSLA":
                pos[node] = np.array([0.0, 0.0])
            else:
                angle = 2 * np.pi * i / max(n, 1)
                pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    # Anything without layer
    for node in G.nodes():
        if node not in pos:
            pos[node] = np.array([np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
    return pos


# ── Plotly Force Graph ─────────────────────────────────────────────────────────

def render_force_graph(
    G: nx.DiGraph,
    layout: str = "spring",
    highlight_node: str = None,
    title: str = "TSLA Market Neural Network",
) -> go.Figure:
    """Render NetworkX graph as interactive Plotly figure."""

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No data available", template="plotly_dark")
        return fig

    # Compute layout
    if layout == "radial":
        pos = _layer_radial_layout(G)
    else:
        pos = _spring_layout_deterministic(G)

    # ── Edge traces (grouped by color for performance) ──────────────────
    edge_traces = []
    for (src, tgt, data) in G.edges(data=True):
        if src not in pos or tgt not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        color = data.get("color", "#444")
        width = data.get("width", 1)
        corr = data.get("corr", 0)
        is_highlighted = highlight_node and highlight_node in (src, tgt)

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(
                width=width * (1.8 if is_highlighted else 1.0),
                color=color if not is_highlighted else "#FFFFFF",
            ),
            opacity=0.85 if is_highlighted else 0.45,
            hoverinfo="text",
            hovertext=f"{src} ↔ {tgt}<br>Correlation: {corr:.3f}",
            showlegend=False,
        ))

    # ── Node trace ───────────────────────────────────────────────────────
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    node_hover = []
    node_symbols = []

    for node, data in G.nodes(data=True):
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        is_tsla = data.get("is_tsla", False)
        is_hl = highlight_node == node

        node_text.append(f"<b>{node}</b>" if (is_tsla or is_hl) else node)
        node_color.append("#E31937" if is_tsla else (
            "#FFFFFF" if is_hl else data.get("color", "#888888")
        ))
        node_size.append(data.get("size", 20) * (1.3 if is_hl else 1.0))
        node_symbols.append("star" if is_tsla else "circle")

        layer = data.get("layer", "Unknown")
        degree = G.degree(node)
        node_hover.append(
            f"<b>{node}</b><br>"
            f"Layer: {layer}<br>"
            f"Connections: {degree}<br>"
            f"{'⭐ SUPER NODE' if is_tsla else ''}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(
            family="IBM Plex Mono, monospace",
            size=10,
            color="#E8E0D0",
        ),
        marker=dict(
            color=node_color,
            size=node_size,
            symbol=node_symbols,
            line=dict(
                color=["#FFD700" if s == "star" else "#1A1A2E" for s in node_symbols],
                width=[3 if s == "star" else 1 for s in node_symbols],
            ),
            opacity=0.92,
        ),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    )

    # ── Legend traces ────────────────────────────────────────────────────
    legend_traces = []
    for layer, color_val in LAYER_COLORS.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color_val, size=12, symbol="circle"),
            name=layer,
            showlegend=True,
        ))

    # ── Assemble ─────────────────────────────────────────────────────────
    all_traces = edge_traces + [node_trace] + legend_traces

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="IBM Plex Mono", size=18, color="#E8E0D0"),
            x=0.5,
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="#E8E0D0", size=11),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#333",
            borderwidth=1,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0A0E1A",
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="closest",
        height=620,
    )

    return fig


# ── Correlation Heatmap ────────────────────────────────────────────────────────

def render_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Professional dark correlation heatmap."""
    if corr_matrix.empty:
        return go.Figure()

    labels = corr_matrix.columns.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#FF1744"],
            [0.5, "#1A1A2E"],
            [1.0, "#00E5FF"],
        ],
        zmin=-1, zmax=1,
        hoverongaps=False,
        hovertemplate="%{y} ↔ %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color="#E8E0D0"),
            title=dict(text="ρ", font=dict(color="#E8E0D0")),
        ),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Dynamic Correlation Matrix</b>",
            font=dict(family="IBM Plex Mono", size=15, color="#E8E0D0"),
            x=0.5,
        ),
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0A0E1A",
        xaxis=dict(
            tickfont=dict(color="#E8E0D0", size=9),
            tickangle=45,
        ),
        yaxis=dict(tickfont=dict(color="#E8E0D0", size=9)),
        margin=dict(l=80, r=20, t=60, b=80),
        height=480,
    )

    return fig


# ── Contagion Sankey ───────────────────────────────────────────────────────────

def render_contagion_chart(contagion_df: pd.DataFrame) -> go.Figure:
    """Bar chart showing TSLA shock transmission."""
    if contagion_df.empty:
        return go.Figure()

    df = contagion_df.head(15).sort_values("expected_impact_pct")
    colors = ["#FF4444" if v < 0 else "#00D4AA" for v in df["expected_impact_pct"]]

    fig = go.Figure(go.Bar(
        x=df["expected_impact_pct"],
        y=df["symbol"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        text=[f"{v:+.2f}%" for v in df["expected_impact_pct"]],
        textposition="outside",
        textfont=dict(color="#E8E0D0", size=9, family="IBM Plex Mono"),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Risk Contagion: TSLA -5% Shock Impact</b>",
            font=dict(family="IBM Plex Mono", size=14, color="#E8E0D0"),
            x=0.5,
        ),
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0A0E1A",
        xaxis=dict(
            title="Expected Impact (%)",
            titlefont=dict(color="#888"),
            tickfont=dict(color="#E8E0D0", size=9),
            gridcolor="#1E2A3A",
            zerolinecolor="#333",
        ),
        yaxis=dict(tickfont=dict(color="#E8E0D0", size=9)),
        margin=dict(l=60, r=80, t=60, b=40),
        height=420,
    )
    return fig
