"""
app.py — TSLA Market Intelligence Force Graph System
Tesla Market Brain · Hedge Fund Grade UI
Streamlit Cloud Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import sys
import os

# ── Path Setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Page Config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Tesla Market Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from ui.styles import inject_css, metric_card, regime_badge, panel_header, tweet_card, status_bar
from engines.data_engine import (
    fetch_price_data, get_returns,
    get_rolling_volatility, LAYER_CONFIG, display_name, ALL_SYMBOLS
)
from engines.correlation_engine import (
    compute_rolling_correlation, correlation_timeseries,
    compute_lead_lag, compute_granger_scores,
    compute_hidden_relationships, compute_centrality,
    compute_risk_contagion
)
from engines.regime_engine import (
    detect_regime_kmeans, compute_volatility_clusters,
    detect_correlation_breakdown, REGIME_LABELS
)
from engines.graph_engine import (
    build_correlation_graph, render_force_graph,
    render_correlation_heatmap, render_contagion_chart
)
from engines.ai_brain import (
    generate_market_brain, analyze_narrative_impact,
    score_sentiment_simple, build_market_context,
    generate_prompt, get_prompt_for_ai,
    PROMPT_TYPES, TARGET_AIS, call_ai, SYSTEM_ROLES,
)
from engines.news_engine import (
    fetch_news_feed, render_news_feed, get_simulated_tweets
)
from engines.holdings_engine import (
    fetch_tsla_institutional_holders, fetch_tsla_mutualfund_holders,
    compute_qoq_changes, fetch_ark_tsla_position,
    get_ownership_summary, TOP_FUNDS, FUND_COLORS, FUND_TYPES,
)

inject_css()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Tesla Market Brain</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">實時金融神經網絡 · TSLA 超級核心智能</div>',
    unsafe_allow_html=True,
)
status_bar(datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\';font-size:0.6rem;color:#334455;'
        'letter-spacing:0.2em;text-transform:uppercase;margin-bottom:16px">'
        '◈ System Controls</div>',
        unsafe_allow_html=True,
    )

    data_period = st.selectbox(
        "數據週期",
        ["1mo", "3mo", "6mo", "1y"],
        index=1,
    )

    corr_window = st.slider("相關性窗口（天）", 5, 60, 20, 5)
    corr_threshold = st.slider("邊閾值 (|ρ|)", 0.1, 0.8, 0.35, 0.05)
    graph_layout = st.radio("圖形佈局", ["spring", "radial"], horizontal=True)

    st.markdown("---")

    highlight_node = st.selectbox(
        "高亮節點",
        ["None", "TSLA", "NVDA", "META", "QQQ", "VIX", "BTC"],
    )
    if highlight_node == "None":
        highlight_node = None

    st.markdown("---")

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\';font-size:0.6rem;color:#334455;'
        'letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px">'
        '◈ AI Brain</div>',
        unsafe_allow_html=True,
    )
    run_ai = st.button("🧠 生成市場分析", use_container_width=True)
    shock_size = st.slider("TSLA 衝擊幅度 (%)", -20, -1, -5, 1)

    st.markdown("---")

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\';font-size:0.6rem;color:#334455;'
        'letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px">'
        '◈ Narrative Analyzer</div>',
        unsafe_allow_html=True,
    )
    tweet_input = st.text_area("分析推文 / 新聞標題", height=80,
                                placeholder="貼上 Elon 推文或新聞標題...")
    analyze_tweet = st.button("⚡ 分析影響", use_container_width=True)

    st.markdown("---")
    auto_refresh = st.checkbox("自動刷新（5分鐘）", value=False)
    if auto_refresh:
        time.sleep(1)
        st.rerun()


# ── Data Loading ──────────────────────────────────────────────────────────────
with st.spinner("⚡ 載入市場數據..."):
    close_df = fetch_price_data(period=data_period)
    # ── Debug info (shows on Streamlit Cloud to diagnose data issues) ──────
    if close_df.empty:
        st.error("❌ 數據載入失敗 — Yahoo Finance 暫時無法連線。請稍後重試或刷新頁面。")
    else:
        n_cols = len(close_df.columns)
        has_tsla = "TSLA" in close_df.columns
        if not has_tsla:
            st.warning(f"⚠️ 數據已載入 ({n_cols} 欄) 但缺少 TSLA。現有欄位: {list(close_df.columns[:5])}")
    returns = get_returns(close_df)
    corr_matrix = compute_rolling_correlation(returns, window=corr_window)

# ── Compute Analytics ─────────────────────────────────────────────────────────
with st.spinner("🔬 計算神經網絡..."):
    centrality = compute_centrality(corr_matrix, threshold=corr_threshold)
    lead_lag = compute_lead_lag(returns, target="TSLA", max_lag=5)
    hidden_rel = compute_hidden_relationships(returns, target="TSLA")
    contagion = compute_risk_contagion(returns, "TSLA", shock_size / 100)
    regime = detect_regime_kmeans(returns)
    breakdown = detect_correlation_breakdown(returns)
    vol_clusters = compute_volatility_clusters(returns)


# ── TOP METRICS ROW ───────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4, col5, col6 = st.columns(6)

# TSLA price/change
if not close_df.empty and "TSLA" in close_df.columns and close_df["TSLA"].dropna().shape[0] >= 2:
    _tsla_s = close_df["TSLA"].dropna()
    tsla_latest = float(_tsla_s.iloc[-1])
    tsla_prev   = float(_tsla_s.iloc[-2])
    tsla_chg    = (tsla_latest - tsla_prev) / tsla_prev * 100 if tsla_prev != 0 else 0.0
    tsla_vol_20  = get_rolling_volatility(close_df).dropna()
    tsla_vol_val = float(tsla_vol_20.iloc[-1]) * 100 if not tsla_vol_20.empty else 0.0
else:
    tsla_latest, tsla_chg, tsla_vol_val = 0.0, 0.0, 0.0

with col1:
    color = "#00FF88" if tsla_chg >= 0 else "#FF4444"
    metric_card("TSLA 股價", f"${tsla_latest:.2f}",
                f"{'▲' if tsla_chg >= 0 else '▼'} {abs(tsla_chg):.2f}%", color)

with col2:
    metric_card("TSLA 波動率（年化）", f"{tsla_vol_val:.1f}%",
                "20日滾動", "#FFB800")

with col3:
    regime_name = regime.get("current_name", "UNKNOWN")
    metric_card("市場狀態", regime_name[:12], f"Conf: {regime.get('confidence', 0):.0%}")

with col4:
    # Top leader
    if not lead_lag.empty:
        leaders = lead_lag[lead_lag["direction"] == "leads TSLA"]
        top_leader = leaders.iloc[0]["symbol"] if not leaders.empty else "—"
        leader_score = leaders.iloc[0]["influence_score"] if not leaders.empty else 0
        metric_card("TSLA 領先指標", top_leader, f"Score: {leader_score:.1f}", "#00D4FF")
    else:
        metric_card("TSLA 領先指標", "—", "計算中...")

with col5:
    if not centrality.empty:
        top_central = centrality.iloc[0]["symbol"]
        metric_card("市場大腦", top_central, "最高中心度", "#FF6B35")
    else:
        metric_card("市場大腦", "—", "—")

with col6:
    if not corr_matrix.empty and "TSLA" in corr_matrix.columns:
        n_strong = max(0, int((corr_matrix["TSLA"].abs() > corr_threshold).sum()) - 1)
        metric_card("TSLA 活躍連結", str(n_strong), f"Threshold ρ>{corr_threshold}", "#E31937")
    else:
        metric_card("TSLA 活躍連結", "—", "—")

st.markdown("---")


# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🕸️  力導向圖",
    "📊  相關性",
    "⚡  領先滯後",
    "🌊  風險傳染",
    "🧠  AI 大腦",
    "📰  敘事分析",
    "🏦  機構持倉",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORCE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c_left, c_right = st.columns([3, 1])

    with c_left:
        panel_header("TSLA Market Neural Network — Live Force Graph")

        G = build_correlation_graph(
            corr_matrix,
            threshold=corr_threshold,
            centrality=centrality,
            lead_lag=lead_lag,
        )

        fig_graph = render_force_graph(
            G,
            layout=graph_layout,
            highlight_node=highlight_node,
            title=f"TSLA Neural Network  |  ρ > {corr_threshold}  |  {G.number_of_edges()} active edges",
        )
        st.plotly_chart(fig_graph, use_container_width=True, key="pc_graph")

        # Edge stats below graph
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\';font-size:0.68rem;color:#445566;'
            f'text-align:center">Nodes: {G.number_of_nodes()} · '
            f'Edges: {G.number_of_edges()} · '
            f'TSLA Degree: {G.degree("TSLA") if "TSLA" in G else "N/A"}</div>',
            unsafe_allow_html=True,
        )

    with c_right:
        panel_header("市場狀態")
        regime_badge(regime.get("current_name", "UNKNOWN"), regime.get("current_color", "#888"))

        st.markdown("<br>", unsafe_allow_html=True)
        panel_header("中心度排名")
        if not centrality.empty:
            for _, row in centrality.head(8).iterrows():
                bar_w = int(row["eigenvector"] * 100)
                is_tsla = row["symbol"] == "TSLA"
                bar_color = "#E31937" if is_tsla else "#2A4A6A"
                st.markdown(
                    f"""<div style="margin-bottom:5px">
                        <div style="font-family:'IBM Plex Mono';font-size:9px;
                                    color:{'#E31937' if is_tsla else '#C8C0B0'};
                                    margin-bottom:2px">
                            {'⭐ ' if is_tsla else ''}{row['symbol']}
                        </div>
                        <div style="background:#0D1626;border-radius:2px;height:4px">
                            <div style="background:{bar_color};width:{bar_w}%;height:4px;border-radius:2px"></div>
                        </div>
                        <div style="font-family:'IBM Plex Mono';font-size:8px;color:#445566">
                            eigen: {row['eigenvector']:.3f} · deg: {int(row['degree'])}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        panel_header("層級圖例")
        for layer, cfg in LAYER_CONFIG.items():
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\';font-size:9px;margin-bottom:4px">'
                f'<span style="color:{cfg["color"]}">■</span> {layer}: '
                f'{", ".join(cfg["symbols"][:3])}{"..." if len(cfg["symbols"]) > 3 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        panel_header("相關性異常警報")
        if not breakdown.empty:
            breaking = breakdown[breakdown["signal"].str.contains("Breaking|Strengthen")]
            if not breaking.empty:
                for _, row in breaking.head(5).iterrows():
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\';font-size:9px;'
                        f'color:#FF8888;margin-bottom:3px">{row["signal"]} {row["symbol"]}<br>'
                        f'<span style="color:#445566">Δρ: {row["delta"]:+.3f}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="font-family:\'IBM Plex Mono\';font-size:9px;color:#445566">'
                    'All correlations stable</div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_h, col_ts = st.columns([1, 1])

    with col_h:
        panel_header("相關性矩陣")
        fig_heat = render_correlation_heatmap(corr_matrix)
        st.plotly_chart(fig_heat, use_container_width=True, key="pc_heat")

    with col_ts:
        panel_header("TSLA 相關性時序")
        corr_ts = correlation_timeseries(returns, "TSLA", window=corr_window)
        if not corr_ts.empty:
            top_assets = ["NVDA", "QQQ", "SPY", "ARKK", "BTC", "VIX"]
            plot_assets = [a for a in top_assets if a in corr_ts.columns][:6]

            fig_ts = go.Figure()
            colors_ts = ["#00D4FF", "#00FF88", "#FFB800", "#FF6B35", "#FF8C00", "#E31937"]
            for i, asset in enumerate(plot_assets):
                fig_ts.add_trace(go.Scatter(
                    x=corr_ts.index,
                    y=corr_ts[asset],
                    name=asset,
                    line=dict(color=colors_ts[i % len(colors_ts)], width=1.5),
                    mode="lines",
                ))

            fig_ts.add_hline(y=0, line_dash="dot", line_color="#334455")
            fig_ts.add_hline(y=corr_threshold, line_dash="dash",
                              line_color="#556677", annotation_text=f"ρ={corr_threshold}")
            fig_ts.add_hline(y=-corr_threshold, line_dash="dash", line_color="#556677")

            fig_ts.update_layout(
                title=dict(
                    text=f"<b>滾動 {corr_window} 日相關性 vs TSLA</b>",
                    font=dict(family="IBM Plex Mono", size=13, color="#E8E0D0"),
                    x=0.5,
                ),
                paper_bgcolor="#0A0E1A",
                plot_bgcolor="#0A0E1A",
                xaxis=dict(gridcolor="#1A2A3A", tickfont=dict(color="#E8E0D0", size=9)),
                yaxis=dict(
                    gridcolor="#1A2A3A",
                    tickfont=dict(color="#E8E0D0", size=9),
                    range=[-1.1, 1.1],
                    title="ρ",
                    titlefont=dict(color="#556677"),
                ),
                legend=dict(font=dict(color="#E8E0D0", size=9), bgcolor="rgba(0,0,0,0.5)"),
                height=380,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_ts, use_container_width=True, key="pc_ts")
        else:
            st.info("數據不足，無法生成時序圖。")

    # Hidden Relationships
    panel_header("隱藏非線性關係（互信息）")
    if not hidden_rel.empty:
        top_hidden = hidden_rel.head(12)
        cols_hr = st.columns(4)
        for i, (_, row) in enumerate(top_hidden.iterrows()):
            with cols_hr[i % 4]:
                hs = row["hidden_score"]
                bar_color = "#FF6B35" if hs > 0.1 else "#2A4A6A"
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">{row['symbol']}</div>
                        <div class="metric-value" style="font-size:1rem">{row['relationship_type']}</div>
                        <div class="metric-delta" style="color:#888">
                            MI: {row['mutual_info']:.3f} · ρ: {row['linear_corr']:.3f}
                        </div>
                        <div style="background:#0A1020;border-radius:2px;height:3px;margin-top:6px">
                            <div style="background:{bar_color};width:{min(int(hs*500),100)}%;height:3px;border-radius:2px"></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LEAD-LAG
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_ll, col_gr = st.columns([1, 1])

    with col_ll:
        panel_header("領先滯後分析（交叉相關）")
        if not lead_lag.empty:
            fig_ll = go.Figure()

            leaders = lead_lag[lead_lag["direction"] == "leads TSLA"].head(8)
            followers = lead_lag[lead_lag["direction"] == "follows TSLA"].head(8)

            if not leaders.empty:
                fig_ll.add_trace(go.Bar(
                    x=leaders["influence_score"],
                    y=leaders["symbol"],
                    orientation="h",
                    name="Leads TSLA",
                    marker_color="#00D4FF",
                    hovertemplate="%{y}: score=%{x:.1f}, lag=%{customdata}bars<extra></extra>",
                    customdata=leaders["best_lag_bars"],
                ))

            if not followers.empty:
                fig_ll.add_trace(go.Bar(
                    x=followers["influence_score"],
                    y=followers["symbol"],
                    orientation="h",
                    name="Follows TSLA",
                    marker_color="#FF6B35",
                    hovertemplate="%{y}: score=%{x:.1f}, lag=%{customdata}bars<extra></extra>",
                    customdata=followers["best_lag_bars"].abs(),
                ))

            fig_ll.update_layout(
                title=dict(
                    text="<b>領先滯後影響力評分</b>",
                    font=dict(family="IBM Plex Mono", size=13, color="#E8E0D0"),
                    x=0.5,
                ),
                paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A",
                xaxis=dict(title="Influence Score", titlefont=dict(color="#556677"),
                           tickfont=dict(color="#E8E0D0", size=9), gridcolor="#1A2A3A"),
                yaxis=dict(tickfont=dict(color="#E8E0D0", size=9)),
                legend=dict(font=dict(color="#E8E0D0"), bgcolor="rgba(0,0,0,0.5)"),
                height=400, margin=dict(l=60, r=20, t=50, b=40),
                barmode="group",
            )
            st.plotly_chart(fig_ll, use_container_width=True, key="pc_ll")

            # Table
            st.markdown(
                '<div style="font-family:\'IBM Plex Mono\';font-size:0.68rem;color:#556677;'
                'margin-bottom:8px">FULL LEAD-LAG TABLE</div>',
                unsafe_allow_html=True,
            )
            display_ll = lead_lag[["symbol", "best_lag_bars", "correlation",
                                    "direction", "influence_score"]].copy()
            display_ll.columns = ["Symbol", "Lag (bars)", "Corr", "Direction", "Score"]
            st.dataframe(display_ll, use_container_width=True, height=300)

    with col_gr:
        panel_header("Granger 因果（誰引發 TSLA 波動？）")
        granger = compute_granger_scores(returns, target="TSLA")
        if not granger.empty:
            top_granger = granger.head(10)
            fig_gr = go.Figure()

            fig_gr.add_trace(go.Bar(
                x=top_granger["causes_tsla_f"],
                y=top_granger["symbol"],
                orientation="h",
                name="Causes TSLA (F-stat)",
                marker_color="#00FF88",
            ))
            fig_gr.add_trace(go.Bar(
                x=top_granger["tsla_causes_f"],
                y=top_granger["symbol"],
                orientation="h",
                name="TSLA Causes (F-stat)",
                marker_color="#E31937",
            ))

            fig_gr.update_layout(
                title=dict(
                    text="<b>Granger 因果 F 統計量</b>",
                    font=dict(family="IBM Plex Mono", size=13, color="#E8E0D0"),
                    x=0.5,
                ),
                paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A",
                xaxis=dict(title="F-Statistic", titlefont=dict(color="#556677"),
                           tickfont=dict(color="#E8E0D0", size=9), gridcolor="#1A2A3A"),
                yaxis=dict(tickfont=dict(color="#E8E0D0", size=9)),
                legend=dict(font=dict(color="#E8E0D0"), bgcolor="rgba(0,0,0,0.5)"),
                height=380, margin=dict(l=60, r=20, t=50, b=40),
                barmode="group",
            )
            st.plotly_chart(fig_gr, use_container_width=True, key="pc_gr")

        # Regime History
        panel_header("市場狀態歷史")
        if not regime["history"].empty and "regime_name" in regime["history"].columns:
            hist = regime["history"].tail(60).copy()
            hist["regime_id"] = hist["regime_id"].astype(float)

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=hist.index,
                y=hist["regime_id"],
                mode="lines",
                line=dict(color="#E31937", width=2),
                fill="tozeroy",
                fillcolor="rgba(231,25,55,0.1)",
                hovertext=hist["regime_name"],
                hoverinfo="text+x",
            ))

            fig_reg.update_layout(
                title=dict(
                    text="<b>市場狀態時間軸</b>",
                    font=dict(family="IBM Plex Mono", size=13, color="#E8E0D0"),
                    x=0.5,
                ),
                paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A",
                yaxis=dict(
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=["風險偏好", "中性", "風險規避", "恐慌", "AI 狂熱"],
                    tickfont=dict(color="#E8E0D0", size=8),
                    gridcolor="#1A2A3A",
                ),
                xaxis=dict(tickfont=dict(color="#E8E0D0", size=9), gridcolor="#1A2A3A"),
                height=220, margin=dict(l=80, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_reg, use_container_width=True, key="pc_reg")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONTAGION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        panel_header(f"風險傳染：TSLA {shock_size}% 衝擊模擬")
        fig_conta = render_contagion_chart(contagion)
        st.plotly_chart(fig_conta, use_container_width=True, key="pc_conta")

    with col_c2:
        panel_header("波動率聚類（年化）")
        if not vol_clusters.empty:
            plot_vols = [c for c in ["TSLA", "NVDA", "QQQ", "VIX", "BTC"] if c in vol_clusters.columns]
            fig_vol = go.Figure()
            vol_colors = ["#E31937", "#00D4FF", "#00FF88", "#FFB800", "#FF6B35"]
            for i, col_name in enumerate(plot_vols):
                fig_vol.add_trace(go.Scatter(
                    x=vol_clusters.index,
                    y=vol_clusters[col_name] * 100,
                    name=col_name,
                    line=dict(color=vol_colors[i % len(vol_colors)], width=1.5),
                    mode="lines",
                ))

            fig_vol.update_layout(
                title=dict(
                    text="<b>滾動波動率 (%)</b>",
                    font=dict(family="IBM Plex Mono", size=13, color="#E8E0D0"),
                    x=0.5,
                ),
                paper_bgcolor="#0A0E1A", plot_bgcolor="#0A0E1A",
                xaxis=dict(gridcolor="#1A2A3A", tickfont=dict(color="#E8E0D0", size=9)),
                yaxis=dict(gridcolor="#1A2A3A", tickfont=dict(color="#E8E0D0", size=9),
                           title="年化波動率 %", titlefont=dict(color="#556677")),
                legend=dict(font=dict(color="#E8E0D0"), bgcolor="rgba(0,0,0,0.5)"),
                height=380, margin=dict(l=50, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_vol, use_container_width=True, key="pc_vol")

    # Contagion table
    panel_header("完整風險傳染衝擊表")
    if not contagion.empty:
        conta_disp = contagion.copy()
        for col, dec in [("beta_to_tsla", 4), ("contagion_score", 4)]:
            if col in conta_disp.columns:
                conta_disp[col] = pd.to_numeric(conta_disp[col], errors="coerce").round(dec).astype(str).replace("nan", "—")
        if "expected_impact_pct" in conta_disp.columns:
            conta_disp["expected_impact_pct"] = pd.to_numeric(conta_disp["expected_impact_pct"], errors="coerce").round(2).astype(str).replace("nan", "—")
        st.dataframe(conta_disp, use_container_width=True, height=350)

    # Correlation breakdown alerts
    panel_header("⚠️ 相關性崩潰警報")
    if not breakdown.empty:
        alerts = breakdown[~breakdown["signal"].str.contains("Stable")]
        if not alerts.empty:
            for _, row in alerts.iterrows():
                color = "#FF4444" if "Breaking" in row["signal"] else "#00FF88"
                st.markdown(
                    f'<div class="alert-box" style="border-color:{color}40;color:{color}">'
                    f'{row["signal"]} — <b>{row["symbol"]}</b> · '
                    f'Short: {row["short_corr"]:.3f} vs Long: {row["long_corr"]:.3f} (Δ{row["delta"]:+.3f})'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("未偵測到相關性異常。")


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI BRAIN (Groq 免費 + Prompt Generator)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:

    # Pre-build market context once
    market_ctx = build_market_context(
        regime, corr_matrix, lead_lag, contagion, breakdown, returns
    )

    # Three sub-tabs
    sub_ai, sub_prompt, sub_tweet = st.tabs([
        "🤖 AI 分析 (Groq)",
        "📋 Prompt 生成器",
        "⚡ 推文分析器",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # SUB A — GROQ AI ANALYSIS
    # ════════════════════════════════════════════════════════════════════════════
    with sub_ai:
        col_b1, col_b2 = st.columns([3, 2])

        with col_b1:
            panel_header("🧠 Tesla Market Brain — Groq AI (免費)")

            # ── API Status Banner ─────────────────────────────────────────────
            import os as _os
            _has_groq = bool(
                (st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else "") or
                _os.environ.get("GROQ_API_KEY", "")
            )
            _has_claude = bool(
                (st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else "") or
                _os.environ.get("ANTHROPIC_API_KEY", "")
            )

            if _has_groq:
                st.markdown(
                    '<div style="font-family:\'IBM Plex Mono\';font-size:9px;'
                    'color:#00FF88;margin-bottom:8px">'
                    '<span class="status-dot"></span>🟢 Groq API 已連接（免費）</div>',
                    unsafe_allow_html=True,
                )
            elif _has_claude:
                st.markdown(
                    '<div style="font-family:\'IBM Plex Mono\';font-size:9px;'
                    'color:#4488FF;margin-bottom:8px">'
                    '🔵 Claude API 已連接（備用）</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alert-box">'
                    '⚠️ 未設置 API Key。<br>'
                    '請在 Streamlit Cloud → Settings → Secrets 設置：<br>'
                    '<code>GROQ_API_KEY = "gsk_..."</code>（免費，至 console.groq.com 申請）<br>'
                    '或 <code>ANTHROPIC_API_KEY = "sk-ant-..."</code><br><br>'
                    '💡 即使沒有 API Key，也可以使用「📋 Prompt 生成器」標籤，'
                    '把 Prompt 複製到任何 AI！'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # Session state
            if "brain_output" not in st.session_state:
                st.session_state.brain_output = None
            if "brain_model" not in st.session_state:
                st.session_state.brain_model = ""

            if run_ai:
                with st.spinner("🧠 Groq 正在生成市場分析..."):
                    output, model_used = generate_market_brain(
                        regime=regime,
                        corr_matrix=corr_matrix,
                        lead_lag=lead_lag,
                        contagion=contagion,
                        breakdown=breakdown,
                        returns=returns,
                    )
                    st.session_state.brain_output = output
                    st.session_state.brain_model  = model_used

            if st.session_state.brain_output:
                if st.session_state.brain_model:
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\';font-size:9px;'
                        f'color:#445566;margin-bottom:6px;text-align:right">'
                        f'生成模型: {st.session_state.brain_model}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="brain-output chinese">{st.session_state.brain_output}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("📋 複製原始文本"):
                    st.text_area("", st.session_state.brain_output,
                                 height=200, key="brain_copy")
            else:
                st.markdown(
                    '<div class="brain-output" style="color:#334455">'
                    '點擊左側「🧠 Generate Market Analysis」啟動分析...\n\n'
                    '如無 API Key → 使用「📋 Prompt 生成器」標籤，'
                    '複製 Prompt 到任何 AI 使用！'
                    '</div>',
                    unsafe_allow_html=True,
                )

        with col_b2:
            panel_header("市場摘要儀表板")
            st.markdown("**Current Regime:**")
            regime_badge(regime.get("current_name", "UNKNOWN"), regime.get("current_color", "#888"))
            st.markdown(f"*Confidence: {regime.get('confidence', 0):.0%}*")
            st.markdown("<br>", unsafe_allow_html=True)

            panel_header("TSLA 頂部相關資產")
            if not corr_matrix.empty and "TSLA" in corr_matrix.columns:
                tsla_corr = corr_matrix["TSLA"].drop("TSLA").sort_values(ascending=False)
                for sym, val in tsla_corr.head(8).items():
                    bar_color = "#00D4AA" if val > 0 else "#FF4455"
                    bar_w = int(abs(val) * 100)
                    st.markdown(
                        f"""<div style="margin-bottom:4px;font-family:'IBM Plex Mono';font-size:9px">
                            <span style="color:#E8E0D0;display:inline-block;width:50px">{sym}</span>
                            <span style="color:{bar_color}">{val:+.3f}</span>
                            <div style="background:#0D1626;border-radius:2px;height:3px;margin-top:2px">
                                <div style="background:{bar_color};width:{bar_w}%;height:3px;border-radius:2px"></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            panel_header("TSLA 領先指標排名")
            if not lead_lag.empty:
                leaders_ai = lead_lag[lead_lag["direction"] == "leads TSLA"].head(5)
                for _, row in leaders_ai.iterrows():
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\';font-size:9px;'
                        f'color:#00D4FF;margin-bottom:3px">'
                        f'◀ {row["symbol"]} → TSLA · {row["best_lag_bars"]}bars'
                        f' · ρ={row["correlation"]:.3f}</div>',
                        unsafe_allow_html=True,
                    )

    # ════════════════════════════════════════════════════════════════════════════
    # SUB B — PROMPT GENERATOR
    # ════════════════════════════════════════════════════════════════════════════
    with sub_prompt:
        panel_header("📋 AI Prompt 生成器 — 複製到任何 AI 使用")

        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\';font-size:10px;'
            'color:#667788;margin-bottom:16px">'
            '系統自動從實時市場數據生成結構化 Prompt。'
            '複製後可貼到 ChatGPT / Claude / Gemini / Grok / Perplexity 等任何 AI 直接使用。'
            '</div>',
            unsafe_allow_html=True,
        )

        pg_col1, pg_col2, pg_col3 = st.columns([2, 1, 1])
        with pg_col1:
            prompt_type_label = st.selectbox(
                "分析類型",
                list(PROMPT_TYPES.values()),
                key="pg_type",
            )
            prompt_type_key = {v: k for k, v in PROMPT_TYPES.items()}.get(
                prompt_type_label, "market_brain"
            )
        with pg_col2:
            pg_target_ai = st.selectbox("目標 AI 平台", TARGET_AIS, key="pg_ai")
        with pg_col3:
            pg_language = st.selectbox("語言", ["繁體中文", "简体中文", "English"], key="pg_lang")

        pg_extra = st.text_input(
            "補充信息（選填）",
            placeholder="例如：今日 FOMC 會議，市場擔憂加息... 或 Elon 剛發推...",
            key="pg_extra",
        )

        if st.button("⚡ 生成 Prompt", use_container_width=True, key="gen_prompt_btn"):
            st.session_state.generated_prompt = get_prompt_for_ai(
                prompt_type_key, market_ctx, pg_target_ai, pg_language, pg_extra
            )

        if "generated_prompt" not in st.session_state:
            st.session_state.generated_prompt = get_prompt_for_ai(
                "market_brain", market_ctx, "通用", "繁體中文"
            )

        panel_header("生成的 Prompt（全選複製 → 貼到任何 AI）")
        st.text_area(
            "",
            st.session_state.generated_prompt,
            height=420,
            key="prompt_display",
            help="Ctrl+A 全選 → Ctrl+C 複製 → 貼到 ChatGPT / Claude / Gemini / Grok",
        )

        prompt_len = len(st.session_state.generated_prompt)
        token_est  = prompt_len // 4
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\';font-size:8px;color:#445566;text-align:right">'
            f'字符數: {prompt_len:,} · 估計 Tokens: ~{token_est:,} · 目標平台: {pg_target_ai}</div>',
            unsafe_allow_html=True,
        )

        # Quick preset buttons
        st.markdown("<br>", unsafe_allow_html=True)
        panel_header("⚡ 快速生成常用 Prompt")
        presets = [
            ("🧠 今日市場診斷",   "market_brain"),
            ("🎯 TSLA 交易策略",  "tsla_trade"),
            ("⚠️ 風險評估報告",   "risk_report"),
            ("🔄 板塊輪動分析",   "sector_rotation"),
            ("🏛️ 市場狀態分析",  "regime_analysis"),
            ("🌊 Elon 敘事傳播鏈", "narrative_chain"),
        ]
        preset_cols = st.columns(3)
        for i, (label, ptype) in enumerate(presets):
            with preset_cols[i % 3]:
                if st.button(label, use_container_width=True, key="preset_" + ptype):
                    st.session_state.generated_prompt = get_prompt_for_ai(
                        ptype, market_ctx, pg_target_ai, pg_language
                    )
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 查看原始市場數據 Context"):
            st.code(market_ctx, language="text")

    # ════════════════════════════════════════════════════════════════════════════
    # SUB C — TWEET ANALYZER
    # ════════════════════════════════════════════════════════════════════════════
    with sub_tweet:
        panel_header("⚡ Elon 推文 / 新聞 市場衝擊分析器")

        tw_col1, tw_col2 = st.columns([3, 2])

        with tw_col1:
            tweet_input_main = st.text_area(
                "輸入推文或新聞標題",
                height=100,
                placeholder="例如：FSD 13 is shipping this quarter. Game over for legacy auto.",
                key="tweet_main",
            )

            if tweet_input_main:
                quick_tw = score_sentiment_simple(tweet_input_main)
                sent_color = "#00FF88" if quick_tw["score"] > 0.1 else (
                    "#FF4444" if quick_tw["score"] < -0.1 else "#888"
                )
                tq1, tq2, tq3, tq4 = st.columns(4)
                with tq1:
                    metric_card("情緒", quick_tw["sentiment_label"],
                                f"Score: {quick_tw['score']:+.2f}", sent_color)
                with tq2:
                    metric_card("緊急程度", quick_tw["urgency"])
                with tq3:
                    metric_card("類別", quick_tw["category"])
                with tq4:
                    metric_card("多空信號",
                                f"🟢{quick_tw['bull_signals']} / 🔴{quick_tw['bear_signals']}")

                st.markdown("<br>", unsafe_allow_html=True)
                tw_btn1, tw_btn2 = st.columns(2)
                with tw_btn1:
                    run_tweet_ai = st.button(
                        "🤖 AI深度分析 (Groq)", use_container_width=True, key="tw_ai_btn"
                    )
                with tw_btn2:
                    gen_tweet_prompt = st.button(
                        "📋 生成分析 Prompt", use_container_width=True, key="tw_prompt_btn"
                    )

                if run_tweet_ai:
                    with st.spinner("⚡ Groq 分析推文衝擊中..."):
                        tw_text, tw_model = analyze_narrative_impact(
                            tweet_input_main, market_ctx
                        )
                        st.session_state.tweet_analysis = tw_text
                        st.session_state.tweet_model    = tw_model

                if gen_tweet_prompt:
                    tw_p = get_prompt_for_ai(
                        "tweet_impact", market_ctx, "通用", "繁體中文", tweet_input_main
                    )
                    st.session_state.tweet_prompt_text = tw_p

                if st.session_state.get("tweet_analysis"):
                    tw_m = st.session_state.get("tweet_model", "")
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\';font-size:9px;'
                        f'color:#445566;margin-bottom:4px;text-align:right">'
                        f'模型: {tw_m}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="brain-output chinese">'
                        f'{st.session_state.tweet_analysis}</div>',
                        unsafe_allow_html=True,
                    )

                if st.session_state.get("tweet_prompt_text"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    panel_header("📋 可複製到任何 AI 的分析 Prompt")
                    st.text_area(
                        "", st.session_state.tweet_prompt_text,
                        height=280, key="tweet_prompt_display"
                    )

        with tw_col2:
            panel_header("快速分析樣本推文")
            sample_tweets = [
                ("FSD 13 shipping this quarter. Game over for legacy auto.", "🚗 FSD"),
                ("xAI — Grok 4 changes everything we know about intelligence.", "🤖 AI"),
                ("Bitcoin is engineering genius. Don't fight math.", "₿ Crypto"),
                ("Interest rates need to come down. The data is clear.", "📊 Macro"),
                ("Optimus: 1 million robots by 2027 is conservative.", "🦾 Optimus"),
                ("Tesla Robotaxi launches in Austin next month.", "🚗 Robotaxi"),
            ]
            for tw_sample, tw_cat in sample_tweets:
                label_short = tw_cat + " · " + tw_sample[:35] + "..."
                if st.button(label_short, use_container_width=True,
                             key="sample_tw_" + tw_cat):
                    # Use session state to pre-fill
                    st.session_state["_sample_tweet"] = tw_sample
                    st.info("請將以下推文貼入上方輸入框：\n\n" + tw_sample)

            st.markdown("<br>", unsafe_allow_html=True)
            panel_header("Elon 敘事市場傳播鏈")
            st.markdown(
                """<div style="font-family:'IBM Plex Mono';font-size:9px;line-height:2.2;color:#C8C0B0">
<span style="color:#E31937;font-weight:bold">📡 Elon Tweet / News</span><br>
<span style="color:#445566">    ↓ 即時 (&lt;5min)</span><br>
<span style="color:#E31937">⭐ TSLA 成交量激增</span><br>
<span style="color:#445566">    ↓ 5-30min</span><br>
<span style="color:#00D4FF">ARKK · NVDA · AMD</span><br>
<span style="color:#445566">    ↓ 30-120min</span><br>
<span style="color:#00FF88">QQQ · SPY Risk-On</span><br>
<span style="color:#445566">    ↓ 數小時</span><br>
<span style="color:#FFB800">BTC · ETH · COIN</span><br>
<span style="color:#445566">    ↓ 次日</span><br>
<span style="color:#888888">Regime 重新評估</span>
</div>""",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — NARRATIVE FEED
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    col_n1, col_n2 = st.columns([1, 1])

    with col_n1:
        panel_header("📡 實時新聞")
        news_df = fetch_news_feed(max_items=25)
        if not news_df.empty:
            # Category filter
            categories = ["All"] + news_df["narrative"].unique().tolist()
            selected_cat = st.selectbox("Filter by Narrative", categories, key="news_cat")
            filtered = news_df if selected_cat == "All" else news_df[news_df["narrative"] == selected_cat]
            render_news_feed(filtered, max_show=12)
        else:
            st.info("📡 Loading news feed... (requires internet)")

    with col_n2:
        panel_header("𝕏 Elon 敘事推文")
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\';font-size:8px;color:#334455;'
            'margin-bottom:10px">Simulated feed (X API requires premium subscription)</div>',
            unsafe_allow_html=True,
        )

        tweets = get_simulated_tweets()
        for t in tweets:
            tweet_card(t["text"], t["time"], t["likes"], t["category"])

        st.markdown("<br>", unsafe_allow_html=True)
        panel_header("Elon 敘事分類")

        # Pie chart of tweet categories
        cat_counts = {}
        for t in tweets:
            cat_counts[t["category"]] = cat_counts.get(t["category"], 0) + 1

        fig_pie = go.Figure(go.Pie(
            labels=list(cat_counts.keys()),
            values=list(cat_counts.values()),
            hole=0.6,
            marker_colors=["#00D4FF", "#00FF88", "#FF6B35", "#FFB800", "#888", "#FF4444", "#E31937"],
            textfont=dict(family="IBM Plex Mono", size=9, color="#E8E0D0"),
        ))
        fig_pie.update_layout(
            title=dict(
                text="<b>Narrative Distribution</b>",
                font=dict(family="IBM Plex Mono", size=12, color="#E8E0D0"),
                x=0.5,
            ),
            paper_bgcolor="#0A0E1A",
            legend=dict(font=dict(color="#E8E0D0", size=9)),
            height=280,
            margin=dict(l=10, r=10, t=50, b=10),
            annotations=[dict(
                text="Narratives",
                font=dict(family="IBM Plex Mono", size=10, color="#E8E0D0"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pc_pie")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — 機構持倉追蹤 (Institutional Holdings)
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:

    panel_header("🏦 TSLA 十大機構持倉 — 季度追蹤")

    # ── 數據載入 ──────────────────────────────────────────────────────────────
    with st.spinner("📡 載入機構持倉數據..."):
        inst_df   = fetch_tsla_institutional_holders()
        top10_df  = compute_qoq_changes(inst_df)
        ownership = get_ownership_summary()
        ark_data  = fetch_ark_tsla_position()

    # ── 頂部概覽指標 ──────────────────────────────────────────────────────────
    ow_c1, ow_c2, ow_c3, ow_c4, ow_c5 = st.columns(5)

    with ow_c1:
        metric_card(
            "機構持股比例",
            f"{ownership['institutional_pct']:.1f}%",
            "Institutional",
            "#00D4FF",
        )
    with ow_c2:
        metric_card(
            "內部人持股",
            f"{ownership['insider_pct']:.1f}%",
            "Insiders",
            "#FFB800",
        )
    with ow_c3:
        metric_card(
            "流通股",
            f"{ownership['float_shares']:.2f}B",
            "Float Shares",
            "#00FF88",
        )
    with ow_c4:
        metric_card(
            "空頭比例",
            f"{ownership['short_pct_float']:.1f}%",
            f"Short Ratio: {ownership['short_ratio']:.1f}",
            "#FF4444",
        )
    with ow_c5:
        metric_card(
            "總發行股數",
            f"{ownership['shares_outstanding']:.2f}B",
            "Outstanding",
            "#888888",
        )

    st.markdown("---")

    # ── 兩欄主體 ──────────────────────────────────────────────────────────────
    h_col1, h_col2 = st.columns([3, 2])

    with h_col1:
        panel_header("十大基金持倉詳情 + 季度變化")

        if not top10_df.empty:
            # ── 持倉條形圖 ────────────────────────────────────────────────────
            fig_hold = go.Figure()

            # Current holdings bar
            fig_hold.add_trace(go.Bar(
                x=top10_df["shares_m"].fillna(0),
                y=top10_df["short_name"].fillna(top10_df["fund_name"]),
                orientation="h",
                name="當前持股 (M股)",
                marker=dict(
                    color=[FUND_COLORS.get(n, "#888") for n in top10_df["fund_name"]],
                    opacity=0.85,
                    line=dict(color="#0A1020", width=1),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "持股: %{x:.1f}M 股<br>"
                    "<extra></extra>"
                ),
                text=[f"{v:.1f}M" if v > 0 else "N/A"
                      for v in top10_df["shares_m"].fillna(0)],
                textposition="outside",
                textfont=dict(color="#E8E0D0", size=9, family="IBM Plex Mono"),
            ))

            # QoQ change markers
            if "qoq_shares_m" in top10_df.columns:
                fig_hold.add_trace(go.Scatter(
                    x=top10_df["qoq_shares_m"].fillna(0),
                    y=top10_df["short_name"].fillna(top10_df["fund_name"]),
                    mode="markers",
                    name="季度變化 (M股)",
                    marker=dict(
                        symbol="diamond",
                        size=10,
                        color=["#00FF88" if v >= 0 else "#FF4444"
                               for v in top10_df["qoq_shares_m"].fillna(0)],
                        line=dict(color="#0A1020", width=1),
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "QoQ 變化: %{x:+.1f}M 股<br>"
                        "<extra></extra>"
                    ),
                ))

            fig_hold.update_layout(
                title=dict(
                    text="<b>十大機構持股量 + 季度增減</b>",
                    font=dict(family="IBM Plex Mono", size=13, color="#E8E0D0"),
                    x=0.5,
                ),
                paper_bgcolor="#0A0E1A",
                plot_bgcolor="#0A0E1A",
                xaxis=dict(
                    title="持股數量（百萬股）",
                    titlefont=dict(color="#556677"),
                    tickfont=dict(color="#E8E0D0", size=9),
                    gridcolor="#1A2A3A",
                    zeroline=True,
                    zerolinecolor="#334455",
                ),
                yaxis=dict(
                    tickfont=dict(color="#E8E0D0", size=10),
                    autorange="reversed",
                ),
                legend=dict(
                    font=dict(color="#E8E0D0", size=9),
                    bgcolor="rgba(0,0,0,0.5)",
                    x=0.6, y=0.05,
                ),
                height=420,
                margin=dict(l=80, r=100, t=60, b=40),
                barmode="overlay",
            )
            st.plotly_chart(fig_hold, use_container_width=True, key="pc_hold")

            # ── 詳細數據表格 ──────────────────────────────────────────────────
            panel_header("持倉數據明細")

            for _, row in top10_df.iterrows():
                fname   = str(row.get("fund_name", ""))
                sname   = str(row.get("short_name", fname[:12]))
                ftype   = str(row.get("fund_type", ""))
                color   = str(row.get("color", "#888"))
                shares  = float(row.get("shares_m", 0) or 0)
                val_b   = float(row.get("value_b", 0) or 0)
                pct     = float(row.get("pct_held", 0) or 0)
                qoq     = float(row.get("qoq_shares_m", 0) or 0)
                qoq_pct = float(row.get("qoq_pct", 0) or 0)
                action  = str(row.get("action", "⚪ 持平"))
                date_r  = str(row.get("date_reported", "N/A"))

                qoq_color = "#00FF88" if qoq >= 0 else "#FF4444"
                bar_w = min(int(shares / 3), 100)  # scale to max ~300M

                st.markdown(
                    f"""<div style="
                        background:#0C1520;
                        border:1px solid #1A2A3A;
                        border-left:3px solid {color};
                        border-radius:0 6px 6px 0;
                        padding:10px 14px;
                        margin-bottom:6px;
                        font-family:'IBM Plex Mono';
                    ">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                            <div>
                                <span style="color:{color};font-size:11px;font-weight:600">{sname}</span>
                                <span style="color:#445566;font-size:9px;margin-left:8px">{ftype}</span>
                            </div>
                            <span style="color:{qoq_color};font-size:11px">{action} {qoq_pct:+.1f}%</span>
                        </div>
                        <div style="display:flex;gap:24px;font-size:9px;color:#C8C0B0;margin-bottom:6px">
                            <span>持股: <b style="color:#E8E0D0">{shares:.1f}M 股</b></span>
                            <span>市值: <b style="color:#E8E0D0">${val_b:.1f}B</b></span>
                            <span>佔比: <b style="color:#E8E0D0">{pct:.2f}%</b></span>
                            <span style="color:#445566">截至: {date_r}</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:8px">
                            <div style="flex:1;background:#0D1626;border-radius:2px;height:4px">
                                <div style="background:{color};width:{bar_w}%;height:4px;border-radius:2px;opacity:0.7"></div>
                            </div>
                            <span style="color:{qoq_color};font-size:9px;width:60px;text-align:right">
                                QoQ: {qoq:+.1f}M
                            </span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("⏳ 持倉數據載入中...")

    with h_col2:

        # ── 持股比例 Donut Chart ───────────────────────────────────────────────
        panel_header("持股結構概覽")

        if not top10_df.empty:
            labels_pie = list(top10_df["short_name"].fillna(top10_df["fund_name"]))
            values_pie = list(top10_df["shares_m"].fillna(0))
            colors_pie = [FUND_COLORS.get(n, "#888") for n in top10_df["fund_name"]]

            # Add "其他機構" slice
            total_top10 = sum(values_pie)
            other_inst  = max(0, ownership["institutional_pct"] / 100 *
                              ownership["shares_outstanding"] * 1000 - total_top10)
            if other_inst > 0:
                labels_pie.append("其他機構")
                values_pie.append(other_inst)
                colors_pie.append("#334455")

            fig_donut = go.Figure(go.Pie(
                labels=labels_pie,
                values=values_pie,
                hole=0.55,
                marker=dict(colors=colors_pie, line=dict(color="#060B18", width=2)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value:.1f}M 股<br>%{percent}<extra></extra>",
            ))
            fig_donut.update_layout(
                title=dict(
                    text="<b>機構持股分佈</b>",
                    font=dict(family="IBM Plex Mono", size=12, color="#E8E0D0"),
                    x=0.5,
                ),
                paper_bgcolor="#0A0E1A",
                legend=dict(
                    font=dict(color="#E8E0D0", size=8),
                    bgcolor="rgba(0,0,0,0)",
                ),
                annotations=[dict(
                    text=f"{ownership['institutional_pct']:.0f}%<br><span style='font-size:8px'>機構</span>",
                    font=dict(family="IBM Plex Mono", size=14, color="#E8E0D0"),
                    showarrow=False,
                )],
                height=300,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True, key="pc_donut")

        # ── QoQ 增減排名 ─────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        panel_header("季度增減排名 (QoQ)")

        if not top10_df.empty and "qoq_shares_m" in top10_df.columns:
            sorted_qoq = top10_df.sort_values("qoq_shares_m", ascending=False)

            fig_qoq = go.Figure(go.Bar(
                x=sorted_qoq["qoq_shares_m"].fillna(0),
                y=sorted_qoq["short_name"].fillna(sorted_qoq["fund_name"]),
                orientation="h",
                marker=dict(
                    color=["#00FF88" if v >= 0 else "#FF4444"
                           for v in sorted_qoq["qoq_shares_m"].fillna(0)],
                    opacity=0.85,
                ),
                hovertemplate="<b>%{y}</b><br>QoQ: %{x:+.1f}M 股<extra></extra>",
                text=[f"{v:+.1f}M" for v in sorted_qoq["qoq_shares_m"].fillna(0)],
                textposition="outside",
                textfont=dict(color="#E8E0D0", size=9, family="IBM Plex Mono"),
            ))
            fig_qoq.add_vline(x=0, line_color="#334455", line_width=1)
            fig_qoq.update_layout(
                title=dict(
                    text="<b>季度增減（估算）</b>",
                    font=dict(family="IBM Plex Mono", size=11, color="#E8E0D0"),
                    x=0.5,
                ),
                paper_bgcolor="#0A0E1A",
                plot_bgcolor="#0A0E1A",
                xaxis=dict(
                    tickfont=dict(color="#E8E0D0", size=8),
                    gridcolor="#1A2A3A",
                    zeroline=False,
                ),
                yaxis=dict(
                    tickfont=dict(color="#E8E0D0", size=9),
                    autorange="reversed",
                ),
                height=320,
                margin=dict(l=70, r=60, t=40, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_qoq, use_container_width=True, key="pc_qoq")

        # ── ARK 持倉（每日更新）──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        panel_header("ARK Invest 每日持倉")

        if ark_data:
            for fund_name, data in ark_data.items():
                weight = data.get("weight", 0)
                shares = data.get("shares", 0) / 1e6
                mktval = data.get("market_value", 0) / 1e6
                date_s = data.get("date", "N/A")
                st.markdown(
                    f"""<div style="
                        background:#0C1520;border:1px solid #1A2A3A;
                        border-left:3px solid #FF6B35;border-radius:0 6px 6px 0;
                        padding:8px 12px;margin-bottom:6px;font-family:'IBM Plex Mono'
                    ">
                        <span style="color:#FF6B35;font-size:10px;font-weight:600">{fund_name}</span>
                        <span style="color:#445566;font-size:8px;margin-left:6px">{date_s}</span><br>
                        <span style="font-size:9px;color:#C8C0B0">
                            持股: <b style="color:#E8E0D0">{shares:.2f}M</b> ·
                            市值: <b style="color:#E8E0D0">${mktval:.1f}M</b> ·
                            權重: <b style="color:#FFB800">{weight:.1f}%</b>
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            # ARK website blocked — show estimated data
            st.markdown(
                """<div style="
                    background:#0C1520;border:1px solid #1A2A3A;
                    border-left:3px solid #FF6B35;border-radius:0 6px 6px 0;
                    padding:8px 12px;margin-bottom:6px;font-family:'IBM Plex Mono'
                ">
                    <span style="color:#FF6B35;font-size:10px;font-weight:600">ARKK</span>
                    <span style="color:#445566;font-size:8px;margin-left:6px">估算數據</span><br>
                    <span style="font-size:9px;color:#C8C0B0">
                        持股: <b style="color:#E8E0D0">~52M</b> ·
                        權重: <b style="color:#FFB800">~6-8%</b>
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-family:\'IBM Plex Mono\';font-size:8px;color:#334455">'
                '💡 ARK 每日更新：ark-funds.com/funds/arkk</div>',
                unsafe_allow_html=True,
            )

    # ── 底部：全機構持倉排行 ──────────────────────────────────────────────────
    st.markdown("---")
    panel_header("全機構持倉排行（yfinance 實時數據）")

    h_sub1, h_sub2 = st.columns(2)

    with h_sub1:
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\';font-size:9px;'
            'color:#445566;margin-bottom:8px">機構投資者（Institutional Holders）</div>',
            unsafe_allow_html=True,
        )
        if not inst_df.empty:
            inst_rows = []
            for _, r in inst_df.iterrows():
                def _f(key, dec):
                    try: return round(float(r[key]), dec) if key in r.index and r[key]==r[key] else "—"
                    except: return "—"
                inst_rows.append({"持倉機構": str(r.get("holder","")),"持股(M)": _f("shares_m",1),"市值($B)": _f("value_b",2),"佔比(%)": _f("pct_held",2),"申報日期": str(r.get("date_reported","N/A"))})
            st.dataframe(pd.DataFrame(inst_rows), use_container_width=True, height=300)
        else:
            st.info("數據載入中...")

    with h_sub2:
        mf_df = fetch_tsla_mutualfund_holders()
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\';font-size:9px;'
            'color:#445566;margin-bottom:8px">共同基金（Mutual Fund Holders）</div>',
            unsafe_allow_html=True,
        )
        if not mf_df.empty:
            mf_rows = []
            for _, r in mf_df.iterrows():
                def _f2(key, dec):
                    try: return round(float(r[key]), dec) if key in r.index and r[key]==r[key] else "—"
                    except: return "—"
                mf_rows.append({"基金名稱": str(r.get("holder","")),"持股(M)": _f2("shares_m",1),"市值($B)": _f2("value_b",2),"佔比(%)": _f2("pct_held",2),"申報日期": str(r.get("date_reported","N/A"))})
            st.dataframe(pd.DataFrame(mf_rows), use_container_width=True, height=300)
        else:
            st.info("共同基金數據載入中...")

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\';font-size:8px;color:#223344;'
        'text-align:center;padding:8px;margin-top:8px">'
        '⚠️ 機構持倉數據來源：Yahoo Finance (yfinance) / SEC 13F 申報 · '
        '季度延遲約 45 天 · QoQ 變化為估算值 · 僅供研究參考</div>',
        unsafe_allow_html=True,
    )



# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """<div style="font-family:'IBM Plex Mono';font-size:0.6rem;color:#223344;text-align:center;padding:8px">
    ⚠️ TESLA MARKET BRAIN · For research purposes only · Not financial advice ·
    數據：Yahoo Finance · AI：Groq llama-3.3 · 架構：Streamlit Cloud
    </div>""",
    unsafe_allow_html=True,
)
