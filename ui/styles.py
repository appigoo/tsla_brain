"""
styles.py — Global CSS and UI component helpers
"""

DARK_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

/* ── Base ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    background-color: #060B18 !important;
    color: #E8E0D0 !important;
}

.stApp {
    background: linear-gradient(160deg, #060B18 0%, #0A1020 50%, #08141E 100%) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #080D1A !important;
    border-right: 1px solid #1A2A3A !important;
}

[data-testid="stSidebar"] * {
    color: #C8C0B0 !important;
}

/* ── Main header ──────────────────────────────────────── */
.main-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    background: linear-gradient(90deg, #E31937 0%, #FF6B35 40%, #FFB800 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 0.5rem 0;
    text-transform: uppercase;
}

.sub-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #556677;
    text-align: center;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── Metric Cards ─────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #0D1626 0%, #111E30 100%);
    border: 1px solid #1A2A3A;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}

.metric-card:hover {
    border-color: #2A4A6A;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #E31937, #FF6B35);
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #556677;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #E8E0D0;
}

.metric-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    margin-top: 2px;
}

/* ── Regime Badge ─────────────────────────────────────── */
.regime-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 4px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Panel Section ────────────────────────────────────── */
.panel-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #556677;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid #1A2A3A;
    padding-bottom: 8px;
    margin-bottom: 14px;
}

/* ── AI Brain Output ──────────────────────────────────── */
.brain-output {
    background: #0A1425;
    border: 1px solid #1A2A3A;
    border-left: 3px solid #E31937;
    border-radius: 0 8px 8px 0;
    padding: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #C8C0B0;
    white-space: pre-wrap;
}

/* ── Tweet Card ───────────────────────────────────────── */
.tweet-card {
    background: #0C1520;
    border: 1px solid #1A2A3A;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
}

.tweet-text {
    font-size: 0.8rem;
    color: #D0C8B8;
    line-height: 1.5;
    margin-bottom: 6px;
}

.tweet-meta {
    font-size: 0.68rem;
    color: #445566;
}

/* ── Tabs ─────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    color: #556677 !important;
    text-transform: uppercase !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #E31937 !important;
    border-bottom-color: #E31937 !important;
}

/* ── Streamlit overrides ──────────────────────────────── */
.stSelectbox label, .stSlider label, .stRadio label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #556677 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

div[data-baseweb="select"] {
    background: #0D1626 !important;
}

div[data-baseweb="select"] * {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Spinner ──────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #E31937 !important;
}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060B18; }
::-webkit-scrollbar-thumb { background: #2A3A4A; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #3A5A7A; }

/* ── Status Dot ───────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00FF88;
    animation: pulse 2s infinite;
    margin-right: 6px;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

/* ── Noto Sans TC for Chinese ────────────────────────── */
.chinese { font-family: 'Noto Sans TC', 'IBM Plex Mono', sans-serif !important; }

/* ── Warning/Alert ───────────────────────────────────── */
.alert-box {
    background: rgba(231, 25, 55, 0.08);
    border: 1px solid rgba(231, 25, 55, 0.3);
    border-radius: 6px;
    padding: 10px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #FF8899;
    margin-bottom: 8px;
}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", delta_color: str = "#00FF88"):
    import streamlit as st
    delta_html = f'<div class="metric-delta" style="color:{delta_color}">{delta}</div>' if delta else ""
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


def regime_badge(name: str, color: str):
    import streamlit as st
    text_color = "#000" if color in ("#FFB800", "#00FF88") else "#FFF"
    st.markdown(
        f'<div class="regime-badge" style="background:{color};color:{text_color}">{name}</div>',
        unsafe_allow_html=True,
    )


def panel_header(title: str):
    import streamlit as st
    st.markdown(f'<div class="panel-header">◈ {title}</div>', unsafe_allow_html=True)


def tweet_card(text: str, time_str: str, likes: int, category: str):
    import streamlit as st
    cat_colors = {
        "🤖 AI/xAI": "#00D4FF",
        "🚗 FSD/Robotaxi": "#00FF88",
        "🦾 Optimus": "#FF6B35",
        "₿ Crypto": "#FFB800",
        "📊 Macro": "#888",
        "🗳️ Political": "#FF4444",
        "⚡ Tesla": "#E31937",
    }
    cat_color = cat_colors.get(category, "#556677")
    st.markdown(
        f"""<div class="tweet-card">
            <span style="font-size:9px;color:{cat_color};letter-spacing:0.1em">{category}</span>
            <div class="tweet-text">"{text}"</div>
            <div class="tweet-meta">
                {time_str} · ❤️ {likes:,}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def status_bar(last_update: str):
    import streamlit as st
    st.markdown(
        f"""<div style="font-family:'IBM Plex Mono';font-size:0.65rem;color:#334455;
                        text-align:right;padding:4px 0">
            <span class="status-dot"></span>LIVE · Last updated: {last_update}
        </div>""",
        unsafe_allow_html=True,
    )
