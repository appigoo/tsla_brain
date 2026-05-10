"""
news_engine.py — News + Narrative Feed
Uses free RSS feeds (no API key required) — Streamlit Cloud compatible
"""
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import streamlit as st
import re


RSS_FEEDS = {
    "Tesla": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
    ],
    "Elon": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA,COIN,MSTR&region=US&lang=en-US",
    ],
    "Market": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ,VIX&region=US&lang=en-US",
    ],
}

NARRATIVE_KEYWORDS = {
    "🤖 AI/xAI":     ["ai", "xai", "grok", "artificial intelligence", "machine learning"],
    "🚗 FSD/Robotaxi": ["fsd", "robotaxi", "self-driving", "autopilot", "autonomy"],
    "🦾 Optimus":    ["optimus", "robot", "humanoid"],
    "₿ Crypto":      ["bitcoin", "btc", "crypto", "dogecoin", "doge"],
    "📊 Macro":      ["federal reserve", "fed", "inflation", "tariff", "rate", "gdp"],
    "⚡ Tesla":      ["tesla", "gigafactory", "cybertruck", "model y", "model 3"],
    "🗳️ Political": ["doge", "government", "trump", "congress", "regulation"],
    "💰 Earnings":   ["earnings", "revenue", "profit", "eps", "guidance"],
}


def _clean_html(text: str) -> str:
    """Strip HTML tags."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


@st.cache_data(ttl=600)
def fetch_news_feed(max_items: int = 30) -> pd.DataFrame:
    """Fetch news from RSS feeds."""
    items = []
    for category, urls in RSS_FEEDS.items():
        for url in urls:
            try:
                resp = requests.get(url, timeout=10,
                                    headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.content)
                channel = root.find("channel")
                if channel is None:
                    continue
                for item in channel.findall("item")[:15]:
                    title = _clean_html(item.findtext("title", ""))
                    desc = _clean_html(item.findtext("description", ""))
                    link = item.findtext("link", "")
                    pub = item.findtext("pubDate", "")

                    if not title:
                        continue

                    # Parse date
                    try:
                        dt = datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S")
                    except Exception:
                        dt = datetime.utcnow()

                    # Classify narrative
                    text_lower = (title + " " + desc).lower()
                    narrative = classify_narrative(text_lower)

                    # Sentiment
                    sentiment = quick_sentiment(text_lower)

                    items.append({
                        "title": title,
                        "description": desc[:150] + "..." if len(desc) > 150 else desc,
                        "link": link,
                        "published": dt,
                        "category": category,
                        "narrative": narrative,
                        "sentiment": sentiment,
                        "sentiment_emoji": "🟢" if sentiment > 0.1 else ("🔴" if sentiment < -0.1 else "⚪"),
                    })
            except Exception:
                continue

    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["title"])
    df = df.sort_values("published", ascending=False).head(max_items)
    return df.reset_index(drop=True)


def classify_narrative(text: str) -> str:
    """Classify article into narrative category."""
    for narrative, keywords in NARRATIVE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return narrative
    return "📰 General"


def quick_sentiment(text: str) -> float:
    """Simple lexicon-based sentiment (-1 to +1)."""
    positive = ["surge", "rally", "gain", "profit", "bullish", "record",
                 "beat", "strong", "growth", "breakthrough", "launch", "positive"]
    negative = ["crash", "drop", "loss", "bearish", "miss", "warn",
                 "risk", "concern", "decline", "cut", "fraud", "recall"]

    p = sum(1 for w in positive if w in text)
    n = sum(1 for w in negative if w in text)
    total = p + n
    if total == 0:
        return 0.0
    return round((p - n) / total, 2)


def render_news_feed(df: pd.DataFrame, max_show: int = 10) -> None:
    """Render news feed in Streamlit."""
    if df.empty:
        st.info("📡 Fetching news...")
        return

    for _, row in df.head(max_show).iterrows():
        sentiment_color = "#00FF88" if row["sentiment"] > 0.1 else (
            "#FF4444" if row["sentiment"] < -0.1 else "#888888"
        )
        time_str = row["published"].strftime("%H:%M") if pd.notna(row.get("published")) else ""

        st.markdown(
            f"""<div style="
                border-left: 3px solid {sentiment_color};
                padding: 6px 10px;
                margin-bottom: 6px;
                background: #0F1627;
                border-radius: 0 6px 6px 0;
            ">
            <span style="color:#888;font-size:10px;font-family:'IBM Plex Mono'">
                {time_str} · {row['narrative']}
            </span><br>
            <a href="{row['link']}" target="_blank" style="
                color:#E8E0D0;
                font-size:12px;
                font-family:'IBM Plex Mono';
                text-decoration:none;
            ">{row['sentiment_emoji']} {row['title']}</a>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Simulated Elon Tweet Feed (real X API costs $$$) ──────────────────────────

SIMULATED_TWEETS = [
    {"text": "The thing I find most surprising is how few people understand the significance of what we're building at xAI", "time": "2h ago", "likes": 45200, "category": "🤖 AI/xAI"},
    {"text": "FSD 13 is shipping this quarter. Game over for legacy auto.", "time": "4h ago", "likes": 89300, "category": "🚗 FSD/Robotaxi"},
    {"text": "Optimus production scaling faster than expected. 1M robots by 2027 is conservative.", "time": "6h ago", "likes": 72100, "category": "🦾 Optimus"},
    {"text": "Interest rates need to come down. The data is clear.", "time": "8h ago", "likes": 33500, "category": "📊 Macro"},
    {"text": "Bitcoin is engineering genius. Don't fight it.", "time": "12h ago", "likes": 56800, "category": "₿ Crypto"},
    {"text": "Tesla Robotaxi launch in Austin next month. This changes everything.", "time": "1d ago", "likes": 112000, "category": "🚗 FSD/Robotaxi"},
    {"text": "Grok 4 performance is off the charts. Running final safety checks.", "time": "1d ago", "likes": 67400, "category": "🤖 AI/xAI"},
]


def get_simulated_tweets() -> list:
    """Return simulated tweet feed when X API unavailable."""
    return SIMULATED_TWEETS
