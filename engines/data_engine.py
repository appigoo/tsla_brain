"""
data_engine.py — Market Data Fetcher
Streamlit Cloud compatible (no websockets, poll-based)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

# ── Symbol Registry ────────────────────────────────────────────────────────────
LAYER_CONFIG = {
    "Mega Cap": {
        "color": "#E31937",       # Tesla red
        "symbols": ["TSLA", "NVDA", "META", "AAPL", "MSFT", "AMD", "AMZN", "GOOGL"],
        "size_base": 40,
    },
    "ETF": {
        "color": "#00D4FF",
        "symbols": ["QQQ", "SPY", "ARKK", "SMH", "XLY", "IWM"],
        "size_base": 30,
    },
    "Macro": {
        "color": "#FFB800",
        "symbols": ["^VIX", "TLT", "GLD", "USO", "^TNX"],
        "size_base": 28,
    },
    "Crypto": {
        "color": "#FF6B35",
        "symbols": ["BTC-USD", "ETH-USD", "COIN", "MSTR"],
        "size_base": 25,
    },
}

DISPLAY_NAMES = {
    "^VIX": "VIX", "^TNX": "TNX", "BTC-USD": "BTC", "ETH-USD": "ETH"
}

ALL_SYMBOLS = []
SYMBOL_TO_LAYER = {}
for layer, cfg in LAYER_CONFIG.items():
    for s in cfg["symbols"]:
        ALL_SYMBOLS.append(s)
        SYMBOL_TO_LAYER[s] = layer


def display_name(sym: str) -> str:
    return DISPLAY_NAMES.get(sym, sym)


@st.cache_data(ttl=300)
def fetch_price_data(period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV for all symbols. Returns wide DataFrame of Close prices."""
    try:
        raw = yf.download(
            ALL_SYMBOLS,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw

        # Rename ^VIX etc
        close.columns = [display_name(c) for c in close.columns]
        close = close.ffill().dropna(how="all")
        return close
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_intraday(symbol: str = "TSLA", period: str = "5d", interval: str = "15m") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_current_prices() -> dict:
    """Snapshot current prices for all symbols."""
    prices = {}
    try:
        tickers = yf.Tickers(" ".join(ALL_SYMBOLS))
        for sym in ALL_SYMBOLS:
            try:
                info = tickers.tickers[sym].fast_info
                prices[display_name(sym)] = {
                    "price": getattr(info, "last_price", None),
                    "change_pct": getattr(info, "three_month_return", None),
                    "volume": getattr(info, "three_month_average_volume", None),
                    "market_cap": getattr(info, "market_cap", None),
                }
            except Exception:
                prices[display_name(sym)] = {"price": None, "change_pct": None}
    except Exception:
        pass
    return prices


def get_returns(close_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute log returns."""
    return np.log(close_df / close_df.shift(1)).dropna()


def get_rolling_volatility(close_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualised rolling volatility for TSLA."""
    if "TSLA" not in close_df.columns:
        return pd.Series(dtype=float)
    ret = np.log(close_df["TSLA"] / close_df["TSLA"].shift(1))
    return ret.rolling(window).std() * np.sqrt(252)
