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
    """
    Fetch Close prices for all symbols.
    Handles all known yfinance MultiIndex column layouts robustly.
    """
    try:
        # Try curl_cffi session first (better rate limit bypass)
        session = None
        try:
            from curl_cffi import requests as cf_req
            session = cf_req.Session(impersonate="chrome110")
        except Exception:
            pass

        raw = yf.download(
            ALL_SYMBOLS,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
            session=session,
        )

        close = _extract_close(raw)

        if close.empty or "TSLA" not in close.columns:
            return _fetch_fallback(period, session)

        close = close.ffill().dropna(how="all")
        return close

    except Exception as e:
        st.warning(f"Batch fetch failed ({e}), trying individual...")
        return _fetch_fallback(period)


def _extract_close(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Close column(s) from yfinance raw output.
    Handles:
      - MultiIndex (Price, Ticker)  — yfinance >= 0.2.50 default
      - MultiIndex (Ticker, Price)  — group_by='ticker'
      - Flat columns                — single ticker
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    frames = {}

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = list(raw.columns.get_level_values(0).unique())
        lvl1 = list(raw.columns.get_level_values(1).unique())

        # Case A: (Price, Ticker) — e.g. ("Close", "TSLA")
        if "Close" in lvl0:
            close_block = raw["Close"]
            if isinstance(close_block, pd.Series):
                close_block = close_block.to_frame(name=lvl1[0] if lvl1 else "TSLA")
            for col in close_block.columns:
                name = display_name(str(col))
                frames[name] = close_block[col]

        # Case B: (Ticker, Price) — e.g. ("TSLA", "Close")
        elif "Close" in lvl1:
            for sym in lvl0:
                try:
                    s = raw[sym]["Close"]
                    frames[display_name(str(sym))] = s
                except Exception:
                    continue

        # Case C: price fields without "Close" — try lowercase
        elif "close" in [str(x).lower() for x in lvl0]:
            for col0, col1 in raw.columns:
                if str(col0).lower() == "close":
                    frames[display_name(str(col1))] = raw[(col0, col1)]
        elif "close" in [str(x).lower() for x in lvl1]:
            for col0, col1 in raw.columns:
                if str(col1).lower() == "close":
                    frames[display_name(str(col0))] = raw[(col0, col1)]

    else:
        # Flat columns — single ticker download
        for candidate in ["Close", "close"]:
            if candidate in raw.columns:
                # Try to detect which symbol from context
                frames["TSLA"] = raw[candidate]
                break

    return pd.DataFrame(frames) if frames else pd.DataFrame()


def _fetch_fallback(period: str = "3mo", session=None) -> pd.DataFrame:
    """Fetch each symbol individually — reliable fallback."""
    frames = {}
    for sym in ALL_SYMBOLS:
        try:
            kwargs = dict(period=period, interval="1d",
                         auto_adjust=True, progress=False)
            if session:
                kwargs["session"] = session
            df = yf.download(sym, **kwargs)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for candidate in ["Close", "close"]:
                if candidate in df.columns:
                    s = df[candidate].dropna()
                    if not s.empty:
                        frames[display_name(sym)] = s
                    break
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).ffill().dropna(how="all")


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
