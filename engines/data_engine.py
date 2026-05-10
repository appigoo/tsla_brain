"""
data_engine.py — 市場數據引擎
支援 yfinance 0.2.61 + curl_cffi（繞過 Yahoo 限速）
Streamlit Cloud 兼容
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

# ── 資產配置 ─────────────────────────────────────────────────────────────────
LAYER_CONFIG = {
    "大型科技": {
        "color": "#E31937",
        "symbols": ["TSLA", "NVDA", "META", "AAPL", "MSFT", "AMD", "AMZN", "GOOGL"],
        "size_base": 40,
    },
    "ETF": {
        "color": "#00D4FF",
        "symbols": ["QQQ", "SPY", "ARKK", "SMH", "XLY", "IWM"],
        "size_base": 30,
    },
    "宏觀": {
        "color": "#FFB800",
        "symbols": ["^VIX", "TLT", "GLD", "USO", "^TNX"],
        "size_base": 28,
    },
    "加密": {
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


def _get_curl_session():
    """取得 curl_cffi session 繞過 Yahoo 限速。"""
    try:
        from curl_cffi import requests as cf_req
        return cf_req.Session(impersonate="chrome110")
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_price_data(period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    取得所有資產收盤價。
    策略：
    1. curl_cffi session + multi_level_index=False
    2. 標準 batch download + MultiIndex 解析
    3. 逐個下載 fallback
    """
    session = _get_curl_session()

    # 方法 1：multi_level_index=False
    try:
        kwargs = dict(
            period=period, interval=interval,
            auto_adjust=True, progress=False,
            threads=True, multi_level_index=False,
        )
        if session:
            kwargs["session"] = session
        raw = yf.download(ALL_SYMBOLS, **kwargs)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                close = _parse_multiindex(raw)
            elif "Close" in raw.columns:
                block = raw["Close"]
                if isinstance(block, pd.DataFrame):
                    close = block.copy()
                    close.columns = [display_name(str(c)) for c in close.columns]
                else:
                    close = pd.DataFrame({"TSLA": block})
            else:
                close = pd.DataFrame()
            if not close.empty and "TSLA" in close.columns:
                close = close.ffill().dropna(how="all")
                if close["TSLA"].dropna().shape[0] > 5:
                    return close
    except Exception:
        pass

    # 方法 2：標準 batch
    try:
        kwargs2 = dict(period=period, interval=interval,
                       auto_adjust=True, progress=False, threads=True)
        if session:
            kwargs2["session"] = session
        raw2 = yf.download(ALL_SYMBOLS, **kwargs2)
        close2 = _parse_multiindex(raw2)
        if not close2.empty and "TSLA" in close2.columns:
            return close2.ffill().dropna(how="all")
    except Exception:
        pass

    # 方法 3：逐個下載
    return _fetch_one_by_one(period, session)


def _parse_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
    """解析所有已知的 yfinance MultiIndex 格式。"""
    if raw is None or raw.empty:
        return pd.DataFrame()

    frames = {}

    if not isinstance(raw.columns, pd.MultiIndex):
        for c in ["Close", "close"]:
            if c in raw.columns:
                frames["TSLA"] = raw[c]
                break
        return pd.DataFrame(frames)

    lvl0 = list(raw.columns.get_level_values(0).unique())
    lvl1 = list(raw.columns.get_level_values(1).unique())

    # (Price, Ticker) — yfinance 0.2.50+ 預設
    if "Close" in lvl0:
        block = raw["Close"]
        if isinstance(block, pd.Series):
            name = display_name(str(lvl1[0])) if lvl1 else "TSLA"
            frames[name] = block
        else:
            for col in block.columns:
                frames[display_name(str(col))] = block[col]
    # (Ticker, Price)
    elif "Close" in lvl1:
        for sym in lvl0:
            try:
                frames[display_name(str(sym))] = raw[sym]["Close"]
            except Exception:
                pass
    else:
        for c0, c1 in raw.columns:
            if str(c0).lower() == "close":
                frames[display_name(str(c1))] = raw[(c0, c1)]
            elif str(c1).lower() == "close":
                frames[display_name(str(c0))] = raw[(c0, c1)]

    return pd.DataFrame(frames) if frames else pd.DataFrame()


def _fetch_one_by_one(period: str, session=None) -> pd.DataFrame:
    """逐個 ticker 下載（最穩定 fallback）。"""
    frames = {}
    for sym in ALL_SYMBOLS:
        try:
            kwargs = dict(period=period, interval="1d",
                          auto_adjust=True, progress=False,
                          multi_level_index=False)
            if session:
                kwargs["session"] = session
            df = yf.download(sym, **kwargs)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for c in ["Close", "close"]:
                if c in df.columns:
                    s = df[c].dropna()
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
        session = _get_curl_session()
        kwargs = dict(period=period, interval=interval,
                      auto_adjust=True, progress=False, multi_level_index=False)
        if session:
            kwargs["session"] = session
        df = yf.download(symbol, **kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def get_returns(close_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """計算對數報酬率。"""
    if close_df.empty:
        return pd.DataFrame()
    return np.log(close_df / close_df.shift(1)).dropna()


def get_rolling_volatility(close_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """TSLA 年化滾動波動率。"""
    if "TSLA" not in close_df.columns:
        return pd.Series(dtype=float)
    ret = np.log(close_df["TSLA"] / close_df["TSLA"].shift(1))
    return ret.rolling(window).std() * np.sqrt(252)
