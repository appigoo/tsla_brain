"""
regime_engine.py — Market Regime Detection
Uses Hidden Markov Model + Volatility Clustering + Rule-based fallback
Streamlit Cloud compatible
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

REGIME_LABELS = {
    0: {"name": "RISK-ON 🟢", "color": "#00FF88", "emoji": "🟢"},
    1: {"name": "NEUTRAL ⚪", "color": "#888888", "emoji": "⚪"},
    2: {"name": "RISK-OFF 🔴", "color": "#FF4444", "emoji": "🔴"},
    3: {"name": "PANIC 🚨",    "color": "#FF0000", "emoji": "🚨"},
    4: {"name": "AI MANIA 🤖", "color": "#00D4FF", "emoji": "🤖"},
}


def _build_features(returns: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Build regime feature matrix."""
    feats = pd.DataFrame(index=returns.index)

    # TSLA momentum
    if "TSLA" in returns.columns:
        feats["tsla_ret_roll"] = returns["TSLA"].rolling(window).mean()
        feats["tsla_vol"] = returns["TSLA"].rolling(window).std()

    # VIX proxy
    if "VIX" in returns.columns:
        feats["vix_ret"] = returns["VIX"].rolling(window).mean()

    # Broad market
    for col in ["SPY", "QQQ"]:
        if col in returns.columns:
            feats[f"{col}_roll"] = returns[col].rolling(window).mean()

    # Tech vs defensive
    if "TLT" in returns.columns:
        feats["tlt_roll"] = returns["TLT"].rolling(window).mean()

    # AI/tech proxy
    if "NVDA" in returns.columns and "TSLA" in returns.columns:
        feats["ai_momentum"] = (
            returns["NVDA"].rolling(window).mean() +
            returns["TSLA"].rolling(window).mean()
        )

    # BTC risk appetite
    if "BTC" in returns.columns:
        feats["btc_roll"] = returns["BTC"].rolling(window).mean()

    return feats.dropna()


def detect_regime_kmeans(returns: pd.DataFrame, n_clusters: int = 5) -> dict:
    """
    KMeans-based regime detection.
    Returns current regime + history.
    """
    feats = _build_features(returns)
    if feats.empty or len(feats) < 20:
        return _fallback_regime(returns)

    scaler = StandardScaler()
    X = scaler.fit_transform(feats.fillna(0))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Map cluster IDs to regime meaning via centroid analysis
    centers = km.cluster_centers_
    regime_map = _assign_regime_labels(centers, feats.columns.tolist())

    mapped = [regime_map.get(l, 1) for l in labels]

    history = pd.DataFrame({
        "date": feats.index,
        "raw_cluster": labels,
        "regime_id": mapped,
    }).set_index("date")
    history["regime_name"] = history["regime_id"].map(
        lambda x: REGIME_LABELS.get(x, REGIME_LABELS[1])["name"]
    )
    history["regime_color"] = history["regime_id"].map(
        lambda x: REGIME_LABELS.get(x, REGIME_LABELS[1])["color"]
    )

    current_id = mapped[-1] if mapped else 1
    current = REGIME_LABELS.get(current_id, REGIME_LABELS[1])

    return {
        "current_id": current_id,
        "current_name": current["name"],
        "current_color": current["color"],
        "current_emoji": current["emoji"],
        "history": history,
        "confidence": _regime_confidence(X[-1], km, current_id),
    }


def _assign_regime_labels(centers: np.ndarray, feature_names: list) -> dict:
    """Heuristically assign regime meaning to cluster IDs."""
    mapping = {}
    n = len(centers)

    # Score each cluster
    scores = []
    for i, c in enumerate(centers):
        score = {}
        feat_map = {f: idx for idx, f in enumerate(feature_names)}

        # Risk-on score: high TSLA, high QQQ, low VIX
        risk_on = 0.0
        if "tsla_ret_roll" in feat_map:
            risk_on += c[feat_map["tsla_ret_roll"]]
        if "QQQ_roll" in feat_map:
            risk_on += c[feat_map["QQQ_roll"]]
        if "vix_ret" in feat_map:
            risk_on -= c[feat_map["vix_ret"]]
        score["risk_on"] = risk_on

        # AI mania: very high ai_momentum + BTC up
        ai_score = 0.0
        if "ai_momentum" in feat_map:
            ai_score += c[feat_map["ai_momentum"]] * 2
        if "btc_roll" in feat_map:
            ai_score += c[feat_map["btc_roll"]]
        score["ai_mania"] = ai_score

        # Risk-off: high VIX, negative TSLA
        risk_off = 0.0
        if "vix_ret" in feat_map:
            risk_off += c[feat_map["vix_ret"]]
        if "tsla_ret_roll" in feat_map:
            risk_off -= c[feat_map["tsla_ret_roll"]]
        if "tlt_roll" in feat_map:
            risk_off += c[feat_map["tlt_roll"]]
        score["risk_off"] = risk_off

        # Panic: extreme risk-off + high volatility
        panic = 0.0
        if "tsla_vol" in feat_map:
            panic += c[feat_map["tsla_vol"]]
        if "vix_ret" in feat_map:
            panic += c[feat_map["vix_ret"]] * 2
        score["panic"] = panic

        scores.append(score)

    # Assign regimes
    assigned = {}
    used = set()

    # Find panic (highest panic score)
    panic_idx = max(range(n), key=lambda i: scores[i]["panic"] if i not in used else -999)
    assigned[panic_idx] = 3
    used.add(panic_idx)

    # Find AI mania
    ai_idx = max(range(n), key=lambda i: scores[i]["ai_mania"] if i not in used else -999)
    assigned[ai_idx] = 4
    used.add(ai_idx)

    # Find strongest risk-on
    ron_idx = max(range(n), key=lambda i: scores[i]["risk_on"] if i not in used else -999)
    assigned[ron_idx] = 0
    used.add(ron_idx)

    # Find strongest risk-off
    roff_idx = max(range(n), key=lambda i: scores[i]["risk_off"] if i not in used else -999)
    assigned[roff_idx] = 2
    used.add(roff_idx)

    # Remaining = neutral
    for i in range(n):
        if i not in used:
            assigned[i] = 1

    return assigned


def _regime_confidence(x_latest: np.ndarray, km, regime_id: int) -> float:
    """Confidence based on distance to cluster center."""
    try:
        distances = np.linalg.norm(km.cluster_centers_ - x_latest, axis=1)
        min_dist = distances.min()
        second_min = np.sort(distances)[1]
        if second_min == 0:
            return 1.0
        confidence = 1 - (min_dist / (min_dist + second_min))
        return round(float(confidence), 2)
    except Exception:
        return 0.5


def _fallback_regime(returns: pd.DataFrame) -> dict:
    """Rule-based fallback regime."""
    if returns.empty:
        return {
            "current_id": 1, "current_name": "NEUTRAL ⚪",
            "current_color": "#888888", "current_emoji": "⚪",
            "history": pd.DataFrame(), "confidence": 0.5
        }
    # Simple rule
    tsla_5d = returns["TSLA"].tail(5).mean() if "TSLA" in returns.columns else 0
    vix = returns.get("VIX", pd.Series([0])).tail(5).mean() if "VIX" in returns.columns else 0

    if tsla_5d > 0.01 and vix < 0:
        regime_id = 0  # Risk-on
    elif vix > 0.02 or tsla_5d < -0.02:
        regime_id = 3  # Panic
    elif tsla_5d < -0.005:
        regime_id = 2  # Risk-off
    else:
        regime_id = 1  # Neutral

    r = REGIME_LABELS[regime_id]
    return {
        "current_id": regime_id,
        "current_name": r["name"],
        "current_color": r["color"],
        "current_emoji": r["emoji"],
        "history": pd.DataFrame(),
        "confidence": 0.6,
    }


def compute_volatility_clusters(returns: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling volatility for regime overlay."""
    result = pd.DataFrame()
    for col in returns.columns:
        result[col] = returns[col].rolling(window).std() * np.sqrt(252)
    return result.dropna()


def detect_correlation_breakdown(returns: pd.DataFrame,
                                  window_short: int = 5,
                                  window_long: int = 20) -> pd.DataFrame:
    """
    Detect when correlations are breaking down (regime shift signal).
    """
    if returns.empty or len(returns) < window_long:
        return pd.DataFrame()

    short_corr = returns.tail(window_short).corr()
    long_corr = returns.tail(window_long).corr()

    breakdowns = []
    for col in returns.columns:
        if col == "TSLA":
            continue
        sc = short_corr["TSLA"].get(col, 0)
        lc = long_corr["TSLA"].get(col, 0)
        delta = sc - lc
        breakdowns.append({
            "symbol": col,
            "short_corr": round(float(sc), 3),
            "long_corr": round(float(lc), 3),
            "delta": round(float(delta), 3),
            "signal": "⚠️ Breaking Down" if delta < -0.2 else (
                "🔥 Strengthening" if delta > 0.2 else "→ Stable"
            ),
        })

    return pd.DataFrame(breakdowns).sort_values("delta")
