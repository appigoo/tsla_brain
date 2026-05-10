"""
correlation_engine.py — Dynamic Correlation + Lead-Lag + Hidden Relationships
Pure Python / sklearn / scipy — Streamlit Cloud compatible
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


# ── Rolling Correlation Matrix ─────────────────────────────────────────────────

def compute_rolling_correlation(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Latest rolling correlation snapshot."""
    if returns.empty or len(returns) < window:
        corr = returns.corr()
    else:
        subset = returns.tail(window)
        corr = subset.corr()
    # Fill NaN with 0 so downstream never sees NaN correlations
    return corr.fillna(0)


def correlation_timeseries(returns: pd.DataFrame, target: str, window: int = 20) -> pd.DataFrame:
    """Rolling correlation of all assets vs target."""
    if target not in returns.columns:
        return pd.DataFrame()
    cols = [c for c in returns.columns if c != target]
    result = {}
    for col in cols:
        pair = returns[[target, col]].dropna()
        if len(pair) >= window:
            result[col] = pair[target].rolling(window).corr(pair[col])
    return pd.DataFrame(result)


# ── Lead-Lag Engine ────────────────────────────────────────────────────────────

def compute_lead_lag(returns: pd.DataFrame, target: str = "TSLA",
                     max_lag: int = 5) -> pd.DataFrame:
    """
    Cross-correlation to find lead-lag relationships with TSLA.
    Returns DataFrame: symbol | best_lag | best_corr | direction
    positive lag = symbol LEADS TSLA
    negative lag = symbol FOLLOWS TSLA
    """
    if target not in returns.columns:
        return pd.DataFrame()

    results = []
    tsla = returns[target].dropna()

    for col in returns.columns:
        if col == target:
            continue
        other = returns[col].dropna()
        aligned = pd.concat([tsla, other], axis=1).dropna()
        if len(aligned) < max_lag * 3:
            continue

        t = aligned.iloc[:, 0].values
        o = aligned.iloc[:, 1].values

        best_lag, best_corr = 0, 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                c = np.corrcoef(t[lag:], o[:-lag])[0, 1] if lag < len(t) else 0
            elif lag < 0:
                c = np.corrcoef(t[:lag], o[-lag:])[0, 1] if -lag < len(t) else 0
            else:
                c = np.corrcoef(t, o)[0, 1]
            if abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag

        direction = "leads TSLA" if best_lag > 0 else ("follows TSLA" if best_lag < 0 else "coincident")
        results.append({
            "symbol": col,
            "best_lag_bars": best_lag,
            "correlation": round(best_corr, 4),
            "direction": direction,
            "influence_score": round(abs(best_corr) * 100, 1),
        })

    if not results:
        return pd.DataFrame(columns=["symbol","best_lag_bars","correlation","direction","influence_score"])
    df = pd.DataFrame(results).sort_values("influence_score", ascending=False)
    return df


# ── Granger Causality (simplified) ────────────────────────────────────────────

def granger_simple(x: np.ndarray, y: np.ndarray, max_lag: int = 3) -> float:
    """
    Simplified Granger: does x help predict y?
    Returns F-statistic proxy (higher = more causal).
    """
    n = len(y)
    if n < max_lag * 4:
        return 0.0

    # Restricted: y ~ y_lagged
    Y = y[max_lag:]
    X_r = np.column_stack([y[max_lag - l - 1:n - l - 1] for l in range(max_lag)])

    # Unrestricted: y ~ y_lagged + x_lagged
    X_u = np.column_stack([X_r] + [x[max_lag - l - 1:n - l - 1] for l in range(max_lag)])

    def ols_sse(X, Y):
        try:
            beta = np.linalg.lstsq(
                np.column_stack([np.ones(len(X)), X]), Y, rcond=None
            )[0]
            resid = Y - np.column_stack([np.ones(len(X)), X]) @ beta
            return np.sum(resid ** 2)
        except Exception:
            return np.var(Y) * len(Y)

    sse_r = ols_sse(X_r, Y)
    sse_u = ols_sse(X_u, Y)
    if sse_u == 0:
        return 0.0
    f_stat = ((sse_r - sse_u) / max_lag) / (sse_u / (len(Y) - 2 * max_lag - 1))
    return max(0.0, float(f_stat))


def compute_granger_scores(returns: pd.DataFrame, target: str = "TSLA",
                            max_lag: int = 3) -> pd.DataFrame:
    """Granger causality scores: who causes TSLA moves?"""
    if target not in returns.columns:
        return pd.DataFrame()

    tsla = returns[target].dropna().values
    results = []

    for col in returns.columns:
        if col == target:
            continue
        other = returns[col].dropna()
        aligned = pd.concat([returns[target], other], axis=1).dropna()
        if len(aligned) < 20:
            continue
        t_vals = aligned.iloc[:, 0].values
        o_vals = aligned.iloc[:, 1].values

        # x → TSLA
        f_x_causes_tsla = granger_simple(o_vals, t_vals, max_lag)
        # TSLA → x
        f_tsla_causes_x = granger_simple(t_vals, o_vals, max_lag)

        results.append({
            "symbol": col,
            "causes_tsla_f": round(f_x_causes_tsla, 2),
            "tsla_causes_f": round(f_tsla_causes_x, 2),
            "net_influence": round(f_x_causes_tsla - f_tsla_causes_x, 2),
        })

    if not results:
        return pd.DataFrame(columns=["symbol","causes_tsla_f","tsla_causes_f","net_influence"])
    return pd.DataFrame(results).sort_values("causes_tsla_f", ascending=False)


# ── Mutual Information (Hidden Relationships) ─────────────────────────────────

def mutual_information_score(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Discretised mutual information — detects non-linear relationships."""
    try:
        x_d = pd.cut(x, bins=bins, labels=False, duplicates="drop")
        y_d = pd.cut(y, bins=bins, labels=False, duplicates="drop")
        df = pd.DataFrame({"x": x_d, "y": y_d}).dropna()
        if len(df) < 10:
            return 0.0
        joint = pd.crosstab(df["x"], df["y"], normalize=True)
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        mi = 0.0
        for i in joint.index:
            for j in joint.columns:
                pxy = joint.loc[i, j]
                if pxy > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy * np.log(pxy / (px[i] * py[j]))
        return max(0.0, float(mi))
    except Exception:
        return 0.0


def compute_hidden_relationships(returns: pd.DataFrame, target: str = "TSLA") -> pd.DataFrame:
    """Find non-linear hidden relationships via MI."""
    if target not in returns.columns:
        return pd.DataFrame()

    tsla = returns[target].dropna()
    results = []

    for col in returns.columns:
        if col == target:
            continue
        aligned = pd.concat([tsla, returns[col]], axis=1).dropna()
        if len(aligned) < 20:
            continue
        x = aligned.iloc[:, 0].values
        y = aligned.iloc[:, 1].values

        linear_corr = abs(np.corrcoef(x, y)[0, 1])
        mi = mutual_information_score(x, y)
        # Hidden = MI captures what linear correlation misses
        hidden_score = max(0.0, mi - linear_corr * 0.5)

        results.append({
            "symbol": col,
            "linear_corr": round(linear_corr, 4),
            "mutual_info": round(mi, 4),
            "hidden_score": round(hidden_score, 4),
            "relationship_type": "Non-linear" if hidden_score > 0.05 else "Linear",
        })

    if not results:
        return pd.DataFrame(columns=["symbol","linear_corr","mutual_info","hidden_score","relationship_type"])
    return pd.DataFrame(results).sort_values("hidden_score", ascending=False)


# ── Centrality Scores ──────────────────────────────────────────────────────────

def compute_centrality(corr_matrix: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Graph centrality from correlation matrix.
    Returns degree, weighted degree (strength), betweenness proxy.
    """
    adj = (corr_matrix.abs() > threshold).astype(float) * corr_matrix.abs()
    adj_vals = adj.values.copy()
    np.fill_diagonal(adj_vals, 0)
    adj = pd.DataFrame(adj_vals, index=adj.index, columns=adj.columns)

    degree = (adj > 0).sum(axis=1)
    strength = adj.sum(axis=1)

    # Eigenvector centrality via power iteration
    n = len(adj)
    v = np.ones(n) / n
    for _ in range(100):
        v_new = adj.values @ v
        norm = np.linalg.norm(v_new)
        if norm == 0:
            break
        v_new /= norm
        if np.allclose(v, v_new, atol=1e-6):
            break
        v = v_new

    result = pd.DataFrame({
        "symbol": corr_matrix.columns,
        "degree": degree.values,
        "strength": strength.values.round(4),
        "eigenvector": v.round(4),
    }).sort_values("eigenvector", ascending=False)
    return result


# ── Risk Contagion ─────────────────────────────────────────────────────────────

def compute_risk_contagion(returns: pd.DataFrame, shock_asset: str = "TSLA",
                            shock_size: float = -0.05) -> pd.DataFrame:
    """
    Simulate TSLA shock transmission.
    Returns expected impact on each asset.
    """
    if shock_asset not in returns.columns:
        return pd.DataFrame()

    corr = returns.corr()
    tsla_corr = corr[shock_asset].drop(shock_asset)
    vol = returns.std()
    tsla_vol = vol.get(shock_asset, 0.02)

    results = []
    for sym in tsla_corr.index:
        beta = tsla_corr[sym] * (vol.get(sym, 0.02) / tsla_vol) if tsla_vol > 0 else 0
        expected_impact = beta * shock_size
        contagion_score = abs(beta * tsla_corr[sym])
        results.append({
            "symbol": sym,
            "beta_to_tsla": round(float(beta), 4),
            "expected_impact_pct": round(float(expected_impact) * 100, 2),
            "contagion_score": round(float(contagion_score), 4),
            "direction": "↓ Infected" if expected_impact < -0.01 else (
                "↑ Inverse" if expected_impact > 0.01 else "→ Neutral"
            ),
        })

    if not results:
        return pd.DataFrame(columns=["symbol","beta_to_tsla","expected_impact_pct","contagion_score","direction"])
    return pd.DataFrame(results).sort_values("contagion_score", ascending=False)
