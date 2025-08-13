import numpy as np
import pandas as pd

def _var_cvar(returns: pd.Series, alpha=0.95):
    r = returns.dropna()
    if r.empty:
        return np.nan, np.nan
    q = r.quantile(1-alpha)
    cvar = r[r <= q].mean() if (r[r <= q].size>0) else np.nan
    return float(-q), float(-cvar)

def _max_drawdown(prices: pd.Series):
    roll = prices.cummax()
    dd = prices/roll - 1.0
    return float(dd.min())

def risk_report(prices: pd.DataFrame, benchmark: str = 'SPY'):
    df = prices.copy()
    df['ret'] = df['Close'].pct_change()
    var95, cvar95 = _var_cvar(df['ret'], 0.95)
    mdd = _max_drawdown(df['Close'])
    ann_vol = float(df['ret'].rolling(20).std().iloc[-1] * (252**0.5)) if df['ret'].notna().any() else np.nan
    return {
        "ann_vol_20d": ann_vol,
        "VaR_95_daily": var95,
        "CVaR_95_daily": cvar95,
        "max_drawdown": mdd,
        "notes": "Historical estimates; not predictive; use with caution."
    }
