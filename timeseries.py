import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime as dt

def load_prices(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
        return None
    except Exception:
        return None

def _rolling_beta(returns, benchmark_returns, win=60):
    # simple rolling OLS beta
    betas = []
    idx = returns.index
    for i in range(win, len(returns)):
        x = benchmark_returns.iloc[i-win:i].values.reshape(-1,1)
        y = returns.iloc[i-win:i].values
        if np.isfinite(x).all() and np.isfinite(y).all():
            m = LinearRegression().fit(x, y)
            betas.append(m.coef_[0])
        else:
            betas.append(np.nan)
    ser = pd.Series([np.nan]*win + betas, index=idx)
    return ser

def features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df['ret'] = df['Close'].pct_change()
    df['vol_20'] = df['ret'].rolling(20).std() * np.sqrt(252)
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['ma_200'] = df['Close'].rolling(200).mean()
    # beta vs SPY
    spy = yf.download('SPY', start=df.index.min(), end=df.index.max(), auto_adjust=True, progress=False)
    if not spy.empty:
        spy_ret = spy['Close'].pct_change()
        df['beta_60'] = _rolling_beta(df['ret'], spy_ret, win=60)
    # drawdown
    roll_max = df['Close'].cummax()
    df['drawdown'] = df['Close']/roll_max - 1.0
    return df

def features_summary(prices: pd.DataFrame) -> pd.DataFrame:
    f = features(prices)
    last = f.iloc[-1]
    rows = {
        "Price": prices['Close'].iloc[-1],
        "YTD Return": (prices['Close'].iloc[-1]/prices['Close'].iloc[0]-1) if len(prices)>1 else np.nan,
        "Ann Vol (20d)": last.get("vol_20", np.nan),
        "Drawdown": last.get("drawdown", np.nan),
        "MA20>MA50": bool(last.get("ma_20", np.nan) > last.get("ma_50", np.nan)),
        "Beta(60d)": last.get("beta_60", np.nan),
    }
    return pd.DataFrame(rows, index=["Value"]).T

def plot_prices(prices: pd.DataFrame):
    f = features(prices)
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.plot(prices.index, prices['Close'], label='Close')
    ax.plot(prices.index, f['ma_20'], label='MA20')
    ax.plot(prices.index, f['ma_50'], label='MA50')
    ax.plot(prices.index, f['ma_200'], label='MA200')
    ax.legend()
    ax.set_title("Price with Moving Averages")
    return fig
