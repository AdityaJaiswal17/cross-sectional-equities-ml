from pathlib import Path

import numpy as np
import pandas as pd

# today's large caps, so the backtest has survivorship bias (see README)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CSCO", "INTC",
    "IBM", "ADBE", "CRM", "QCOM", "TXN", "AMD", "AVGO", "MU", "AMAT", "NFLX",
    "CMCSA", "T", "VZ", "DIS", "ACN", "ADP",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW", "USB",
    "PNC", "BK", "MET", "AIG", "CB", "MMC", "SPGI", "CME",
    "JNJ", "PFE", "MRK", "ABT", "TMO", "UNH", "LLY", "BMY", "AMGN", "GILD",
    "MDT", "CVS", "DHR", "SYK", "BDX", "ISRG",
    "PG", "KO", "PEP", "WMT", "COST", "HD", "LOW", "MCD", "NKE", "SBUX",
    "TGT", "MO", "PM", "CL", "KMB", "EL", "TJX", "BKNG", "YUM", "GIS",
    "GE", "HON", "MMM", "CAT", "DE", "UPS", "FDX", "LMT", "RTX", "BA",
    "UNP", "CSX", "NSC", "EMR", "ITW", "LIN", "APD", "SHW", "ECL", "NEM",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "NEE", "DUK", "SO", "D",
    "AEP", "EXC", "AMT", "PLD", "SPG",
]


def load_prices(start="2010-01-01", end="2025-01-01", cache_dir="data"):

    """ Downloads daily prices and volumes from Yahoo Finance.

    Parameters
    -----------
    start : str, first date.
    end : str, last date.
    cache_dir : str, folder where the download is saved.

    Returns
    --------
    close, volume : DataFrames, rows are days, columns are stocks.
        Prices are adjusted for splits and dividends.

    Notes
    -----
    The download is saved the first time, so later runs are offline.
    Stocks missing more than 20% of the days are dropped.
    """

    cache = Path(cache_dir) / f"prices_{start}_{end}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    import yfinance as yf

    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)
    close = raw["Close"]
    volume = raw["Volume"]

    keep = close.columns[close.isna().mean() <= 0.2]
    close = close[keep].dropna(how="all")
    volume = volume[keep].reindex(close.index)

    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((close, volume), cache)
    return close, volume


def make_synthetic(n_stocks=100, n_days=3024, signal=0.0015, seed=0):

    """ Makes fake prices with a hidden signal, to test the pipeline.

    Parameters
    -----------
    n_stocks : int, number of stocks.
    n_days : int, number of trading days.
    signal : float, strength of the hidden drift, 0 for none.
    seed : int, random seed.

    Returns
    --------
    close, volume : DataFrames, rows are days, columns are stocks.

    Notes
    -----
    Each stock has a slow, random drift, so stocks that went up keep
    going up for a while. The models should find this. With signal=0
    there is nothing to find and the IC should be about 0.
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n_days)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]

    # slow drift with a half life of about six months
    phi = 0.5 ** (1 / 126)
    drift = np.zeros((n_days, n_stocks))
    drift[0] = rng.normal(0, 1, n_stocks)
    for t in range(1, n_days):
        drift[t] = (phi * drift[t - 1]
                    + np.sqrt(1 - phi ** 2) * rng.normal(0, 1, n_stocks))

    market = rng.normal(0.0003, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    noise = rng.uniform(0.01, 0.025, n_stocks)
    ret = (market[:, None] * beta + signal * drift
           + rng.normal(0, 1, (n_days, n_stocks)) * noise)

    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)),
                         index=dates, columns=tickers)
    base = rng.lognormal(15, 1, n_stocks)
    volume = pd.DataFrame(base * rng.lognormal(0, 0.3, (n_days, n_stocks)),
                          index=dates, columns=tickers)
    return close, volume
