import numpy as np
import pandas as pd


def make_features(close, volume):

    """ Builds the features for every stock on every day.

    Parameters
    -----------
    close : DataFrame, closing prices, rows are days, columns are stocks.
    volume : DataFrame, shares traded, same shape as close.

    Returns
    --------
    dict of DataFrames, one per feature, same shape as close.

    Notes
    -----
    Each value only uses prices up to and including that day.
    A month is 21 trading days and a year is 252.
    """

    logp = np.log(close)
    ret = close.pct_change(fill_method=None)
    dollar_volume = close * volume

    features = {}
    features["mom_12_1"] = logp.shift(21) - logp.shift(252)
    features["mom_6_1"] = logp.shift(21) - logp.shift(126)
    features["mom_1m"] = logp - logp.shift(21)
    features["rev_1w"] = logp - logp.shift(5)
    features["vol_1m"] = ret.rolling(21).std()
    features["vol_3m"] = ret.rolling(63).std()
    features["max_ret_1m"] = ret.rolling(21).max()
    features["log_dvol"] = np.log(dollar_volume.rolling(60).mean())
    features["dist_52w_high"] = close / close.rolling(252).max() - 1
    return features


def forward_returns(close, hold=5):

    """ Return from tomorrow's close to `hold` days after that.

    Parameters
    -----------
    close : DataFrame, closing prices.
    hold : int, holding period in trading days.

    Returns
    --------
    DataFrame, same shape as close, NaN at the end of the record.

    Notes
    -----
    It starts tomorrow, so we never trade at the same price we used to
    make the decision.
    """

    return close.shift(-(1 + hold)) / close.shift(-1) - 1


def make_table(features, fwd_ret, dates):

    """ Makes the X / y table, one row per stock per week.

    Parameters
    -----------
    features : dict of DataFrames, from make_features.
    fwd_ret : DataFrame, from forward_returns.
    dates : list, the weekly dates to use.

    Returns
    --------
    DataFrame with columns ticker, date, the features, fwd_ret and target.

    Notes
    -----
    Features are ranked across the stocks on each date and scaled to
    -0.5 to 0.5, so the model only learns which stocks are better than
    the others that week. The target is the rank of next week's return.
    """

    names = list(features)
    rows = []
    for d in dates:
        table = pd.DataFrame({name: features[name].loc[d] for name in names})
        table["fwd_ret"] = fwd_ret.loc[d]
        table = table.dropna(subset=["mom_12_1", "fwd_ret"])

        for name in names:
            table[name] = table[name].rank(pct=True) - 0.5
        table[names] = table[names].fillna(0.0)

        table["target"] = table["fwd_ret"].rank(pct=True) - 0.5
        table["date"] = d
        rows.append(table)

    data = pd.concat(rows)
    data.index.name = "ticker"
    return data.reset_index()
