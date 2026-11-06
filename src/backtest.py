import numpy as np
import pandas as pd


def information_coefficient(preds, model):

    """ Spearman correlation between predicted and real returns each week.

    Parameters
    -----------
    preds : DataFrame, from walk_forward.
    model : str, the prediction column.

    Returns
    --------
    Series, one IC per week, indexed by date.
    """

    ics = []
    dates = []
    for d, week in preds.groupby("date"):
        ics.append(week[model].corr(week["fwd_ret"], method="spearman"))
        dates.append(d)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def long_short(preds, model, cost=0.001):

    """ Buys the top 10% and shorts the bottom 10% every week.

    Parameters
    -----------
    preds : DataFrame, from walk_forward.
    model : str, the prediction column.
    cost : float, cost per unit of turnover (0.001 is 10 basis points).

    Returns
    --------
    Series, the weekly profit after costs, indexed by date.

    Notes
    -----
    Each side is equally weighted and adds up to 1, so 1 is long and
    1 is short. Turnover is how much of the portfolio changed since
    last week, and building it in the first week counts as turnover too.
    """

    profits = []
    dates = []
    old_weights = pd.Series(dtype=float)
    for d, week in preds.groupby("date"):
        week = week.set_index("ticker")
        rank = week[model].rank(pct=True)
        weights = pd.Series(0.0, index=week.index)
        weights[rank > 0.9] = 1 / (rank > 0.9).sum()
        weights[rank <= 0.1] = -1 / (rank <= 0.1).sum()

        both = weights.index.union(old_weights.index)
        turnover = (weights.reindex(both, fill_value=0)
                    - old_weights.reindex(both, fill_value=0)).abs().sum()

        profits.append((weights * week["fwd_ret"]).sum() - turnover * cost)
        dates.append(d)
        old_weights = weights
    return pd.Series(profits, index=pd.DatetimeIndex(dates))


def summary(preds, models, hold=5, cost=0.001):

    """ The main numbers for each model.

    Parameters
    -----------
    preds : DataFrame, from walk_forward.
    models : list, the prediction columns.
    hold : int, holding period in trading days.
    cost : float, cost per unit of turnover.

    Returns
    --------
    DataFrame, one column per model. Weekly turnover is the total
        weight traded each week: 4 would mean every stock was replaced.
    """

    periods_per_year = 252 / hold
    results = {}
    for model in models:
        ic = information_coefficient(preds, model)
        profit = long_short(preds, model, cost)
        gross = long_short(preds, model, 0.0)
        turnover = (gross - profit).mean() / cost
        gross_sharpe = gross.mean() / gross.std() * np.sqrt(periods_per_year)
        annual_return = profit.mean() * periods_per_year
        annual_vol = profit.std() * np.sqrt(periods_per_year)
        drawdown = (1 + profit).cumprod() / (1 + profit).cumprod().cummax() - 1
        results[model] = {
            "mean IC": ic.mean(),
            "IC t-stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
            "weekly turnover": turnover,
            "gross Sharpe": gross_sharpe,
            "net annual return": annual_return,
            "net annual vol": annual_vol,
            "net Sharpe": annual_return / annual_vol,
            "max drawdown": drawdown.min(),
        }
    return pd.DataFrame(results)
