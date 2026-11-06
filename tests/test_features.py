import numpy as np
import pandas as pd

from src import make_synthetic, make_features, forward_returns, make_table


def small_market():
    close, volume = make_synthetic(n_stocks=20, n_days=400, seed=1)
    return close, volume


class TestMakeFeatures:
    """Features only use the past."""

    def test_future_prices_do_not_change_features(self):
        """Changing prices after a day leaves that day's features alone."""
        close, volume = small_market()
        day = close.index[300]
        before = make_features(close, volume)

        changed = close.copy()
        changed.loc[changed.index > day] *= 2
        after = make_features(changed, volume)

        for name in before:
            assert before[name].loc[day].equals(after[name].loc[day])

    def test_momentum_value(self):
        """12-1 momentum is the log return from 252 to 21 days ago."""
        close, volume = small_market()
        mom = make_features(close, volume)["mom_12_1"]
        t = 300
        expected = np.log(close.iloc[t - 21] / close.iloc[t - 252])
        assert np.allclose(mom.iloc[t], expected)


class TestForwardReturns:
    """Next week's return, starting tomorrow."""

    def test_simple_example(self):
        """Prices 1 to 8: from day 1 (price 2) to day 6 (price 7)."""
        close = pd.DataFrame({"A": [1.0, 2, 3, 4, 5, 6, 7, 8]})
        result = forward_returns(close, hold=5)
        assert result["A"].iloc[0] == 7 / 2 - 1
        assert np.isnan(result["A"].iloc[-1])


class TestMakeTable:
    """The X / y table."""

    def test_ranks_are_between_minus_and_plus_half(self):
        """Ranked features and target stay within -0.5 and 0.5."""
        close, volume = small_market()
        features = make_features(close, volume)
        fwd_ret = forward_returns(close)
        dates = list(close.index[260:380:5])
        data = make_table(features, fwd_ret, dates)

        columns = list(features) + ["target"]
        assert data[columns].min().min() >= -0.5
        assert data[columns].max().max() <= 0.5
        assert data[columns].isna().sum().sum() == 0
