import numpy as np
import pandas as pd
import pytest

from src import information_coefficient, long_short


def one_week(prediction, fwd_ret, date="2020-01-03"):
    tickers = [f"S{i}" for i in range(len(fwd_ret))]
    return pd.DataFrame({"ticker": tickers,
                         "date": pd.Timestamp(date),
                         "model": prediction,
                         "fwd_ret": fwd_ret})


class TestInformationCoefficient:
    """Rank correlation between predictions and real returns."""

    def test_perfect_prediction(self):
        """Same order as the real returns gives an IC of 1."""
        fwd_ret = np.linspace(-0.05, 0.05, 10)
        preds = one_week(fwd_ret * 3, fwd_ret)
        ic = information_coefficient(preds, "model")
        assert ic.iloc[0] == pytest.approx(1)

    def test_reversed_prediction(self):
        """Opposite order gives an IC of -1."""
        fwd_ret = np.linspace(-0.05, 0.05, 10)
        preds = one_week(-fwd_ret, fwd_ret)
        ic = information_coefficient(preds, "model")
        assert ic.iloc[0] == pytest.approx(-1)


class TestLongShort:
    """Long top 10%, short bottom 10%, minus costs."""

    def test_first_week(self):
        """Long the best stock (+5%), short the worst (-5%): 10% profit,
        minus building the book (turnover 2) at 10 bps."""
        fwd_ret = np.linspace(-0.05, 0.05, 10)
        preds = one_week(fwd_ret, fwd_ret)
        profit = long_short(preds, "model", cost=0.001)
        assert profit.iloc[0] == pytest.approx(0.10 - 2 * 0.001)

    def test_no_cost_when_nothing_changes(self):
        """Same portfolio two weeks running costs nothing the second week."""
        fwd_ret = np.linspace(-0.05, 0.05, 10)
        preds = pd.concat([one_week(fwd_ret, fwd_ret, "2020-01-03"),
                           one_week(fwd_ret, fwd_ret, "2020-01-10")])
        profit = long_short(preds, "model", cost=0.001)
        assert profit.iloc[1] == pytest.approx(0.10)
