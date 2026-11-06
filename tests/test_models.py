from src import (make_synthetic, make_features, forward_returns, make_table,
                 walk_forward, information_coefficient)


def pipeline(signal):
    close, volume = make_synthetic(n_stocks=50, n_days=1600, signal=signal)
    features = make_features(close, volume)
    fwd_ret = forward_returns(close)
    dates = list(close.index[252::5])[:-2]
    data = make_table(features, fwd_ret, dates)
    return walk_forward(data, dates, list(features), first_train=104)


class TestWalkForward:
    """Training never sees the test period."""

    def test_no_test_dates_in_training(self):
        """Every predicted week comes after the first training block."""
        close, volume = make_synthetic(n_stocks=20, n_days=1200)
        features = make_features(close, volume)
        fwd_ret = forward_returns(close)
        dates = list(close.index[252::5])[:-2]
        data = make_table(features, fwd_ret, dates)
        preds = walk_forward(data, dates, list(features), first_train=104)
        assert preds["date"].min() == dates[104]

    def test_finds_planted_signal(self):
        """With a hidden signal, the models find a positive IC."""
        preds = pipeline(signal=0.0015)
        assert information_coefficient(preds, "ridge").mean() > 0.03

    def test_no_signal_no_skill(self):
        """With no signal, the IC is close to 0, so nothing leaks."""
        preds = pipeline(signal=0.0)
        assert abs(information_coefficient(preds, "ridge").mean()) < 0.02
