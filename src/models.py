import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def make_models():

    """ The two models, with settings fixed before looking at any results.

    Returns
    --------
    dict, model name to an unfitted model.
    """

    ridge = Ridge(alpha=10.0)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                       min_child_weight=200, subsample=0.8,
                       colsample_bytree=0.8, random_state=0)
    return {"ridge": ridge, "xgboost": xgb}


def walk_forward(data, dates, names, first_train=156, test_size=52,
                 embargo=2):

    """ Trains on the past and predicts the next year, then moves forward.

    Parameters
    -----------
    data : DataFrame, from make_table.
    dates : list, the weekly dates in order.
    names : list, the feature columns.
    first_train : int, weeks in the first training set (about 3 years).
    test_size : int, weeks predicted before retraining (about 1 year).
    embargo : int, training weeks left out just before each test period.

    Returns
    --------
    DataFrame, the test rows with a prediction column for each model,
        and a momentum column (12-1 momentum, no model) as the baseline.

    Notes
    -----
    Each target is the return from the day after a date to six days
    after it. So the target of the last week before a test period ends
    one day after that period starts, and would not be known yet when
    the first test prediction is made. Leaving out the last `embargo`
    weeks avoids this. One week is the minimum needed here, and two
    gives a margin.
    """

    predictions = []
    for start in range(first_train, len(dates), test_size):
        train_dates = dates[: start - embargo]
        test_dates = dates[start: start + test_size]
        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)].copy()

        for name, model in make_models().items():
            model.fit(train[names], train["target"])
            test[name] = model.predict(test[names])
        test["momentum"] = test["mom_12_1"]
        predictions.append(test)
        print("train up to", train_dates[-1].date(),
              "test", test_dates[0].date(), "to", test_dates[-1].date())

    return pd.concat(predictions)
