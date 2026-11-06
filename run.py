import sys
from pathlib import Path

import matplotlib
import pandas as pd

from src import (load_prices, make_synthetic, make_features, forward_returns,
                 make_table, walk_forward, information_coefficient,
                 long_short, summary)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HOLD = 5
MODELS = ["momentum", "ridge", "xgboost"]

if "--synthetic" in sys.argv:
    close, volume = make_synthetic()
    out = Path("results_synthetic")
else:
    close, volume = load_prices()
    out = Path("results")
out.mkdir(exist_ok=True)
print("stocks:", close.shape[1], "days:", close.shape[0])

features = make_features(close, volume)
names = list(features)
fwd_ret = forward_returns(close, HOLD)

# one date per week, after a year so every feature exists
dates = [d for d in close.index[252::HOLD] if fwd_ret.loc[d].notna().any()]

data = make_table(features, fwd_ret, dates)
preds = walk_forward(data, dates, names)

table = summary(preds, MODELS, HOLD)
by_year = {}
for model in MODELS:
    ic = information_coefficient(preds, model)
    by_year[model] = ic.groupby(ic.index.year).mean()
by_year = pd.DataFrame(by_year)

print()
print(table.round(3))
print()
print("mean IC by year")
print(by_year.round(3))

table.to_csv(out / "summary.csv")
by_year.to_csv(out / "ic_by_year.csv")

# growth of 1 dollar in the long-short portfolio, after costs
fig, ax = plt.subplots(figsize=(9, 4.5))
for model in MODELS:
    profit = long_short(preds, model)
    ax.plot((1 + profit).cumprod(), label=model)
ax.axhline(1, color="black", lw=0.6)
ax.set_title("Long top 10%, short bottom 10%, after costs")
ax.set_ylabel("growth of $1")
ax.legend()
fig.tight_layout()
fig.savefig(out / "long_short.png", dpi=150)

# mean IC each year
fig, ax = plt.subplots(figsize=(9, 4.5))
by_year.plot.bar(ax=ax)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Mean information coefficient by year")
ax.set_ylabel("IC")
fig.tight_layout()
fig.savefig(out / "ic_by_year.png", dpi=150)

print()
print("saved to", out)
