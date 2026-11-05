"""Repeat-buyer retention, dormant-base win-back and loyalty-tier sizing.

Data: UCI Online Retail (UK online gift wholesaler, 1 Dec 2010 - 9 Dec 2011), as mirrored in
github.com/databricks/Spark-The-Definitive-Guide (data/retail-data/all/online-retail-dataset.csv).

Run:  python3 src/retention_analysis.py data/online-retail-dataset.csv
Writes results/summary.json, results/cohort_retention.csv, results/dormant_scores.csv,
results/loyalty_tiers.csv
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
RES = Path("results")
RES.mkdir(exist_ok=True)
out = {}

# ---------------------------------------------------------------- 1. clean
df = pd.read_csv(sys.argv[1], dtype={"InvoiceNo": str})
out["raw_lines"] = len(df)
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].str.startswith("C")]                 # cancellations
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
df["rev"] = df["Quantity"] * df["UnitPrice"]
out["clean_lines"] = len(df)

orders = (df.groupby(["InvoiceNo", "CustomerID"], as_index=False)
            .agg(date=("InvoiceDate", "min"), rev=("rev", "sum"), skus=("StockCode", "nunique")))
orders["day"] = orders["date"].dt.normalize()
out["buyers"] = int(orders["CustomerID"].nunique())
out["orders"] = len(orders)

cust = orders.groupby("CustomerID").agg(n=("InvoiceNo", "count"), rev=("rev", "sum"))
out["one_time_buyer_share"] = round((cust["n"] == 1).mean(), 3)
out["repeat_buyer_rev_share"] = round(cust.loc[cust["n"] > 1, "rev"].sum() / cust["rev"].sum(), 3)

# ---------------------------------------------------------------- 2. monthly cohorts
orders["om"] = orders["date"].dt.to_period("M")
orders["cohort"] = orders.groupby("CustomerID")["om"].transform("min")
orders["k"] = (orders["om"] - orders["cohort"]).apply(lambda p: p.n)
size = orders.groupby("cohort")["CustomerID"].nunique()
ret = orders.groupby(["cohort", "k"])["CustomerID"].nunique().unstack(fill_value=0).div(size, axis=0)
ret.round(3).to_csv(RES / "cohort_retention.csv")
# Dec-2010 holds the backlog of pre-existing buyers, so new-buyer retention uses Jan-Aug 2011 cohorts
new = ret.loc[pd.Period("2011-01", "M"):pd.Period("2011-08", "M")]
out["new_cohort_m1_retention"] = round(new[1].mean(), 3)
out["new_cohort_m3_retention"] = round(new[3].mean(), 3)

# ---------------------------------------------------------------- 3. dormant base at 1 Sep 2011
CUT = pd.Timestamp("2011-09-01")
END = orders["day"].max()
H = (END - CUT).days + 1                                        # 100-day outcome window
cal, fut = orders[orders["day"] < CUT], orders[orders["day"] >= CUT]
g = cal.groupby("CustomerID")
f = pd.DataFrame({
    "recency": (CUT - g["day"].max()).dt.days,
    "tenure": (CUT - g["day"].min()).dt.days,
    "orders": g["InvoiceNo"].count(),
    "spend": g["rev"].sum(),
    "aov": g["rev"].mean(),
    "skus_per_order": g["skus"].mean(),
})
f["median_gap"] = cal.sort_values("day").groupby("CustomerID")["day"].apply(
    lambda s: s.diff().dt.days.median())
f["overdue_ratio"] = f["recency"] / f["median_gap"].fillna(f["tenure"]).clip(lower=1)
f["bought_xmas_2010"] = f.index.isin(cal.loc[cal["day"] < "2011-01-01", "CustomerID"]).astype(int)
f["uk"] = f.index.isin(df.loc[df["Country"] == "United Kingdom", "CustomerID"]).astype(int)
f["returned"] = f.index.isin(fut["CustomerID"]).astype(int)
f["future_spend"] = fut.groupby("CustomerID")["rev"].sum().reindex(f.index).fillna(0)

DORMANT_DAYS = 90
d = f[f["recency"] > DORMANT_DAYS].copy()
out["outcome_window_days"] = H
out["base_at_cut"] = len(f)
out["dormant_n"] = len(d)
out["dormant_share"] = round(len(d) / len(f), 3)
out["dormant_past_rev_share"] = round(d["spend"].sum() / f["spend"].sum(), 3)
out["dormant_organic_return_rate"] = round(d["returned"].mean(), 3)

# ---------------------------------------------------------------- 4. return-propensity score (5-fold CV)
feats = ["recency", "tenure", "orders", "spend", "aov", "skus_per_order",
         "overdue_ratio", "bought_xmas_2010", "uk"]
gbm = GradientBoostingClassifier(random_state=7, max_depth=2, n_estimators=150, learning_rate=0.05)
cv = StratifiedKFold(5, shuffle=True, random_state=7)
d["p_return"] = cross_val_predict(gbm, d[feats].fillna(-1), d["returned"], cv=cv,
                                  method="predict_proba")[:, 1]
out["propensity_cv_auc"] = round(roc_auc_score(d["returned"], d["p_return"]), 3)
out["recency_rule_auc"] = round(roc_auc_score(d["returned"], -d["recency"]), 3)
d["band"] = pd.qcut(d["p_return"].rank(method="first"), [0, .2, .8, 1], labels=["low", "mid", "high"])
bands = d.groupby("band", observed=True).agg(n=("returned", "size"), return_rate=("returned", "mean"),
                                             avg_past_spend=("spend", "mean"))
out["bands"] = bands.round(3).reset_index().astype({"band": str}).to_dict(orient="records")
d.sort_values("p_return", ascending=False).to_csv(RES / "dormant_scores.csv")

# ---------------------------------------------------------------- 5. holdout sizing for a win-back test
def n_per_arm(p0, lift, alpha=0.05, power=0.8):
    p1, pbar = p0 + lift, p0 + lift / 2
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return int(np.ceil((za * np.sqrt(2 * pbar * (1 - pbar)) +
                        zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / lift ** 2))


p0 = out["dormant_organic_return_rate"]
out["n_per_arm_5pp"] = n_per_arm(p0, 0.05)
out["n_per_arm_10pp"] = n_per_arm(p0, 0.10)
out["holdout_30pct_n"] = int(round(0.3 * len(d)))

# ---------------------------------------------------------------- 6. loyalty tiers on 12-month spend
c = cust.sort_values("rev", ascending=False).copy()
c["pct"] = np.arange(1, len(c) + 1) / len(c)
c["tier"] = np.select([c["pct"] <= .05, c["pct"] <= .20, c["pct"] <= .50],
                      ["Platinum", "Gold", "Silver"], "Base")
REWARD = {"Platinum": .04, "Gold": .03, "Silver": .02, "Base": 0.0}
MARGIN = 0.30                                                   # assumption for break-even maths
t = c.groupby("tier").agg(buyers=("rev", "size"), revenue=("rev", "sum"), median_orders=("n", "median"))
t["rev_share"] = t["revenue"] / t["revenue"].sum()
t["reward_rate"] = t.index.map(REWARD)
t["reward_cost"] = t["revenue"] * t["reward_rate"]
t.round(3).to_csv(RES / "loyalty_tiers.csv")
paid = t["reward_rate"] > 0
out["top5_rev_share"] = round(t.loc["Platinum", "rev_share"], 3)
out["top20_rev_share"] = round(t.loc[["Platinum", "Gold"], "rev_share"].sum(), 3)
out["reward_cost_share_of_rev"] = round(t["reward_cost"].sum() / t["revenue"].sum(), 3)
out["breakeven_spend_lift_paid_tiers"] = round(t["reward_cost"].sum() / MARGIN / t.loc[paid, "revenue"].sum(), 3)

(RES / "summary.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
