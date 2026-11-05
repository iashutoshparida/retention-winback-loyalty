# Repeat-Buyer Retention, Dormant Win-Back and Loyalty Tiers

A retention study on one year of real transactions from a UK online gift wholesaler
(UCI Online Retail, 1 Dec 2010 to 9 Dec 2011). It answers four questions a growth team asks:
how fast do new buyers drop off, how big is the dormant base, which dormant buyers are worth
a win-back offer, and what a spend-based loyalty program would cost.

## Data

`online-retail-dataset.csv` from
[databricks/Spark-The-Definitive-Guide](https://github.com/databricks/Spark-The-Definitive-Guide/tree/master/data/retail-data/all).
541,909 raw lines; 397,884 after dropping rows with no customer ID, cancellations and
zero or negative quantities and prices. 4,338 buyers and 18,532 orders remain.

```
mkdir -p data && curl -L -o data/online-retail-dataset.csv \
  https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv
pip install -r requirements.txt
python3 src/retention_analysis.py data/online-retail-dataset.csv
```

Outputs land in `results/`.

## Findings

| Question | Result |
|---|---|
| How concentrated is value? | 34.4% of buyers ordered once; repeat buyers made 93.1% of revenue |
| How fast do new buyers drop off? | Jan to Aug 2011 cohorts kept 19.0% of buyers in month 1 and 22.8% in month 3 |
| How big is the dormant base? | On 1 Sep 2011, 1,394 of 3,317 buyers (42.0%) had not ordered in 90+ days; they made 16.2% of past revenue |
| Do dormant buyers come back on their own? | 42.4% ordered again within the next 100 days with no campaign |
| Can we tell who will come back? | Gradient boosting on RFM, overdue ratio and a Christmas-2010 flag: 5-fold CV AUC 0.634 vs 0.583 for "most recent first". Top 20% returned at 63.1%, bottom 20% at 30.8% |
| How big a holdout? | Detecting a 10-point lift on a 42% base rate at 80% power needs 391 buyers per arm, so a 30% holdout (418 buyers) works. A 5-point lift would need 1,553 per arm |
| What would loyalty tiers cost? | Top 5% of buyers made 50.3% of revenue, top 20% made 74.6%. Rewards of 4% / 3% / 2% for the top 5% / next 15% / next 30% cost 3.1% of revenue and break even if those tiers spend 11.2% more, at an assumed 30% gross margin |

## What I would do with this

1. Spend win-back incentives on the middle propensity band. The top band mostly returns anyway
   (63%), so a discount there pays for orders that would have happened.
2. Always keep a holdout. With 42% organic return, a campaign without one would claim roughly
   twice its real effect.
3. Launch tiers only for the top 50% by spend, and judge the program on incremental spend per
   tier against the 11% break-even, not on redemption counts.

## Limits

One year of data, so seasonality is seen once. The propensity model is modest (AUC 0.63) and
ranks buyers; it does not measure uplift, which needs a real treated vs holdout test. The 30%
margin is an assumption, not a figure from the data.
