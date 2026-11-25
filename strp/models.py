"""Hedonic price model and the price elasticity of stays.

Hedonic: the log nightly price on location, size, quality and booking rules. A log-linear model
gives prices a person can read (and that the page computes in the browser); XGBoost on the same
features shows how much a flexible model adds. Both are scored on listings that first appear in
the latest scrape, which neither model saw.

Elasticity: for listings seen in several scrapes, log(1 + stays in the next 90 days) on log price,
with a fixed effect for each listing and for each neighbourhood in each scrape. The listing
effect removes lasting quality; the neighbourhood-scrape effect removes the season and local
demand. What is left is a host moving the price of the same place, relative to the area. Hosts
may move prices when they expect demand to change, so price is also instrumented by the mean log
price of the same host's other listings in that scrape: a pricing decision made for the whole
portfolio, not for this place's demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb
from linearmodels.iv import IV2SLS

NUM = ["log_acc", "bedrooms", "bathrooms", "rating", "log_reviews", "n_amenities", "superhost", "instant", "log_min_nights"]


def features(l: pd.DataFrame) -> pd.DataFrame:
    f = l.copy()
    f["log_acc"] = np.log(f["accommodates"].fillna(2).clip(1, 16))
    f["bedrooms"] = f["bedrooms"].fillna(1).clip(0, 6)
    f["bathrooms"] = f["bathrooms"].fillna(1).clip(0, 5)
    f["rating"] = f["review_scores_rating"].fillna(f["review_scores_rating"].median()).clip(1, 5)
    f["log_reviews"] = np.log1p(f["number_of_reviews"].fillna(0))
    f["log_min_nights"] = np.log(f["minimum_nights"].fillna(1).clip(1, 365))
    f["n_amenities"] = f["n_amenities"].fillna(0).clip(0, 120)
    f["lp"] = np.log(f["price"])
    return f


def hedonic(train: pd.DataFrame, test: pd.DataFrame, min_listings: int = 30) -> dict:
    keep = train["neighbourhood_cleansed"].value_counts()
    keep = set(keep[keep >= min_listings].index)
    tr = train.assign(nb=np.where(train["neighbourhood_cleansed"].isin(keep), train["neighbourhood_cleansed"], "Other"))
    te = test.assign(nb=np.where(test["neighbourhood_cleansed"].isin(keep), test["neighbourhood_cleansed"], "Other"))
    ols = smf.ols("lp ~ C(nb) + C(room_type) + " + " + ".join(NUM), data=tr).fit()
    X = lambda d: pd.get_dummies(d[NUM + ["nb", "room_type"]], columns=["nb", "room_type"], dtype=float)
    Xtr = X(tr)
    Xte = X(te).reindex(columns=Xtr.columns, fill_value=0.0)
    gb = xgb.XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          random_state=2026, n_jobs=8)
    gb.fit(Xtr, tr["lp"])
    p_ols, p_gb = np.exp(ols.predict(te)), np.exp(gb.predict(Xte))
    mape = lambda p: float(np.nanmedian(np.abs(p - te["price"]) / te["price"]))
    return {"ols": ols, "neighbourhoods": sorted(keep), "mdape_ols": mape(p_ols), "mdape_xgb": mape(p_gb),
            "r2_ols_train": float(ols.rsquared), "n_train": int(len(tr)), "n_test": int(len(te))}


def _within(d: pd.DataFrame, cols: list[str], g1: str, g2: str) -> pd.DataFrame:
    x = d[cols].astype(float).copy()
    for _ in range(30):
        x = x - x.groupby(d[g1]).transform("mean")
        x = x - x.groupby(d[g2]).transform("mean")
    return x


def elasticity_panel(p: pd.DataFrame, stays: dict[str, pd.Series]) -> pd.DataFrame:
    d = p[p["snap"].isin(list(stays))].copy()
    d["stays"] = [stays[s].get(i, 0) for s, i in zip(d["snap"], d["id"])]
    d["y"] = np.log1p(d["stays"])
    d["lp"] = np.log(d["price"])
    d = d[d.groupby("id")["snap"].transform("nunique") >= 2]
    d["ns"] = d["neighbourhood_cleansed"] + "_" + d["snap"]
    g = d.groupby(["host_id", "snap"])["lp"]
    n = g.transform("count")
    d["host_other_lp"] = ((g.transform("sum") - d["lp"]) / (n - 1)).where(n > 1)
    return d


def elasticity(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cl_id = d["id"].astype("category").cat.codes
    z = _within(d, ["y", "lp", "stays"], "id", "ns")
    m = IV2SLS(z["y"], z[["lp"]], None, None).fit(cov_type="clustered", clusters=cl_id)
    rows.append({"spec": "fixed effects", "coef": m.params["lp"], "se": m.std_errors["lp"], "n": int(m.nobs),
                 "listings": int(d["id"].nunique()), "first_stage_f": np.nan})
    m2 = IV2SLS(z["stays"], z[["lp"]], None, None).fit(cov_type="clustered", clusters=cl_id)
    rows.append({"spec": "fixed effects, stays in levels (elasticity at the mean)", "coef": m2.params["lp"] / d["stays"].mean(),
                 "se": m2.std_errors["lp"] / d["stays"].mean(), "n": int(m2.nobs), "listings": int(d["id"].nunique()),
                 "first_stage_f": np.nan})
    e = d.dropna(subset=["host_other_lp"])
    ze = _within(e, ["y", "lp", "host_other_lp"], "id", "ns")
    cl_h = e["host_id"].astype("category").cat.codes
    o = IV2SLS(ze["y"], ze[["lp"]], None, None).fit(cov_type="clustered", clusters=cl_h)
    rows.append({"spec": "fixed effects, multi-listing hosts", "coef": o.params["lp"], "se": o.std_errors["lp"],
                 "n": int(o.nobs), "listings": int(e["id"].nunique()), "first_stage_f": np.nan})
    iv = IV2SLS(ze["y"], None, ze[["lp"]], ze[["host_other_lp"]]).fit(cov_type="clustered", clusters=cl_h)
    rows.append({"spec": "instrumented by the host's other listings", "coef": iv.params["lp"], "se": iv.std_errors["lp"],
                 "n": int(iv.nobs), "listings": int(e["id"].nunique()),
                 "first_stage_f": float(iv.first_stage.diagnostics.loc["lp", "f.stat"])})
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["coef"] - 1.96 * out["se"], out["coef"] + 1.96 * out["se"]
    return out
