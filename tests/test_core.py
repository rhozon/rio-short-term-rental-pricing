import numpy as np
import pandas as pd
import pytest

from strp import data, models


def test_parse_price():
    s = pd.Series(["$1,234.50", "$80.00", None, "abc"])
    out = data.parse_price(s)
    assert out.iloc[0] == pytest.approx(1234.5) and out.iloc[1] == 80
    assert out.iloc[2:].isna().all()


def test_stays_after_counts_the_window_only():
    rv = pd.DataFrame({"listing_id": [1, 1, 1, 2], "date": pd.to_datetime(["2026-03-30", "2026-04-15", "2026-07-01", "2026-05-01"])})
    s = data.stays_after(rv, "2026-03-30", 90)
    assert s.get(1) == 1 and s.get(2) == 1   # the review on the scrape day and the one after 90 days are out


def _panel(beta=-0.3, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        quality = rng.normal()
        host = i // 3                     # hosts with three listings each
        for s in ("a", "b", "c"):
            host_shock = rng.normal(0, 0.2)
            lp = 6 + 0.5 * quality + host_shock + rng.normal(0, 0.05)
            rows.append({"id": i, "host_id": host, "snap": s, "neighbourhood_cleansed": "X",
                         "price": np.exp(lp), "_q": quality, "_hs": host_shock})
    d = pd.DataFrame(rows)
    # host shocks shared across the host's listings in a scrape
    d["_hs"] = d.groupby(["host_id", "snap"])["_hs"].transform("first")
    d["price"] = np.exp(6 + 0.5 * d["_q"] + d["_hs"] + rng.normal(0, 0.05, len(d)))
    lam = np.exp(1 + 0.8 * d["_q"] + beta * (np.log(d["price"]) - 6 - 0.5 * d["_q"]))
    stays = {s: pd.Series(rng.poisson(lam[d["snap"] == s].to_numpy()), index=d.loc[d["snap"] == s, "id"].to_numpy())
             for s in ("a", "b", "c")}
    return d, stays


def test_host_instrument_leaves_the_listing_out():
    d, stays = _panel()
    e = models.elasticity_panel(d, stays)
    r = e[(e["host_id"] == 0) & (e["snap"] == "a")]
    others = r["lp"].sum() - r["lp"].iloc[0]
    assert r["host_other_lp"].iloc[0] == pytest.approx(others / 2)


def test_fixed_effects_remove_quality_and_recover_a_negative_response():
    d, stays = _panel(beta=-0.6)
    e = models.elasticity_panel(d, stays)
    out = models.elasticity(e).set_index("spec")
    # quality raises both price and stays; without listing effects the slope would be positive
    assert out.loc["fixed effects", "coef"] < -0.2
    assert out.loc["instrumented by the host's other listings", "coef"] < -0.2


def test_features_have_no_missing_values():
    l = pd.DataFrame({"price": [100.0, 200.0], "accommodates": [np.nan, 4], "bedrooms": [np.nan, 2], "bathrooms": [1, np.nan],
                      "review_scores_rating": [np.nan, 4.8], "number_of_reviews": [0, 10], "minimum_nights": [np.nan, 2],
                      "n_amenities": [np.nan, 30], "superhost": [0, 1], "instant": [1, 0]})
    f = models.features(l)
    assert not f[models.NUM].isna().any().any()
