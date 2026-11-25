"""Command line.

    python -m strp.cli download   # Inside Airbnb files for the Rio scrapes (into STRP_RAW)
    python -m strp.cli analyse    # tables in results/
    python -m strp.cli export F   # JSON for the interactive page
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, models

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVENTS = {"New Year's Eve": ("2026-12-31", "2026-12-31"), "Carnival 2027": ("2027-02-06", "2027-02-09")}


def cmd_download() -> None:
    for s in data.SNAPS:
        d = data.RAW / f"rio-{s}"
        d.mkdir(parents=True, exist_ok=True)
        files = ["listings.csv.gz"] + (["calendar.csv.gz", "reviews.csv.gz"] if s == data.LATEST else [])
        for f in files:
            if not (d / f).exists():
                urllib.request.urlretrieve(data.BASE.format(snap=s, file=f), d / f)


def cmd_analyse() -> dict:
    RESULTS.mkdir(exist_ok=True)
    p = models.features(data.panel())
    latest = p[p["snap"] == data.LATEST]
    seen_before = set(p.loc[p["snap"] != data.LATEST, "id"])
    h = models.hedonic(latest[latest["id"].isin(seen_before)], latest[~latest["id"].isin(seen_before)])
    rnd = np.random.default_rng(2026).random(len(latest)) < 0.8
    h_random = models.hedonic(latest[rnd], latest[~rnd])
    coef = h["ols"].params.rename("coef").reset_index().rename(columns={"index": "term"})
    coef.to_csv(RESULTS / "hedonic_coefficients.csv", index=False)
    summary = {"listings_latest": int(len(latest)), "hosts_latest": int(latest["host_id"].nunique()),
               "snapshots": data.SNAPS, "hedonic_mdape_ols": h["mdape_ols"], "hedonic_mdape_xgb": h["mdape_xgb"],
               "hedonic_r2_ols": h["r2_ols_train"], "hedonic_train": h["n_train"], "hedonic_test": h["n_test"],
               "hedonic_random_mdape_ols": h_random["mdape_ols"], "hedonic_random_mdape_xgb": h_random["mdape_xgb"],
               "smearing": float(np.mean(np.exp(h["ols"].resid)))}

    rv = data.reviews()
    stays = {s: data.stays_after(rv, s) for s in data.SNAPS[:-1]}
    d = models.elasticity_panel(p, stays)
    el = models.elasticity(d)
    el.to_csv(RESULTS / "elasticity.csv", index=False)
    summary.update({"panel_listings": int(d["id"].nunique()), "panel_rows": int(len(d)),
                    "within_sd_log_price": float((d["lp"] - d.groupby("id")["lp"].transform("mean")).std()),
                    "mean_stays_90d": float(d["stays"].mean()), "reviews": int(len(rv))})

    cal = data.forward_calendar()
    cal.to_csv(RESULTS / "forward_calendar.csv", index=False)
    base = cal[(cal["date"] >= "2026-10-01")]["share_unavailable"].median()
    ev = {}
    for k, (a, b) in EVENTS.items():
        ev[k] = float(cal[(cal["date"] >= a) & (cal["date"] <= b)]["share_unavailable"].mean())
    win = lambda a, b: float(cal[(cal["date"] >= a) & (cal["date"] <= b)]["share_unavailable"].median())
    summary.update({"calendar_listings": int(cal["listings"].iloc[0]), "calendar_median_unavailable": float(base),
                    "events_unavailable": ev,
                    "baseline_unavailable": {"1 to 20 December 2026": win("2026-12-01", "2026-12-20"),
                                             "15 February to 5 March 2027": win("2027-02-15", "2027-03-05"),
                                             "June to September 2027": win("2027-06-01", "2027-09-20")}})

    nb = latest.groupby("neighbourhood_cleansed").agg(
        listings=("id", "size"), median_price=("price", "median"),
        booked_next_90=("availability_90", lambda a: float(1 - np.median(a) / 90)),
        entire_home_share=("room_type", lambda r: float((r == "Entire home/apt").mean()))).reset_index()
    nb = nb[nb["listings"] >= 30].sort_values("listings", ascending=False)
    nb.to_csv(RESULTS / "neighbourhoods.csv", index=False)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.round(6)
    return df.astype(object).where(df.notna(), None).to_dict("records")


def cmd_export(path: str) -> None:
    cal = pd.read_csv(RESULTS / "forward_calendar.csv")
    out = {"summary": json.loads((RESULTS / "summary.json").read_text(encoding="utf-8")),
           "hedonic": _records(pd.read_csv(RESULTS / "hedonic_coefficients.csv")),
           "elasticity": _records(pd.read_csv(RESULTS / "elasticity.csv")),
           "calendar": _records(cal[["date", "share_unavailable"]]),
           "neighbourhoods": _records(pd.read_csv(RESULTS / "neighbourhoods.csv"))}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    print("wrote", path)


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "analyse"
    if cmd == "download":
        cmd_download()
    elif cmd == "analyse":
        print(json.dumps(cmd_analyse(), indent=2))
    elif cmd == "export":
        cmd_export(argv[1])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
