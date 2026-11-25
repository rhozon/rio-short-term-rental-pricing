"""Inside Airbnb snapshots of Rio de Janeiro (CC BY 4.0).

Each quarterly scrape lists every active listing with its characteristics and one nightly price,
quoted for a stay a few days after the scrape. The calendar of the latest scrape says, for each
listing and each of the next 365 days, whether the night is still available; it no longer carries
prices. Reviews carry their date, and a review follows a stay, so reviews in the 90 days after a
scrape are the proxy for stays.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(os.environ.get("STRP_RAW", "C:/Users/rodri/data/insideairbnb"))
BASE = "https://data.insideairbnb.com/brazil/rj/rio-de-janeiro/{snap}/data/{file}"
# Quarterly scrapes with prices (the 28 December 2025 scrape has none).
SNAPS = ["2025-11-27", "2026-03-30", "2026-06-24", "2026-09-22"]
LATEST = SNAPS[-1]
COLS = ["id", "host_id", "price", "room_type", "property_type", "accommodates", "bedrooms", "beds", "bathrooms",
        "neighbourhood_cleansed", "number_of_reviews", "review_scores_rating", "availability_90",
        "availability_365", "minimum_nights", "host_listings_count", "host_is_superhost", "amenities",
        "latitude", "longitude", "instant_bookable"]


def parse_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def listings(snap: str, raw: Path = RAW) -> pd.DataFrame:
    l = pd.read_csv(raw / f"rio-{snap}" / "listings.csv.gz", usecols=lambda c: c in COLS)
    l["snap"] = snap
    l["price"] = parse_price(l["price"])
    l["n_amenities"] = l["amenities"].fillna("[]").str.count('",') + 1
    l["superhost"] = (l["host_is_superhost"] == "t").astype(int)
    l["instant"] = (l["instant_bookable"] == "t").astype(int)
    return l.drop(columns=["amenities", "host_is_superhost", "instant_bookable"])


def panel(raw: Path = RAW, lo: float = 30, hi: float = 20000) -> pd.DataFrame:
    p = pd.concat([listings(s, raw) for s in SNAPS])
    return p[(p["price"] >= lo) & (p["price"] <= hi)].copy()


def reviews(raw: Path = RAW) -> pd.DataFrame:
    r = pd.read_csv(raw / f"rio-{LATEST}" / "reviews.csv.gz", usecols=["listing_id", "date"])
    r["date"] = pd.to_datetime(r["date"])
    return r


def stays_after(rv: pd.DataFrame, snap: str, days: int = 90) -> pd.Series:
    a = pd.Timestamp(snap)
    w = rv[(rv["date"] > a) & (rv["date"] <= a + pd.Timedelta(days=days))]
    return w.groupby("listing_id").size()


def forward_calendar(raw: Path = RAW) -> pd.DataFrame:
    """Share of active listings whose night is no longer available, by date (latest scrape)."""
    c = pd.read_csv(raw / f"rio-{LATEST}" / "calendar.csv.gz", usecols=["listing_id", "date", "available"])
    active = c.groupby("listing_id")["available"].apply(lambda a: (a == "t").any())
    c = c[c["listing_id"].isin(active[active].index)]
    out = c.groupby("date")["available"].apply(lambda a: (a == "f").mean()).rename("share_unavailable").reset_index()
    out["date"] = pd.to_datetime(out["date"])
    out["listings"] = int(active.sum())
    return out
