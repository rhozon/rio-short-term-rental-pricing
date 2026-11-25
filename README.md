# Pricing short-term rentals in Rio

What a night is worth in each part of Rio de Janeiro, how much stays fall when a host raises the
price, and how full New Year's Eve and Carnival already are, from four quarterly Inside Airbnb
scrapes (November 2025 to September 2026, CC BY 4.0).

Case study page, with a pricing simulator:
<https://rhozon.github.io/portfolio/rio-short-term-rental-pricing.html>

## What is in it

| Part | Module | What it answers |
|---|---|---|
| Data | `strp/data.py` | Scrapes, dated reviews as the proxy for stays, forward availability by date |
| Hedonic prices | `strp/models.py` | Log-linear and XGBoost price models, scored on a random holdout and on new listings |
| Price response | `strp/models.py` | Stays in the next 90 days on log price, listing and neighbourhood-by-scrape fixed effects, instrumented by the host's other listings |
| Tables | `strp/cli.py` | Download, analysis, export for the page |

## Results (from `results/`)

| | |
|---|---|
| Hedonic median error, random holdout: log-linear / XGBoost | 31% / 28% |
| Same, listings new in the latest scrape | 55% / 47% |
| Log stays on log price, fixed effects | -0.17 (95% CI -0.18 to -0.16) |
| Instrumented by the host's other listings | -0.17 (-0.24 to -0.11), first-stage F 392 |
| Nights unavailable on 22 Sep 2026: New Year's Eve / 1 to 20 Dec | 67% / 23% |
| Carnival 2027 / 15 Feb to 5 Mar 2027 | 61% / 36% |

The calendar files no longer carry prices, and the 28 December 2025 scrape has none, so both are
outside the analysis.

## Run it

```bash
pip install -e ".[dev]"
pytest -v
python -m strp.cli download
python -m strp.cli analyse
python -m strp.cli export site/data.json
```

## License

MIT. Rodrigo Hermont Ozon. Data: Inside Airbnb, CC BY 4.0.
