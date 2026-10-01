# research

LLM market-prediction experiments plus a library of SEC-sourced company profiles.

## Contents

| Path | What it is |
| --- | --- |
| `backtest_simple.ipynb` | QQQ direction backtest (N=50, sequential, in-memory) |
| `sp500.csv` | S&P 500 constituents: ticker, name, GICS sector/sub-industry, HQ, date added, CIK, founded |
| `descriptions/` | 502 `descriptions/<SYMBOL>.json` company profiles extracted from Form 10-K Item 1 |
| `.pi/skills/extracting-company-profiles/` | Skill for producing those profiles |

## Install

```sh
pip install -r requirements.txt   # typesafe-sdk, yfinance, pandas, numpy
export TYPESAFE_API_KEY="..."     # needed for the backtest and for profile validation
```

## QQQ direction backtest

`backtest_simple.ipynb` is self-contained — no threads, no cache files, no CLI.
Everything stays in memory. Run it top-to-bottom: `Kernel → Restart & Run All`.

It downloads QQQ daily closes with `yfinance` starting `2025-10-01` as a warmup
buffer, then loops every trading day from `2026-01-02` to `2026-08-30`. For each
target day it takes the previous 50 trading days (current day excluded), computes
50 simple percent changes, and asks the model for the direction of the next day.

The state sent to the model is an anonymized object map
`{"1": v1, ..., "50": v50, "51": null}` — keys `"1"`–`"50"` hold the observed
change values, key `"51"` is `null` and is the target. No ticker, asset class,
price, date, or calendar tokens are included (`assert_anonymized` enforces this).
The question is a binary `Choice(up/down)`; the model returns `UP` or `DOWN` with
a confidence. Days where the API call fails are printed and skipped.

The final cell reports accuracy, a confusion matrix (UP/UP, UP/DOWN, DOWN/UP,
DOWN/DOWN), and a chart of predicted-direction runs over time.

## Company profiles

Each file in `descriptions/` holds one profile drafted from Item 1 (Business) of a
company's latest Form 10-K on SEC EDGAR, then gated by a TypeSafe `noul` check
(`answers.check_profile.noul >= 0.5`) before being written:

```json
{
  "symbol": "AAPL",
  "description": "Apple Inc. designs, manufactures and markets ...",
  "company_name": "Apple Inc.",
  "form": "10-K",
  "filing_url": "https://www.sec.gov/Archives/edgar/data/320193/...",
  "source": "SEC EDGAR",
  "source_section": "Item 1. Business"
}
```

To add or refresh a profile, follow the skill in
`.pi/skills/extracting-company-profiles/SKILL.md` — read the filing directly (no
scraper script), validate with the `check_profile` question, then write
`descriptions/<SYMBOL>.json`.
