# QQQ timepoint-21 direction predictor (anonymized, TypeSafe SDK)

## Install

```sh
pip install -r requirements.txt   # real SDK is `typesafe-sdk`
                                  # (see https://docs.typesafe.ai/sdk/python#pip)
export TYPESAFE_API_KEY="..."
```

## Run

```sh
python predict_qqq_direction.py --print-prompt                # live: fetch QQQ (N=20), predict UP/DOWN
python predict_qqq_direction.py --timepoints 50 --print-prompt # live: 50 observations, predict key 51
python predict_qqq_direction.py --no-live --print-prompt       # offline: show anonymized prompt only
python test_predict_qqq_direction.py                          # offline unit tests (no key/network)
```

`--timepoints N` sets the observation-window size (default 20, allowed
20–200). The state always holds N observed `change%` values plus a `null`
placeholder at key N+1, which is the to-predict target.

## Prediction log (`predictions.jsonl`)

Every successful live run appends one JSON object per line:

```json
{"prediction_date": "2026-09-30", "direction": "UP", "confidence": 0.82, "actual_change_pct": null, "timepoints": 20, "model": "system-one-x", "run_at": "2026-09-29T09:30:00Z"}
```

- `prediction_date`: next weekday after the last observed trading date
  (exchange holidays not calendar-checked).
- `actual_change_pct`: always `null` (write-once log; realized move is
  unknowable on prediction day and is never backfilled).
- `--output PATH` overrides the log path; `--no-write` suppresses writing;
  `--no-live` never writes.

## Backtest (`backtest_qqq_direction.py`)

Walks every trading day in `--start..--end` (default 2020-01-01..2026-08-30),
predicts each day from the N prior log changes (excluding the target day) via
the client-default model — 20 predictions in parallel (`--workers`, default
20) — and writes a Markdown-only accuracy report (overall accuracy,
confusion matrix, yearly + confidence splits) to `backtest_report.md`.
Predictions cache in `backtest_predictions.jsonl` for skip-and-resume;
API errors are recorded per day and skipped. `--limit K` caps a run for
smoke tests.

## What it does

1. Fetches latest 21 daily QQQ closes via `yfinance` (last N trading rows, so
   weekends/holidays are skipped) and computes 20 logged percentage changes
   `ln(p_t / p_{t-1}) * 100`.
2. Builds an anonymized TypeSafe `state` as one compact JSON **object map**
   `{"1": v1, ..., "20": v20, "21": null}` — string keys `"1"`–`"20"`
   hold the observed change% values, key `"21"` is `null` (to predict).
   Shape is narrated by `STATE_DESCRIPTION` (sent inside the question
   instructions). No ticker, asset class,
   price, date, or calendar tokens (enforced by `assert_anonymized`).
3. Calls `TypeSafeClient.system_one` with a binary `Choice(up/down)` question
   asking for the direction of `timepoint 21`, and prints `UP` or `DOWN` with
   confidence/probabilities.
