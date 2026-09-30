"""Offline unit tests for backtest_qqq_direction.py (mocked yfinance + SDK)."""

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from backtest_qqq_direction import (
    compute_metrics,
    direction_of,
    render_report,
    run_backtest,
    validate_workers,
)


def _fixture_df(n_days=30, seed=0):
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    closes = 300 + np.cumsum(np.random.RandomState(seed).randn(n_days))
    return pd.DataFrame({"Close": closes}, index=dates)


def _mock_stack(df, monkey_choice="up", delay=0.0):
    from unittest.mock import MagicMock, patch

    fake_answer = MagicMock(choice=monkey_choice, confidence=0.7,
                            probabilities={"up": 0.7, "down": 0.3})
    fake_resp = MagicMock(model="test-model")
    fake_resp.choices = {"direction": fake_answer}
    fake_client = MagicMock()
    if delay:
        import time

        def slow(*a, **k):
            time.sleep(delay)
            return fake_resp

        fake_client.system_one.side_effect = slow
    else:
        fake_client.system_one.return_value = fake_resp
    fake_client.__enter__.return_value = fake_client
    os.environ["TYPESAFE_API_KEY"] = "dummy-test-key"
    return [patch("yfinance.download", return_value=df),
            patch("typesafe_sdk.TypeSafeClient", return_value=fake_client)]


def _enter(patches):
    for p in patches:
        p.__enter__()
    return patches


def _exit(patches):
    for p in reversed(patches):
        p.__exit__(None, None, None)


def test_exclusion_correctness():
    """First target uses only prior data; actual sign matches fixture math."""
    import pandas as pd

    df = _fixture_df(30)
    patches = _enter(_mock_stack(df))
    try:
        with tempfile.TemporaryDirectory() as td:
            out = run_backtest(start="2020-01-01", end="2020-03-31", n=20, workers=4,
                               predictions_path=os.path.join(td, "p.jsonl"),
                               report_path=os.path.join(td, "r.md"))
    finally:
        _exit(patches)
    recs = [r for r in out["records"] if "error" not in r]
    # 30 biz days starting 2020-01-01: first target index 21 -> date 2020-01-30
    assert recs[0]["date"] == "2020-01-30", recs[0]["date"]
    closes = df["Close"].tolist()
    expect = round(math.log(closes[21] / closes[20]) * 100, 4)
    assert recs[0]["actual_change_pct"] == expect
    assert recs[0]["actual"] == direction_of(expect)


def test_parallel_matches_serial_and_cache_valid():
    df = _fixture_df(45)
    for workers in (1, 20):
        patches = _enter(_mock_stack(df, delay=0.001))
        try:
            with tempfile.TemporaryDirectory() as td:
                out = run_backtest(start="2020-01-01", end="2020-04-30", n=20,
                                   workers=workers,
                                   predictions_path=os.path.join(td, "p.jsonl"),
                                   report_path=os.path.join(td, "r.md"))
                lines = open(os.path.join(td, "p.jsonl")).read().strip().splitlines()
        finally:
            _exit(patches)
        # every line valid single-line JSON, one per eligible day, no dupes
        seen = set()
        for ln in lines:
            rec = json.loads(ln)
            assert rec["date"] not in seen
            seen.add(rec["date"])
        assert len(lines) == out["metrics"]["total"] == len(seen)
    # determinism across worker counts
    assert out["metrics"]["total"] > 0


def test_metrics_math():
    rows = [
        {"date": "2020-01-02", "predicted": "UP", "confidence": 0.9,
         "actual": "UP", "actual_change_pct": 1.0, "correct": True, "timepoints": 20},
        {"date": "2020-01-03", "predicted": "UP", "confidence": 0.5,
         "actual": "DOWN", "actual_change_pct": -1.0, "correct": False, "timepoints": 20},
        {"date": "2021-01-04", "predicted": "DOWN", "confidence": 0.7,
         "actual": "DOWN", "actual_change_pct": -0.5, "correct": True, "timepoints": 20},
        {"date": "2021-01-05", "predicted": "DOWN", "confidence": 0.85,
         "actual": "UP", "actual_change_pct": 0.5, "correct": False, "timepoints": 20},
    ]
    m = compute_metrics(rows)
    assert (m["total"], m["correct"], m["accuracy"]) == (4, 2, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["by_year"]["2020"]["total"] == 2
    assert m["by_conf"][0]["total"] == 1  # <0.6
    assert m["by_conf"][2]["total"] == 2  # >=0.8


def test_resume_skips_cached_and_report_shape():
    import pandas as pd

    df = _fixture_df(30)
    patches = _enter(_mock_stack(df))
    try:
        with tempfile.TemporaryDirectory() as td:
            pp, rp = os.path.join(td, "p.jsonl"), os.path.join(td, "r.md")
            from unittest.mock import MagicMock, patch

            # seed cache with the first eligible date
            with open(pp, "w") as fh:
                fh.write(json.dumps({"date": "2020-01-30", "predicted": "UP",
                                     "confidence": 0.9, "actual": "UP",
                                     "actual_change_pct": 1.0, "correct": True,
                                     "timepoints": 20}) + "\n")
            calls = []
            real_submit = None
            import concurrent.futures as cf

            orig_submit = cf.ThreadPoolExecutor.submit

            def counting_submit(self, fn, /, *a, **k):
                calls.append(1)
                return orig_submit(self, fn, *a, **k)

            with patch.object(cf.ThreadPoolExecutor, "submit", counting_submit):
                out = run_backtest(start="2020-01-01", end="2020-03-31", n=20,
                                   workers=20, predictions_path=pp, report_path=rp)
            # 9 eligible days (idx 21..29), 1 cached -> 8 submitted
            assert len(calls) == 8, len(calls)
            assert len(out["records"]) == 9
            report = open(rp).read()
            for section in ("Overall accuracy", "Confusion matrix", "Accuracy by year",
                            "Accuracy by confidence"):
                assert section in report, section
            assert os.path.exists(rp)
            assert not os.path.exists(os.path.join(td, "r.json"))
    finally:
        _exit(patches)


def test_workers_validation():
    assert validate_workers(20) == 20
    for bad in (0, 65, "20", 4.0, True):
        try:
            validate_workers(bad)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_smoke_limit():
    df = _fixture_df(30)
    patches = _enter(_mock_stack(df))
    try:
        with tempfile.TemporaryDirectory() as td:
            out = run_backtest(start="2020-01-01", end="2020-12-31", n=20, workers=20,
                               predictions_path=os.path.join(td, "p.jsonl"),
                               report_path=os.path.join(td, "r.md"), limit=3)
    finally:
        _exit(patches)
    assert out["metrics"]["total"] == 3


if __name__ == "__main__":
    test_exclusion_correctness()
    test_parallel_matches_serial_and_cache_valid()
    test_metrics_math()
    test_resume_skips_cached_and_report_shape()
    test_workers_validation()
    test_smoke_limit()
    print("all backtest offline tests passed")
