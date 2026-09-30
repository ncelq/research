"""Offline unit tests for predict_qqq_direction.py (no network / no API key)."""

import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from predict_qqq_direction import (
    DEFAULT_TIMEPOINTS,
    append_prediction_record,
    assert_anonymized,
    build_direction_criteria,
    build_direction_instructions,
    build_full_prompt_text,
    build_prediction_record,
    build_state,
    build_state_description,
    compute_log_pct_change,
    next_trading_weekday,
    normalize_direction,
    validate_num_timepoints,
)


def test_log_pct_change_math():
    closes = [100.0, 101.0, 100.5]
    got = compute_log_pct_change(closes)
    assert len(got) == 2
    assert got[0] == math.log(101 / 100) * 100
    assert got[1] == math.log(100.5 / 101) * 100


def test_default_window_preserved():
    assert DEFAULT_TIMEPOINTS == 20
    vals = [0.1 * i for i in range(20)]
    state = build_state(vals)  # n inferred
    assert list(state.keys()) == [str(i) for i in range(1, 22)]
    assert state["21"] is None
    raw = json.dumps(state)
    assert '"21": null' in raw
    assert_anonymized(state)


def test_custom_n_targets_n_plus_1():
    n = 50
    vals = [0.05 * ((-1) ** i) for i in range(n)]
    state = build_state(vals, n)
    assert len(state) == n + 1
    assert list(state.keys()) == [str(i) for i in range(1, n + 2)]
    assert state[str(n + 1)] is None
    prompt = build_full_prompt_text(vals, n)
    assert f"key '{n + 1}'" in prompt
    assert "DATASTRUCTURE" in prompt
    assert_anonymized(prompt)
    criteria = build_direction_criteria(n)
    assert f"timepoint {n + 1}" in criteria["up"]
    assert_anonymized(build_direction_instructions(n))


def test_bounds_rejected():
    for bad in (19, 201, 0, -5, "50", 20.0, True):
        try:
            validate_num_timepoints(bad)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
    assert validate_num_timepoints(20) == 20
    assert validate_num_timepoints(200) == 200
    # mismatched payload length rejected
    try:
        build_state([0.1] * 20, 30)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for len != n")


def test_choice_boundary_crisp_zero_split():
    from predict_qqq_direction import build_direction_criteria, build_direction_instructions
    instr = build_direction_instructions(20)
    crit = build_direction_criteria(20)
    assert "0.0001" in instr and "0.0001" in crit["up"]
    assert "greater than zero" in crit["up"]
    assert "zero or below" in crit["down"]
    assert "flat" not in (instr + crit["up"] + crit["down"]).lower()


def test_state_description_scales():
    desc = build_state_description(30)
    assert '"31"' in desc and "31 entries" in desc


def test_fetch_sizing_mocked():
    from unittest.mock import patch

    import pandas as pd
    import numpy as np

    dates = pd.date_range("2025-01-01", periods=300, freq="B")
    closes = 600 + np.cumsum(np.random.RandomState(0).randn(300))
    df = pd.DataFrame({"Close": closes}, index=dates)
    with patch("yfinance.download", return_value=df) as dl:
        from predict_qqq_direction import fetch_qqq_closes

        got = fetch_qqq_closes(51)
        assert len(got) == 51
        assert dl.call_args is not None  # history window requested


def test_cli_rejects_out_of_range():
    from predict_qqq_direction import main

    for bad in ("19", "201"):
        try:
            main(["--timepoints", bad])
        except SystemExit as exc:
            assert exc.code == 2, f"expected exit 2 for {bad}, got {exc.code}"
        else:
            raise AssertionError(f"expected SystemExit(2) for --timepoints {bad}")


def test_normalize_direction():
    assert normalize_direction("UP") == "UP"
    assert normalize_direction("  up. ") == "UP"
    assert normalize_direction("down") == "DOWN"
    try:
        normalize_direction("sideways")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for garbage choice")


def test_next_trading_weekday_rolls_weekends():
    import datetime

    fri = datetime.date(2026, 9, 25)  # a Friday
    assert next_trading_weekday(fri) == datetime.date(2026, 9, 28)  # Monday
    sat = datetime.date(2026, 9, 26)
    assert next_trading_weekday(sat) == datetime.date(2026, 9, 28)
    wed = datetime.date(2026, 9, 30)  # a Wednesday
    assert next_trading_weekday(wed) == datetime.date(2026, 10, 1)


def test_prediction_record_shape_null_actual():
    import datetime

    rec = build_prediction_record(
        prediction_date=datetime.date(2026, 9, 30),
        direction="up",
        confidence=0.82,
        n=20,
        model="system-one-x",
        run_at="2026-09-29T09:30:00Z",
    )
    assert rec == {
        "prediction_date": "2026-09-30",
        "direction": "UP",
        "confidence": 0.82,
        "actual_change_pct": None,
        "timepoints": 20,
        "model": "system-one-x",
        "run_at": "2026-09-29T09:30:00Z",
    }
    assert json.loads(json.dumps(rec)) == rec  # JSON round-trip


def test_append_and_no_live_no_write():
    import tempfile
    from unittest.mock import MagicMock, patch

    import pandas as pd
    import numpy as np

    from predict_qqq_direction import main

    dates = pd.date_range("2026-09-01", periods=40, freq="B")
    closes = 600 + np.cumsum(np.random.RandomState(1).randn(40))
    df = pd.DataFrame({"Close": closes}, index=dates)
    last_date = dates[-1].date()
    expected_pred = next_trading_weekday(last_date)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "predictions.jsonl")
        # --no-live writes nothing and creates nothing
        with patch("yfinance.download", return_value=df):
            rc = main(["--timepoints", "20", "--no-live", "--output", out])
        assert rc == 0
        assert not os.path.exists(out)

        # live mocked run appends exactly one line
        fake_answer = MagicMock(choice="up", confidence=0.82, probabilities={"up": 0.82, "down": 0.18})
        fake_resp = MagicMock(model="system-one-x")
        fake_resp.choices = {"direction": fake_answer}
        fake_client = MagicMock()
        fake_client.system_one.return_value = fake_resp
        fake_client.__enter__.return_value = fake_client
        with patch("yfinance.download", return_value=df), patch(
            "typesafe_sdk.TypeSafeClient", return_value=fake_client
        ):
            os.environ["TYPESAFE_API_KEY"] = "dummy-test-key"
            rc = main(["--timepoints", "20", "--output", out])
        assert rc == 0
        lines = open(out).read().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["prediction_date"] == expected_pred.isoformat()
        assert rec["direction"] == "UP"
        assert rec["actual_change_pct"] is None
        assert rec["timepoints"] == 20

        # second live run appends; first line untouched (write-once)
        with patch("yfinance.download", return_value=df), patch(
            "typesafe_sdk.TypeSafeClient", return_value=fake_client
        ):
            rc = main(["--timepoints", "20", "--output", out])
        assert rc == 0
        lines2 = open(out).read().strip().splitlines()
        assert len(lines2) == 2
        assert json.loads(lines2[0])["actual_change_pct"] is None

        # --no-write leaves the file untouched
        with patch("yfinance.download", return_value=df), patch(
            "typesafe_sdk.TypeSafeClient", return_value=fake_client
        ):
            rc = main(["--timepoints", "20", "--output", out, "--no-write"])
        assert rc == 0
        assert len(open(out).read().strip().splitlines()) == 2


def test_banned_token_detector():
    try:
        assert_anonymized("timepoint 1: 0.5 QQQ stock price on 2026-01-01")
    except ValueError:
        pass
    else:
        raise AssertionError("detector should have raised")


def test_detector_allows_202_substrings_in_values():
    # Regression: legit change% values containing '202' must not be flagged;
    # real year leaks still are.
    assert_anonymized({"1": 0.2021, "2": -2.202, "3": 2.2025, "4": 0.5})
    for bad in ("2020-01-01", "as of 2026", "2030 outlook"):
        try:
            assert_anonymized(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"detector should have raised for {bad!r}")


if __name__ == "__main__":
    test_log_pct_change_math()
    test_default_window_preserved()
    test_custom_n_targets_n_plus_1()
    test_bounds_rejected()
    test_state_description_scales()
    test_fetch_sizing_mocked()
    test_cli_rejects_out_of_range()
    test_normalize_direction()
    test_choice_boundary_crisp_zero_split()
    test_detector_allows_202_substrings_in_values()
    test_next_trading_weekday_rolls_weekends()
    test_prediction_record_shape_null_actual()
    test_append_and_no_live_no_write()
    test_banned_token_detector()
    print("all offline tests passed")
