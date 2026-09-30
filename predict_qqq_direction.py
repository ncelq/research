"""Predict the next-timepoint direction from N anonymized log returns of QQQ.

Install (per https://docs.typesafe.ai/sdk/python#pip)::

    pip install -r requirements.txt        # pulls ``typesafe-sdk`` (real SDK)
    export TYPESAFE_API_KEY="..."          # required; never hardcode it

Usage::

    python predict_qqq_direction.py [--timepoints N]
    python predict_qqq_direction.py --timepoints 50 --print-prompt
    python predict_qqq_direction.py --no-live --print-prompt   # offline: show prompt only
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys

# ---------------------------------------------------------------------------
# Pure helpers (no network / no SDK import so they are unit-testable offline)
# ---------------------------------------------------------------------------

DEFAULT_TIMEPOINTS = 20
MIN_TIMEPOINTS = 20
MAX_TIMEPOINTS = 200


def validate_num_timepoints(n: int) -> int:
    """Validate the observation-window size ``N`` (allowed 20-200)."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"timepoints must be an int, got {n!r}")
    if not MIN_TIMEPOINTS <= n <= MAX_TIMEPOINTS:
        raise ValueError(
            f"timepoints must be between {MIN_TIMEPOINTS} and {MAX_TIMEPOINTS}, got {n}"
        )
    return n


BANNED_SUBSTRINGS = (
    "qqq",
    "stock",
    "equity",
    "equities",
    "nasdaq",
    "invesco",
    "ticker",
    "symbol",
    "price",
    "close",
    "open",
    "shares",
    "market",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may ",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "$",
    "usd",
)


def compute_log_pct_change(closes: list[float]) -> list[float]:
    """Return ``ln(p_t / p_{t-1}) * 100`` for each consecutive pair.

    ``closes`` must hold at least 2 strictly positive values.
    """
    if len(closes) < 2:
        raise ValueError(f"need >= 2 closes, got {len(closes)}")
    out: list[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if not math.isfinite(prev) or not math.isfinite(cur):
            raise ValueError(f"non-finite close value: prev={prev} cur={cur}")
        if prev <= 0 or cur <= 0:
            raise ValueError(f"close values must be > 0: prev={prev} cur={cur}")
        out.append(math.log(cur / prev) * 100.0)
    for v in out:
        if not math.isfinite(v):
            raise ValueError("computed non-finite logged percentage change")
    return out


def build_state_description(n: int) -> str:
    """Describe the compact state map for ``N`` observed timepoints (target ``N+1``)."""
    n = validate_num_timepoints(n)
    target = n + 1
    return (
        f"DATASTRUCTURE: the state is one compact JSON object with exactly {target} entries. "
        f'Keys are timepoint labels as strings "1".."{target}" in chronological order '
        "(JSON object keys are always strings). "
        f"Values for keys '1' through '{n}' are numbers: the observed logged percentage "
        "change at that timepoint, in percent units rounded to 4 decimals. "
        f'The value for key "{target}" is always null: it marks the next timepoint whose '
        "direction you must predict (up = greater than zero, down = zero or below). "
        f"There are no other keys and no nested objects: only the {target} timepoint entries."
    )


def build_direction_instructions(n: int) -> str:
    """Build the Choice question instructions for ``N`` observed timepoints."""
    n = validate_num_timepoints(n)
    target = n + 1
    return (
        build_state_description(n)
        + f" Using only the numeric pattern of keys '1' through '{n}', "
        f"predict the direction of key '{target}'. "
        "This is a binary choice with a crisp boundary: "
        f"reply 'up' if and only if you expect the timepoint {target} change% "
        "to be greater than zero (any positive value counts, even 0.0001); "
        "reply 'down' if you expect zero or a negative value."
    )


def build_direction_criteria(n: int) -> dict:
    """Build the up/down Choice criteria for ``N`` observed timepoints."""
    n = validate_num_timepoints(n)
    target = n + 1
    return {
        "up": f"timepoint {target} change% is greater than zero; any positive value, even 0.0001, counts as up.",
        "down": f"timepoint {target} change% is zero or below (zero or any negative value).",
    }


# Default-window aliases (N=20) preserved for backward compatibility.
STATE_DESCRIPTION = build_state_description(DEFAULT_TIMEPOINTS)
DIRECTION_INSTRUCTIONS = build_direction_instructions(DEFAULT_TIMEPOINTS)
DIRECTION_CRITERIA = build_direction_criteria(DEFAULT_TIMEPOINTS)


def build_state(log_changes: list[float], n: int | None = None) -> dict:
    """Build the anonymized ``state`` map sent to TypeSafe.

    Returns a compact JSON-serializable dict ``{"1": v1, ..., "N": vN,
    "N+1": None}``: keys ``"1"``..``"N"`` carry the observed logged
    percentage change, key ``str(N+1)`` is ``None`` as the to-predict placeholder.
    ``n`` defaults to ``len(log_changes)``; either way it must be within 20-200.
    No ticker, asset class, price level, date, or calendar reference anywhere.
    """
    if n is None:
        n = len(log_changes)
    n = validate_num_timepoints(n)
    if len(log_changes) != n:
        raise ValueError(f"expected exactly {n} logged changes, got {len(log_changes)}")
    state: dict = {}
    for i, v in enumerate(log_changes, start=1):
        if not math.isfinite(v):
            raise ValueError(f"log change at index {i} is not finite: {v}")
        state[str(i)] = round(float(v), 4)
    state[str(n + 1)] = None
    return state


def build_full_prompt_text(log_changes: list[float], n: int | None = None) -> str:
    """Human-readable rendering of state + question (for --print-prompt / debugging)."""
    import json as _json

    if n is None:
        n = len(log_changes)
    n = validate_num_timepoints(n)
    state = build_state(log_changes, n)
    return "state = " + _json.dumps(state) + "\n\nQuestion: " + build_direction_instructions(n)


_YEAR_RE = None  # compiled lazily (see assert_anonymized)


def assert_anonymized(text: object) -> None:
    """Raise if ``text`` (str or JSON-serializable state) leaks identity/date context."""
    import json as _json
    import re as _re

    global _YEAR_RE
    if _YEAR_RE is None:
        # Year leak = 202x/2030 as a standalone token, not glued to digits or
        # decimal points (so values like 0.2021 or 2.2025 pass, "2026-01-01"
        # or "as of 2026" do not).
        _YEAR_RE = _re.compile(r"(?<![\d.])(202[0-9]|2030)(?![\d.])")
    if not isinstance(text, str):
        try:
            text = _json.dumps(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cannot serialize state for anonymity check: {exc}") from exc
    lowered = text.lower()
    for banned in BANNED_SUBSTRINGS:
        if banned.lower() in lowered:
            raise ValueError(f"prompt leaks banned token {banned!r}")
    if _YEAR_RE.search(lowered):
        raise ValueError("prompt leaks a year reference")


def normalize_direction(choice: str) -> str:
    """Normalize a raw model choice to ``UP`` / ``DOWN``."""
    key = choice.strip().lower().strip(".!?,;:'\"()[]{} ")
    if key in ("up", "u", "positive", "rise", "higher"):
        return "UP"
    if key in ("down", "d", "negative", "fall", "lower", "flat", "unchanged"):
        return "DOWN"
    raise ValueError(f"unrecognized direction choice: {choice!r}")


# ---------------------------------------------------------------------------
# I/O: market data + TypeSafe SDK (imported lazily so pure helpers stay light)
# ---------------------------------------------------------------------------


def fetch_qqq_history(n_closes: int = 21) -> tuple[list[float], datetime.date]:
    """Fetch the latest ``n_closes`` daily QQQ closes plus the last trading date.

    Uses the last N actual trading rows, so weekends/holidays are skipped
    automatically rather than treated as missing days. Returns
    ``(closes, last_trading_date)``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required: pip install -r requirements.txt") from exc

    df = yf.download("QQQ", period="2y", interval="1d", auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned no rows for QQQ (check network access)")
    # yfinance may return MultiIndex columns ("Close", "QQQ"); flatten defensively.
    if hasattr(df.columns, "droplevel"):
        try:
            df.columns = df.columns.droplevel(1)  # type: ignore[attr-defined]
        except (ValueError, IndexError, KeyError, TypeError):
            pass
    col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if col is None:
        raise RuntimeError(f"yfinance frame has no Close column: {list(df.columns)}")
    series = df[col].dropna()
    if len(series) < n_closes:
        raise RuntimeError(f"only {len(series)} trading rows available, need {n_closes}")
    tail = series.iloc[-n_closes:]
    last_ts = tail.index[-1]
    try:
        last_date = last_ts.date()  # pandas Timestamp
    except AttributeError:
        last_date = datetime.date.fromisoformat(str(last_ts)[:10])
    return [float(v) for v in tail.tolist()], last_date


def fetch_qqq_closes(n_closes: int = 21) -> list[float]:
    """Fetch the latest ``n_closes`` daily QQQ closes via yfinance."""
    closes, _ = fetch_qqq_history(n_closes)
    return closes


DEFAULT_OUTPUT = "predictions.jsonl"


def next_trading_weekday(d: datetime.date) -> datetime.date:
    """Roll ``d`` forward to the next Mon-Fri weekday (exchange holidays not checked)."""
    nxt = d + datetime.timedelta(days=1)
    while nxt.weekday() >= 5:  # 5=Sat, 6=Sun
        nxt += datetime.timedelta(days=1)
    return nxt


def build_prediction_record(
    *,
    prediction_date: datetime.date | str,
    direction: str,
    confidence: float,
    n: int,
    model: str | None,
    run_at: datetime.datetime | str | None = None,
) -> dict:
    """Build the JSON-serializable prediction log record (``actual_change_pct`` always null)."""
    n = validate_num_timepoints(n)
    if isinstance(prediction_date, datetime.date):
        pred_str = prediction_date.isoformat()
    else:
        pred_str = str(prediction_date)
    if run_at is None:
        run_str = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    elif isinstance(run_at, datetime.datetime):
        run_str = run_at.isoformat()
    else:
        run_str = str(run_at)
    return {
        "prediction_date": pred_str,
        "direction": normalize_direction(direction),
        "confidence": float(confidence),
        "actual_change_pct": None,
        "timepoints": n,
        "model": model,
        "run_at": run_str,
    }


def append_prediction_record(path: str, record: dict) -> None:
    """Append one record as a single JSON line (all-or-nothing per line)."""
    import json as _json

    line = _json.dumps(record)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def predict_direction(state: dict, model: str | None = None) -> dict:
    """Call TypeSafe System One with a binary up/down Choice question.

    The window size ``N`` is inferred from the state (``len(state) - 1``
    observed entries plus the ``N+1`` null placeholder) and validated 20-200.
    """
    if not isinstance(state, dict) or len(state) < 2:
        raise ValueError("state must be a dict with N observed entries plus an N+1 placeholder")
    n = validate_num_timepoints(len(state) - 1)
    instructions = build_direction_instructions(n)
    criteria = build_direction_criteria(n)
    assert_anonymized(state)
    assert_anonymized(instructions + str(criteria))
    try:
        from typesafe_sdk import Choice, TypeSafeClient
    except ImportError as exc:
        raise RuntimeError(
            "typesafe-sdk is required: pip install typesafe-sdk "
            "(see https://docs.typesafe.ai/sdk/python#pip)"
        ) from exc

    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise RuntimeError("TYPESAFE_API_KEY env var is not set")

    from typesafe_sdk import TypeSafeAPIError  # noqa: PLC0415

    kwargs = {"model": model} if model else {}
    try:
        with TypeSafeClient() as client:
            response = client.system_one(
                state=state,
                questions={
                    "direction": Choice(
                        instructions=instructions,
                        criteria=criteria,
                    )
                },
                **kwargs,  # type: ignore[arg-type]
            )
    except TypeSafeAPIError as exc:
        raise RuntimeError(f"TypeSafe API call failed: {exc}") from exc
    answer = response.choices.get("direction")
    if answer is None:
        # Fallback: some server versions key answers differently.
        answers = getattr(response, "answers", {})
        answer = answers.get("direction") if isinstance(answers, dict) else None
    if answer is None:
        raise RuntimeError("TypeSafe response contained no 'direction' answer")
    direction = normalize_direction(answer.choice)
    return {
        "prediction": direction,
        "choice": answer.choice,
        "confidence": answer.confidence,
        "probabilities": dict(answer.probabilities),
        "model": response.model,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict next-timepoint UP/DOWN from latest N QQQ log returns (anonymized)."
    )
    parser.add_argument(
        "--timepoints",
        type=int,
        default=DEFAULT_TIMEPOINTS,
        help=f"number of observed timepoints (default {DEFAULT_TIMEPOINTS}, allowed {MIN_TIMEPOINTS}-{MAX_TIMEPOINTS})",
    )
    parser.add_argument("--print-prompt", action="store_true", help="print the anonymized prompt")
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="skip the TypeSafe call (useful offline / without API key)",
    )
    parser.add_argument("--model", default=None, help="optional TypeSafe model override")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"prediction log path (default {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="skip appending to the prediction log",
    )
    args = parser.parse_args(argv)
    try:
        n = validate_num_timepoints(args.timepoints)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    target = n + 1

    try:
        closes, last_date = fetch_qqq_history(n + 1)
    except Exception as exc:  # noqa: BLE001 - surface cleanly on CLI
        print(f"error fetching QQQ data: {exc}", file=sys.stderr)
        return 2
    prediction_date = next_trading_weekday(last_date)
    log_changes = compute_log_pct_change(closes)
    # Keep exactly the latest N (fetch gives N+1 closes -> N changes).
    log_changes = log_changes[-n:]
    state = build_state(log_changes, n)

    if args.print_prompt:
        print("---- anonymized prompt (state + question) ----")
        print(build_full_prompt_text(log_changes, n))
        print("----------------------------------------------")

    if args.no_live:
        print("live prediction skipped (--no-live).")
        return 0

    try:
        result = predict_direction(state, model=args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"error predicting direction: {exc}", file=sys.stderr)
        return 1

    print(f"Timepoint {target} prediction: {result['prediction']}")
    print(f"model={result['model']} confidence={result['confidence']:.3f} "
          f"probabilities={result['probabilities']}")
    if not args.no_write:
        try:
            record = build_prediction_record(
                prediction_date=prediction_date,
                direction=result["prediction"],
                confidence=result["confidence"],
                n=n,
                model=result["model"],
            )
            append_prediction_record(args.output, record)
        except Exception as exc:  # noqa: BLE001
            print(f"error writing prediction log: {exc}", file=sys.stderr)
            return 1
        print(f"logged to {args.output}: prediction_date={record['prediction_date']} "
              f"direction={record['direction']} actual_change_pct=null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
