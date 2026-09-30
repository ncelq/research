"""Backtest the anonymized QQQ direction predictor over 2020-01-01..2026-08-30.

For each trading day T in range (with enough history), builds the state from
the N log changes strictly *before* T, predicts T via the client-default
TypeSafe model (same path as predict_qqq_direction.py, no --model flag),
compares against T's realized direction, and writes a Markdown accuracy report.

Predictions run in a thread pool (default 20 workers). Results are cached in
``backtest_predictions.jsonl`` so reruns skip completed days (skip-and-resume).

Usage::

    export TYPESAFE_API_KEY="..."
    python backtest_qqq_direction.py
    python backtest_qqq_direction.py --timepoints 20 --workers 20 --limit 5  # smoke test
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import threading

from predict_qqq_direction import (
    DEFAULT_TIMEPOINTS,
    MAX_TIMEPOINTS,
    MIN_TIMEPOINTS,
    build_state,
    compute_log_pct_change,
    predict_direction,
    validate_num_timepoints,
)

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-08-30"
DEFAULT_PREDICTIONS = "backtest_predictions.jsonl"
DEFAULT_REPORT = "backtest_report.md"
DEFAULT_WORKERS = 20
MIN_WORKERS = 1
MAX_WORKERS = 64


def validate_workers(w: int) -> int:
    """Validate the thread-pool size (allowed 1-32)."""
    if isinstance(w, bool) or not isinstance(w, int):
        raise ValueError(f"workers must be an int, got {w!r}")
    if not MIN_WORKERS <= w <= MAX_WORKERS:
        raise ValueError(f"workers must be between {MIN_WORKERS} and {MAX_WORKERS}, got {w}")
    return w


def fetch_range_closes(start: str, end: str) -> tuple[list[float], list[datetime.date]]:
    """Fetch daily QQQ closes covering [start, end]; returns (closes, dates)."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required: pip install -r requirements.txt") from exc

    df = yf.download("QQQ", start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned no rows for QQQ (check network access)")
    if hasattr(df.columns, "droplevel"):
        try:
            df.columns = df.columns.droplevel(1)  # type: ignore[attr-defined]
        except (ValueError, IndexError, KeyError, TypeError):
            pass
    col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if col is None:
        raise RuntimeError(f"yfinance frame has no Close column: {list(df.columns)}")
    series = df[col].dropna()
    closes = [float(v) for v in series.tolist()]
    dates = [ts.date() for ts in series.index.tolist()]
    return closes, dates


def direction_of(change_pct: float) -> str:
    """Realized direction: UP iff strictly positive, else DOWN (matches live)."""
    return "UP" if change_pct > 0 else "DOWN"


def compute_metrics(records: list[dict]) -> dict:
    """Compute accuracy metrics over successful (non-error) prediction records."""
    ok = [r for r in records if "error" not in r]
    total = len(ok)
    correct = sum(1 for r in ok if r["correct"])
    tp = sum(1 for r in ok if r["predicted"] == "UP" and r["actual"] == "UP")
    fp = sum(1 for r in ok if r["predicted"] == "UP" and r["actual"] == "DOWN")
    tn = sum(1 for r in ok if r["predicted"] == "DOWN" and r["actual"] == "DOWN")
    fn = sum(1 for r in ok if r["predicted"] == "DOWN" and r["actual"] == "UP")
    n_up_actual = sum(1 for r in ok if r["actual"] == "UP")
    baseline = max(n_up_actual, total - n_up_actual) / total if total else 0.0
    by_year: dict[str, dict] = {}
    for r in ok:
        y = r["date"][:4]
        b = by_year.setdefault(y, {"total": 0, "correct": 0})
        b["total"] += 1
        b["correct"] += 1 if r["correct"] else 0
    buckets = [("conf<0.6", 0.0, 0.6), ("conf0.6-0.8", 0.6, 0.8), ("conf>=0.8", 0.8, 2.0)]
    by_conf = []
    for name, lo, hi in buckets:
        rows = [r for r in ok if lo <= r["confidence"] < hi]
        by_conf.append({
            "bucket": name,
            "total": len(rows),
            "correct": sum(1 for r in rows if r["correct"]),
            "accuracy": (sum(1 for r in rows if r["correct"]) / len(rows)) if rows else 0.0,
        })
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
        "baseline": baseline,
        "by_year": by_year,
        "by_conf": by_conf,
    }


def render_report(metrics: dict, *, start: str, end: str, n: int, workers: int,
                  warmup_skipped: int, error_skipped: int) -> str:
    """Render the Markdown-only accuracy report."""
    import datetime as _dt

    lines = [
        "# QQQ Direction Backtest Report",
        "",
        f"- Range: {start}..{end}",
        f"- Timepoints (N): {n} (target: N+1)",
        f"- Workers: {workers}",
        "- Model: client default (same path as predict_qqq_direction.py)",
        f"- Generated: {_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
        f"- Predicted days: {metrics['total']}",
        f"- Warmup-skipped days: {warmup_skipped}",
        f"- Error-skipped days: {error_skipped}",
        "",
        "## Overall accuracy",
        "",
        f"- Correct: {metrics['correct']}/{metrics['total']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Majority-class baseline: {metrics['baseline']:.4f}",
        "",
        "## Confusion matrix (UP = positive)",
        "",
        "|  | Actual UP | Actual DOWN |",
        "|---|---|---|",
        f"| Predicted UP | {metrics['tp']} | {metrics['fp']} |",
        f"| Predicted DOWN | {metrics['fn']} | {metrics['tn']} |",
        "",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        "",
        "## Accuracy by year",
        "",
        "| Year | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for year in sorted(metrics["by_year"]):
        b = metrics["by_year"][year]
        acc = b["correct"] / b["total"] if b["total"] else 0.0
        lines.append(f"| {year} | {b['correct']} | {b['total']} | {acc:.4f} |")
    lines += [
        "",
        "## Accuracy by confidence",
        "",
        "| Bucket | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for b in metrics["by_conf"]:
        lines.append(f"| {b['bucket']} | {b['correct']} | {b['total']} | {b['accuracy']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def load_cache(path: str) -> dict[str, dict]:
    """Load cached records keyed by date (missing file -> empty)."""
    import json as _json
    import os as _os

    cached: dict[str, dict] = {}
    if not _os.path.exists(path):
        return cached
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            cached[rec["date"]] = rec
    return cached


def predict_one(target_date: datetime.date, state: dict, actual_change: float,
                actual: str, n: int) -> dict:
    """Run one backtest prediction (worker-thread body; no shared mutation)."""
    try:
        result = predict_direction(state)
        predicted = result["prediction"]
        return {
            "date": target_date.isoformat(),
            "predicted": predicted,
            "confidence": float(result["confidence"]),
            "actual": actual,
            "actual_change_pct": round(float(actual_change), 4),
            "correct": predicted == actual,
            "timepoints": n,
        }
    except Exception as exc:  # noqa: BLE001 - skip-and-resume
        return {"date": target_date.isoformat(), "error": str(exc)}


def run_backtest(*, start: str, end: str, n: int, workers: int,
                 predictions_path: str, report_path: str,
                 limit: int | None = None) -> dict:
    """Execute the full backtest; returns {metrics, report, records}."""
    import json as _json

    start_d = datetime.date.fromisoformat(start)
    end_d = datetime.date.fromisoformat(end)
    # History buffer: N+1 closes before the first target (~1.4x calendar days, min 400).
    fetch_start = (start_d - datetime.timedelta(days=max(400, int((n + 5) * 1.4)))).isoformat()
    fetch_end = (end_d + datetime.timedelta(days=1)).isoformat()
    closes, dates = fetch_range_closes(fetch_start, fetch_end)
    changes = compute_log_pct_change(closes)

    # Eligible targets: in-range trading days with N+1 prior closes (index j >= N+1).
    eligible: list[int] = [
        j for j, d in enumerate(dates)
        if start_d <= d <= end_d and j >= n + 1
    ]
    warmup_skipped = sum(
        1 for j, d in enumerate(dates)
        if start_d <= d <= end_d and j < n + 1
    )

    cached = load_cache(predictions_path)
    todo = [j for j in eligible if dates[j].isoformat() not in cached]
    if limit is not None:
        todo = todo[:limit]

    lock = threading.Lock()
    done = 0
    total_todo = len(todo)
    print(f"backtest: {len(eligible)} eligible days, {len(cached)} cached, {total_todo} to predict "
          f"(N={n}, workers={workers})")

    def append_record(rec: dict) -> None:
        with lock:
            with open(predictions_path, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps(rec) + "\n")
                fh.flush()

    # Pre-build states on the main thread (pure + deterministic), model calls in pool.
    tasks = []
    for j in todo:
        state = build_state(changes[j - n - 1:j - 1], n)
        actual_change = changes[j - 1]
        actual = direction_of(actual_change)
        tasks.append((dates[j], state, actual_change, actual))

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_date = {
            pool.submit(predict_one, d, s, ac, a, n): d for d, s, ac, a in tasks
        }
        try:
            for fut in concurrent.futures.as_completed(future_to_date):
                rec = fut.result()
                results.append(rec)
                append_record(rec)
                cached[rec["date"]] = rec
                done += 1
                if done % 25 == 0 or done == total_todo:
                    print(f"backtest: {done}/{total_todo} predictions done")
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print(f"backtest: interrupted at {done}/{total_todo}; rerun resumes from cache")
            raise

    records = [cached[dates[j].isoformat()] for j in eligible if dates[j].isoformat() in cached]
    metrics = compute_metrics(records)
    error_skipped = sum(1 for r in records if "error" in r)
    report = render_report(metrics, start=start, end=end, n=n, workers=workers,
                           warmup_skipped=warmup_skipped, error_skipped=error_skipped)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print(f"backtest: report written to {report_path}; cache at {predictions_path}")
    return {"metrics": metrics, "report": report, "records": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest the anonymized QQQ direction predictor (20-thread parallel)."
    )
    parser.add_argument("--timepoints", type=int, default=DEFAULT_TIMEPOINTS,
                        help=f"previous datapoints per prediction (default {DEFAULT_TIMEPOINTS}, "
                             f"allowed {MIN_TIMEPOINTS}-{MAX_TIMEPOINTS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"parallel prediction threads (default {DEFAULT_WORKERS}, "
                             f"allowed {MIN_WORKERS}-{MAX_WORKERS})")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of predictions this run (smoke tests)")
    args = parser.parse_args(argv)
    try:
        n = validate_num_timepoints(args.timepoints)
        workers = validate_workers(args.workers)
        datetime.date.fromisoformat(args.start)
        datetime.date.fromisoformat(args.end)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    if args.start > args.end:
        parser.error("--start must be <= --end")
        return 2
    try:
        run_backtest(start=args.start, end=args.end, n=n, workers=workers,
                     predictions_path=args.predictions, report_path=args.report,
                     limit=args.limit)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error running backtest: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
