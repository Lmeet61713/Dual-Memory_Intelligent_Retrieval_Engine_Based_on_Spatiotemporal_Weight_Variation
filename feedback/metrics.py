# feedback/metrics.py
"""
Metrics recording and aggregation for the feedback system.

Provides:
- record_metric()  -- append a JSONL metric entry (thread-safe)
- get_stats()      -- aggregate metrics over a time window
"""
from __future__ import annotations
import json
import os
import time
import threading
import config

# --- directory ---
_METRICS_DIR = getattr(config, "FEEDBACK_METRICS_DIR", None)
if _METRICS_DIR is None:
    _METRICS_DIR = os.path.join(config.FEEDBACK_DIR, "metrics")

_metrics_lock = threading.Lock()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _get_metrics_file_path(day: str | None = None) -> str:
    if day is None:
        day = time.strftime("%Y-%m-%d")
    os.makedirs(_METRICS_DIR, exist_ok=True)
    return os.path.join(_METRICS_DIR, f"{day}.jsonl")


def _compute_percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (p in 0..100) of a sorted list."""
    if not values:
        return 0.0
    k = (p / 100.0) * (len(values) - 1)
    f = int(k)
    c = k - f
    if f + 1 < len(values):
        return values[f] + c * (values[f + 1] - values[f])
    return values[f]


def _empty_stats() -> dict:
    return {
        "total": 0,
        "by_type": {"location_wrong": 0, "memory_wrong": 0, "persona_wrong": 0},
        "success_rate": 1.0,
        "avg_latency_ms": 0,
        "p95_latency_ms": 0,
        "p99_latency_ms": 0,
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def record_metric(type: str, latency_ms: float, success: bool) -> None:
    """Append one metric entry to today's JSONL file (thread-safe)."""
    entry = {
        "type": type,
        "latency_ms": latency_ms,
        "success": success,
        "timestamp": time.time(),
    }
    filepath = _get_metrics_file_path()
    try:
        with _metrics_lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # silently ignore — metrics must never break the caller


def get_stats(hours: int = 24) -> dict:
    """Aggregate metrics recorded in the last `hours` hours.

    Returns:
        {
            total, by_type: {location_wrong, memory_wrong, persona_wrong},
            success_rate, avg_latency_ms, p95_latency_ms, p99_latency_ms
        }
    """
    now = time.time()
    cutoff = now - hours * 3600

    entries: list[dict] = []

    if not os.path.exists(_METRICS_DIR):
        return _empty_stats()

    for filename in sorted(os.listdir(_METRICS_DIR)):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(_METRICS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("timestamp", 0) >= cutoff:
                        entries.append(entry)
        except OSError:
            continue

    if not entries:
        return _empty_stats()

    total = len(entries)
    by_type = {"location_wrong": 0, "memory_wrong": 0, "persona_wrong": 0}
    success_count = 0
    latencies: list[float] = []

    for e in entries:
        t = e.get("type", "")
        if t in by_type:
            by_type[t] += 1
        if e.get("success", False):
            success_count += 1
        lat = e.get("latency_ms")
        if lat is not None:
            latencies.append(lat)

    success_rate = success_count / total if total > 0 else 1.0
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lats = sorted(latencies)
    p95 = _compute_percentile(sorted_lats, 95)
    p99 = _compute_percentile(sorted_lats, 99)

    return {
        "total": total,
        "by_type": by_type,
        "success_rate": round(success_rate, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== metrics self-test ===")
    record_metric("location_wrong", 50.5, True)
    record_metric("memory_wrong", 120.1, False)
    record_metric("persona_wrong", 80.0, True)
    record_metric("location_wrong", 200.0, True)
    print("Recorded 4 metrics.")
    stats = get_stats(24)
    print(f"Stats (24h): {json.dumps(stats, indent=2, ensure_ascii=False)}")
    print("=== self-test done ===")
