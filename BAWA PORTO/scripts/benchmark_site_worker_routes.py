#!/usr/bin/env python3
"""Small cached-load benchmark for the D1-backed site Worker routes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "https://odds-genius-worker-site-data-test.hughcwade.workers.dev"
DEFAULT_ROUTES = {
    "current_fixtures": "/api/site/fixtures/current?limit=200",
    "fixture_detail": "/api/site/fixtures/2026_05_09_Club_Brugge_Sint_Truiden",
    "team_detail": "/api/site/teams/belgium_pro/club_brugge",
}


@dataclass
class Sample:
    http_status: int
    network_ms: float
    worker_ms: float
    bytes_read: int
    cache_status: str


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile_value) - 1))
    return ordered[index]


def summarize(samples: list[Sample]) -> dict[str, Any]:
    network = [sample.network_ms for sample in samples]
    worker = [sample.worker_ms for sample in samples]
    cache_counts: dict[str, int] = {}
    for sample in samples:
        cache_counts[sample.cache_status] = cache_counts.get(sample.cache_status, 0) + 1
    return {
        "count": len(samples),
        "http_statuses": sorted({sample.http_status for sample in samples}),
        "bytes_median": int(statistics.median(sample.bytes_read for sample in samples)),
        "cache_statuses": cache_counts,
        "network_total": {
            "median_ms": round(statistics.median(network), 2),
            "p95_ms": round(percentile(network, 0.95), 2),
            "max_ms": round(max(network), 2),
        },
        "worker_elapsed": {
            "median_ms": round(statistics.median(worker), 2),
            "p95_ms": round(percentile(worker, 0.95), 2),
            "max_ms": round(max(worker), 2),
        },
    }


def request_json(base_url: str, route: str) -> Sample:
    url = f"{base_url.rstrip('/')}{route}"
    request = urllib.request.Request(url, headers={"accept": "application/json", "user-agent": "odds-genius-site-benchmark/1"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        status = response.status
        cache_status = response.headers.get("x-og-site-cache", "")
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = json.loads(body.decode("utf-8"))
    body_worker_ms = float((payload.get("meta") or {}).get("worker_elapsed_ms") or 0)
    worker_ms = 0.0 if cache_status == "HIT" else body_worker_ms
    return Sample(
        http_status=status,
        network_ms=elapsed_ms,
        worker_ms=worker_ms,
        bytes_read=len(body),
        cache_status=cache_status or "UNKNOWN",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark D1-backed site Worker routes.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, Any] = {}
    for name, route in DEFAULT_ROUTES.items():
        for _ in range(max(0, args.warmup)):
            request_json(args.base_url, route)
        samples = [request_json(args.base_url, route) for _ in range(max(1, args.iterations))]
        results[name] = summarize(samples)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
