from __future__ import annotations

import json
import socket
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import load_config
from .paths import RAW_DIR, REPORTS_DIR, ensure_dirs

REQUEST_LOG_PATH = REPORTS_DIR / 'api_request_log.jsonl'
ULTRA_DAILY_CAP_DEFAULT = 75000


@dataclass
class RequestBudget:
    daily_cap: int = ULTRA_DAILY_CAP_DEFAULT

    def todays_count(self) -> int:
        if not REQUEST_LOG_PATH.exists():
            return 0
        today = datetime.now(timezone.utc).date().isoformat()
        count = 0
        with REQUEST_LOG_PATH.open('r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if str(row.get('date_utc', '')).startswith(today):
                    count += 1
        return count

    def assert_capacity(self, planned_requests: int = 1) -> None:
        used = self.todays_count()
        if used + planned_requests > self.daily_cap:
            raise RuntimeError(
                f'API-Football daily budget exceeded or too close to limit: used={used} planned={planned_requests} cap={self.daily_cap}'
            )

    def log_request(self, endpoint: str, params: dict[str, Any], status_code: int) -> None:
        ensure_dirs()
        REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'ts_utc': datetime.now(timezone.utc).isoformat(),
            'date_utc': datetime.now(timezone.utc).date().isoformat(),
            'endpoint': endpoint,
            'params': params,
            'status_code': int(status_code),
        }
        with REQUEST_LOG_PATH.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + '\n')


class APIFootballClient:
    def __init__(self, *, sleep_seconds: float | None = None, daily_cap: int = ULTRA_DAILY_CAP_DEFAULT) -> None:
        self.cfg = load_config()
        if not self.cfg.has_live_key:
            raise RuntimeError('API_FOOTBALL_KEY not configured.')
        self.sleep_seconds = float(self.cfg.requests_per_minute and 60.0 / max(self.cfg.requests_per_minute, 1))
        if sleep_seconds is not None:
            self.sleep_seconds = float(sleep_seconds)
        self.budget = RequestBudget(daily_cap=daily_cap)
        self.max_retries = 4

    def _request_url(self, endpoint: str, params: dict[str, Any] | None = None) -> str:
        params = params or {}
        clean = {k: v for k, v in params.items() if v is not None and v != ''}
        query = urlencode(clean, doseq=True)
        base = f"{self.cfg.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return f'{base}?{query}' if query else base

    def get_json(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.budget.assert_capacity(1)
        url = self._request_url(endpoint, params)
        req = Request(url, headers=self.cfg.auth_headers(), method='GET')
        last_error: Exception | None = None
        status = 0
        body = ''
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(req, timeout=60) as resp:
                    body = resp.read().decode('utf-8')
                    status = getattr(resp, 'status', 200)
                last_error = None
                break
            except HTTPError as exc:
                last_error = exc
                status = int(getattr(exc, 'code', 0) or 0)
                # Retry only transient gateway / rate-limit style failures.
                if status not in {408, 425, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise
            except (URLError, ssl.SSLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
            time.sleep(min(30.0, (2 ** attempt) * max(self.sleep_seconds, 0.5)))
        if last_error is not None:
            raise last_error
        self.budget.log_request(endpoint, params or {}, status)
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return json.loads(body)

    def paged_get(self, endpoint: str, base_params: dict[str, Any] | None = None, *, page_param: str = 'page', max_pages: int | None = None) -> list[dict[str, Any]]:
        base_params = dict(base_params or {})
        page = 1
        out: list[dict[str, Any]] = []
        while True:
            params = dict(base_params)
            params[page_param] = page
            payload = self.get_json(endpoint, params)
            out.append(payload)
            paging = payload.get('paging') or {}
            current = int(paging.get('current') or page)
            total = int(paging.get('total') or current)
            if max_pages is not None and page >= max_pages:
                break
            if current >= total:
                break
            page += 1
        return out


def write_raw_json(endpoint_slug: str, rows: Iterable[dict[str, Any]], *, stem: str) -> Path:
    ensure_dirs()
    out = RAW_DIR / f'{stem}__{endpoint_slug}.jsonl'
    with out.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + '\n')
    return out
