from __future__ import annotations

from .config import load_config

ENDPOINTS = {
    "fixtures": "/fixtures",
    "lineups": "/fixtures/lineups",
    "players": "/fixtures/players",
    "events": "/fixtures/events",
    "injuries": "/injuries",
    "odds": "/odds",
    "odds_live": "/odds/live",
    "status": "/status",
}


def build_request_parts(endpoint_key: str, **params):
    cfg = load_config()
    if not cfg.has_live_key:
        raise RuntimeError("API_FOOTBALL_KEY not configured yet. Foundation scaffold only.")
    if endpoint_key not in ENDPOINTS:
        raise KeyError(f"Unknown API-Football endpoint key: {endpoint_key}")
    return {
        "url": f"{cfg.base_url}{ENDPOINTS[endpoint_key]}",
        "headers": cfg.auth_headers(),
        "params": params,
    }


def fetch_endpoint(endpoint_key: str, **params):
    request = build_request_parts(endpoint_key, **params)
    raise NotImplementedError(
        "Live fetch implementation is intentionally deferred until endpoint-specific fetchers are built. "
        f"Prepared request for {request['url']}."
    )


def main() -> None:
    request = build_request_parts("status")
    print("API-Football config looks ready.")
    print(f"Base URL: {request['url']}")
    print(f"Auth header present: {bool(request['headers'])}")


if __name__ == "__main__":
    main()
