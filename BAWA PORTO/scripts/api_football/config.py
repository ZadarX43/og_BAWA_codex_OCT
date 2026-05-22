from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv as _dotenv_load
except Exception:  # pragma: no cover - optional dependency during scaffold stage
    _dotenv_load = None


def _load_local_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_dotenv() -> None:
    if _dotenv_load is not None:
        _dotenv_load()
    _load_local_env()


load_dotenv()


@dataclass
class APIFootballConfig:
    base_url: str = field(default_factory=lambda: os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io"))
    api_key: str = field(default_factory=lambda: os.getenv("API_FOOTBALL_KEY", ""))
    api_host: str = field(default_factory=lambda: os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io"))
    league_ids: str = field(default_factory=lambda: os.getenv("API_FOOTBALL_LEAGUE_IDS", ""))
    seasons: str = field(default_factory=lambda: os.getenv("API_FOOTBALL_SEASONS", ""))
    requests_per_minute: int = field(default_factory=lambda: int(os.getenv("API_FOOTBALL_RPM", "10")))

    @property
    def has_live_key(self) -> bool:
        return bool(self.api_key.strip())

    def auth_headers(self) -> dict[str, str]:
        return {"x-apisports-key": self.api_key} if self.has_live_key else {}


def load_config() -> APIFootballConfig:
    return APIFootballConfig()
