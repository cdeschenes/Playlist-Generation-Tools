"""Central configuration helpers for playlist scripts."""

from __future__ import annotations

import os
import unicodedata
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Load the root .env first, then allow script-local overrides without clobbering.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
load_dotenv(override=False)

LEGACY_ENV_NAMES: Dict[str, list[str]] = {
    "MUSIC_ROOT": ["LIBRARY_ROOT", "MUSIC_DIR", "MEDIA_ROOT"],
    "PLAYLIST_DIR": ["PLAYLIST_ROOT", "PLAYLISTS_DIR"],
    "PLAYLIST_NAME": ["TOP_ARTIST_PLAYLIST_NAME"],
    "PATH_REWRITE_FROM": ["LIBRARY_PATH_PREFIX"],
    "PATH_REWRITE_TO": ["PLAYLIST_PREFIX"],
    "LASTFM_REQUESTS_PER_SEC": ["LASTFM_RATE_LIMIT_PER_SEC"],
    "LASTFM_CACHE_FILE": ["CACHE_FILE"],
    "CSV_INPUT_FILENAME": ["CSV_BASENAME"],
    "FUZZY_MATCH_THRESHOLD": ["FUZZ_THRESHOLD"],
    "FUZZY_TITLE_WEIGHT": ["TITLE_WEIGHT"],
    "FUZZY_ARTIST_WEIGHT": ["ARTIST_WEIGHT"],
    "FUZZY_ALBUM_WEIGHT": ["ALBUM_WEIGHT"],
    "SKIP_RAW_PCM_TAGS": ["SKIP_TAGS_FOR_RAW_PCM"],
    "NONFLAC_PLAYLIST_PATH": ["PLAYLIST_PATH"],
}

_WARNED: set[tuple[str, str]] = set()


def _coerce_str(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def _get_env(name: str) -> Optional[str]:
    candidates = [name] + LEGACY_ENV_NAMES.get(name, [])
    for candidate in candidates:
        raw = os.getenv(candidate)
        if raw is None or raw.strip() == "":
            continue
        if candidate != name:
            _warn_once(candidate, name)
        return raw
    return None


def _warn_once(old_name: str, new_name: str) -> None:
    key = (old_name, new_name)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"Environment variable {old_name} is deprecated; use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = _get_env(name)
    if raw is None or raw.strip() == "":
        return _coerce_str(default) if isinstance(default, str) else default
    return _coerce_str(raw)


def env_bool(name: str, default: Optional[bool] = None) -> bool:
    raw = _get_env(name)
    if raw is None or raw.strip() == "":
        if default is None:
            raise RuntimeError(f"Missing required boolean environment variable: {name}")
        return bool(default)
    normalized = raw.strip().lower()
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise ValueError(f"Environment variable {name} must be boolean-like, got {raw!r}")


def env_int(name: str, default: Optional[int] = None, *, min_value: Optional[int] = None) -> int:
    raw = _get_env(name)
    if raw is None or raw.strip() == "":
        if default is None:
            raise RuntimeError(f"Missing required integer environment variable: {name}")
        value = int(default)
    else:
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"Environment variable {name} must be >= {min_value}, got {value}")
    return value


def env_float(name: str, default: Optional[float] = None) -> float:
    raw = _get_env(name)
    if raw is None or raw.strip() == "":
        if default is None:
            raise RuntimeError(f"Missing required float environment variable: {name}")
        return float(default)
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}") from exc


def env_path(name: str, default: Optional[str] = None) -> Path:
    raw = env_str(name, default)
    if raw is None or raw == "":
        raise RuntimeError(f"Missing required path environment variable: {name}")
    path = Path(raw).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path


def _ensure_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else value + "/"


@dataclass(frozen=True)
class Settings:
    music_root: Path
    playlist_dir: Path
    playlist_name: str
    path_rewrite_from: str
    path_rewrite_to: str
    lastfm_api_key: Optional[str]
    lastfm_requests_per_sec: float
    max_workers: int
    lastfm_cache_file: Path
    csv_input_filename: str
    fuzzy_match_threshold: int
    fuzzy_title_weight: float
    fuzzy_artist_weight: float
    fuzzy_album_weight: float
    skip_raw_pcm_tags: bool
    nonflac_playlist_path: Path
    log_level: str

    @property
    def playlist_path(self) -> Path:
        return self.playlist_dir / self.playlist_name

    @property
    def normalized_path_rewrite_from(self) -> str:
        return _ensure_trailing_slash(self.path_rewrite_from)

    @property
    def normalized_path_rewrite_to(self) -> str:
        return _ensure_trailing_slash(self.path_rewrite_to)

    @classmethod
    def from_env(cls) -> Settings:
        defaults = {
            "MUSIC_ROOT": "/Volumes/NAS/Media/Music/Music_Server/",
            "PLAYLIST_DIR": "/Volumes/NAS/Media/Music/Music_Server/_playlists/",
            "PLAYLIST_NAME": "top_artist_tracks.m3u",
            "PATH_REWRITE_FROM": "/Volumes/NAS/Media/Music/Music_Server/",
            "PATH_REWRITE_TO": "/music/",
            "LASTFM_REQUESTS_PER_SEC": "4",
            "MAX_WORKERS": "8",
            "LASTFM_CACHE_FILE": ".lastfm_cache.json",
            "CSV_INPUT_FILENAME": "Liked_Tracks_clean.csv",
            "FUZZY_MATCH_THRESHOLD": "86",
            "FUZZY_TITLE_WEIGHT": "0.60",
            "FUZZY_ARTIST_WEIGHT": "0.30",
            "FUZZY_ALBUM_WEIGHT": "0.10",
            "SKIP_RAW_PCM_TAGS": "true",
            "NONFLAC_PLAYLIST_PATH": "/Volumes/NAS/Media/Music/Music_Server/_playlists/Look For Replacment.m3u",
            "LOG_LEVEL": "INFO",
        }

        values = {key: env_str(key, defaults.get(key)) for key in defaults}

        music_root = env_path("MUSIC_ROOT", defaults["MUSIC_ROOT"])
        playlist_dir = env_path("PLAYLIST_DIR", defaults["PLAYLIST_DIR"])
        playlist_name = values["PLAYLIST_NAME"] or defaults["PLAYLIST_NAME"]
        path_rewrite_from = values["PATH_REWRITE_FROM"] or defaults["PATH_REWRITE_FROM"]
        path_rewrite_to = values["PATH_REWRITE_TO"] or defaults["PATH_REWRITE_TO"]
        lastfm_api_key = env_str("LASTFM_API_KEY")
        lastfm_requests_per_sec = env_float("LASTFM_REQUESTS_PER_SEC", defaults["LASTFM_REQUESTS_PER_SEC"])
        max_workers = env_int("MAX_WORKERS", defaults["MAX_WORKERS"], min_value=1)

        cache_file_raw = values["LASTFM_CACHE_FILE"] or defaults["LASTFM_CACHE_FILE"]
        cache_file = _resolve_path_relative(cache_file_raw)

        csv_input_filename = values["CSV_INPUT_FILENAME"] or defaults["CSV_INPUT_FILENAME"]
        fuzzy_match_threshold = env_int("FUZZY_MATCH_THRESHOLD", defaults["FUZZY_MATCH_THRESHOLD"], min_value=1)
        fuzzy_title_weight = env_float("FUZZY_TITLE_WEIGHT", defaults["FUZZY_TITLE_WEIGHT"])
        fuzzy_artist_weight = env_float("FUZZY_ARTIST_WEIGHT", defaults["FUZZY_ARTIST_WEIGHT"])
        fuzzy_album_weight = env_float("FUZZY_ALBUM_WEIGHT", defaults["FUZZY_ALBUM_WEIGHT"])
        skip_raw_pcm_tags = env_bool("SKIP_RAW_PCM_TAGS", defaults["SKIP_RAW_PCM_TAGS"] == "true")
        nonflac_playlist_path = env_path("NONFLAC_PLAYLIST_PATH", defaults["NONFLAC_PLAYLIST_PATH"])
        log_level = (values["LOG_LEVEL"] or defaults["LOG_LEVEL"]).upper()

        return cls(
            music_root=music_root,
            playlist_dir=playlist_dir,
            playlist_name=playlist_name,
            path_rewrite_from=path_rewrite_from,
            path_rewrite_to=path_rewrite_to,
            lastfm_api_key=lastfm_api_key,
            lastfm_requests_per_sec=lastfm_requests_per_sec,
            max_workers=max_workers,
            lastfm_cache_file=cache_file,
            csv_input_filename=csv_input_filename,
            fuzzy_match_threshold=fuzzy_match_threshold,
            fuzzy_title_weight=fuzzy_title_weight,
            fuzzy_artist_weight=fuzzy_artist_weight,
            fuzzy_album_weight=fuzzy_album_weight,
            skip_raw_pcm_tags=skip_raw_pcm_tags,
            nonflac_playlist_path=nonflac_playlist_path,
            log_level=log_level,
        )


def _resolve_path_relative(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return path.resolve()
    except OSError:
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance populated from the environment."""
    return Settings.from_env()


def iter_legacy_names(new_name: str) -> Iterable[str]:
    return LEGACY_ENV_NAMES.get(new_name, [])


__all__ = [
    "Settings",
    "env_bool",
    "env_int",
    "env_path",
    "env_str",
    "get_settings",
    "iter_legacy_names",
]
