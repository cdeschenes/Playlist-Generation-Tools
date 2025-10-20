"""Helpers for loading configuration from a local .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

ENV_PATH = Path(__file__).resolve().parent / ".env"

# Public API -----------------------------------------------------------------#


def load_env(path: Path = ENV_PATH) -> None:
    """Populate os.environ with variables defined in *path* if they are unset."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = _parse_env_line(line)
        if not key:
            continue
        os.environ.setdefault(key, value)


def get_str(name: str, default: Optional[str] = None) -> str:
    raw = _lookup(name, default)
    if raw is None:
        raise RuntimeError(f"Missing required string environment variable: {name}")
    return raw


def get_int(name: str, default: Optional[int] = None) -> int:
    raw = _lookup(name, default)
    if raw is None:
        raise RuntimeError(f"Missing required integer environment variable: {name}")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def get_float(name: str, default: Optional[float] = None) -> float:
    raw = _lookup(name, default)
    if raw is None:
        raise RuntimeError(f"Missing required float environment variable: {name}")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}") from exc


def get_bool(name: str, default: Optional[Union[bool, str]] = None) -> bool:
    raw = _lookup(name, default)
    if raw is None:
        raise RuntimeError(f"Missing required boolean environment variable: {name}")
    if isinstance(raw, bool):
        return raw

    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be boolean-like, got {raw!r}")


def get_path(name: str, default: Optional[Union[str, os.PathLike[str]]] = None) -> Path:
    raw = _lookup(name, default)
    if raw is None:
        raise RuntimeError(f"Missing required path environment variable: {name}")
    return Path(str(raw)).expanduser()


# Internal helpers -----------------------------------------------------------#


def _lookup(name: str, default: Optional[Union[str, int, float, bool, os.PathLike[str]]] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        if default is None:
            return None
        if isinstance(default, os.PathLike):
            return os.fspath(default)
        return str(default)
    return value.strip()


def _parse_env_line(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return "", ""

    if stripped.lower().startswith("export "):
        stripped = stripped[7:].strip()

    if "=" not in stripped:
        return "", ""

    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return "", ""

    value = _strip_inline_comment(raw_value.strip())
    value = _strip_quotes(value.strip())
    return key, value


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    result = []

    for char in value:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        result.append(char)

    return "".join(result).rstrip()


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
