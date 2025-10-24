"""Utility to regenerate a playlist with all non-FLAC audio files.

Configuration is read from environment variables (see .env).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from config import Settings, get_settings

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

# --- Configuration ---------------------------------------------------------
SETTINGS: Settings = get_settings()
MUSIC_ROOT = SETTINGS.music_root
PLAYLIST_PATH = SETTINGS.nonflac_playlist_path
# Prefix used inside the playlist so paths look like they are under /music
PLAYLIST_PREFIX = SETTINGS.path_rewrite_to.rstrip("/")

# Known audio file extensions (lowercase) we care about; adjust if needed.
# FLAC is intentionally excluded later so only non-FLAC formats are emitted.
AUDIO_EXTENSIONS = {
    ".3ga",
    ".aac",
    ".aif",
    ".aiff",
    ".alac",
    ".ape",
    ".au",
    ".dts",
    ".m1a",
    ".m2a",
    ".m4a",
    ".m4b",
    ".m4p",
    ".mid",
    ".midi",
    ".mka",
    ".mod",
    ".mp1",
    ".mp2",
    ".mp3",
    ".mpc",
    ".mpga",
    ".oga",
    ".ogg",
    ".opus",
    ".ra",
    ".raw",
    ".spx",
    ".tta",
    ".wav",
    ".wave",
    ".webm",
    ".wma",
    ".wv",
    ".flac",  # included for completeness but filtered out below
}

TARGET_EXTENSIONS = {ext for ext in AUDIO_EXTENSIONS if ext != ".flac"}
# ---------------------------------------------------------------------------


def iter_with_basic_progress(iterable: Iterable[Path], *, report_every: int = 1000) -> Iterable[Path]:
    """Yield items while emitting textual progress updates when tqdm is unavailable."""
    count = 0
    for count, item in enumerate(iterable, 1):
        if count % report_every == 0:
            print(f"Scanning library... {count:,} paths processed", end="\r", flush=True)
        yield item
    if count:
        print(f"Scanning library... {count:,} paths processed", end="\r", flush=True)
    print()  # ensure newline after progress output


def normalize_extension(path: Path) -> str:
    return path.suffix.lower()


def is_target_audio(path: Path) -> bool:
    ext = normalize_extension(path)
    return ext in TARGET_EXTENSIONS


def find_non_flac_audio_files(root: Path, *, show_progress: bool = True) -> list[Path]:
    """Return a sorted list of non-FLAC audio files under *root*."""
    if not root.exists():
        raise FileNotFoundError(f"Music root does not exist: {root}")

    files: list[Path] = []
    iterator = root.rglob("*")
    progress: Optional[object] = None
    if show_progress and tqdm is not None:
        progress = tqdm(iterator, desc="Scanning library", unit="path", dynamic_ncols=True)
        iterable: Iterable[Path] = progress
    elif show_progress:
        iterable = iter_with_basic_progress(iterator)
    else:
        iterable = iterator

    for candidate in iterable:
        if not candidate.is_file():
            continue
        if is_target_audio(candidate):
            files.append(candidate)

    if progress is not None:
        progress.close()

    # Sort paths for deterministic playlist output
    files.sort(key=lambda p: str(p).casefold())
    return files


def to_playlist_entry(path: Path) -> str:
    try:
        relative = path.relative_to(MUSIC_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path {path} is outside MUSIC_ROOT {MUSIC_ROOT}") from exc
    return f"{PLAYLIST_PREFIX}/{relative.as_posix()}"


def write_playlist(entries: Iterable[Path], playlist_path: Path) -> None:
    playlist_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [to_playlist_entry(path) + "\n" for path in entries]
    playlist_path.write_text("".join(lines), encoding="utf-8")



def main() -> None:
    audio_files = find_non_flac_audio_files(MUSIC_ROOT)
    if not audio_files:
        print("No non-FLAC audio files found; playlist left untouched.")
        return

    write_playlist(audio_files, PLAYLIST_PATH)
    print(f"Wrote {len(audio_files)} entries to {PLAYLIST_PATH}")


if __name__ == "__main__":
    main()
