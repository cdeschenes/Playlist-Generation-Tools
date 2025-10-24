#!/usr/bin/env python3
"""Star tracks from an M3U playlist on a Navidrome server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


LOG = logging.getLogger("star_playlist_tracks")

M3U_EXTINF_PATTERN = re.compile(r"^#EXTINF:(?P<duration>-?\d+),(?P<meta>.*)$")


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


class NavidromeError(Exception):
    """Raised when the Navidrome server returns an error."""


class NavidromeAuthError(NavidromeError):
    """Raised when Navidrome authentication fails."""


@dataclass(slots=True)
class PlaylistEntry:
    """Parsed entry from the playlist."""

    index: int
    m3u_path: str
    artist: Optional[str]
    title: str
    duration: Optional[int]


@dataclass(slots=True)
class CandidateTrack:
    """Track candidate returned by the Navidrome API."""

    song_id: str
    artist: Optional[str]
    title: str
    album: Optional[str]
    duration: Optional[int]
    already_starred: bool


@dataclass(slots=True)
class MatchSelection:
    """Selected match for a playlist entry."""

    candidate: CandidateTrack
    title_score: int
    artist_score: int
    duration_delta: Optional[int]


@dataclass(slots=True)
class ProcessingResult:
    """Outcome of processing a playlist entry."""

    entry: PlaylistEntry
    status: str
    match: Optional[MatchSelection]
    reason: Optional[str] = None


def mask_username(user: str) -> str:
    """Return a masked representation of a username for logging."""
    if not user:
        return "***"
    if len(user) <= 2:
        return f"{user[0]}***" if len(user) == 1 else f"{user[0]}***{user[-1]}"
    return f"{user[0]}***{user[-1]}"


def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Make a shallow copy of params with sensitive values masked."""
    sensitive_keys = {"p", "password", "t", "token", "s", "salt"}
    sanitized = {}
    for key, value in params.items():
        if key in sensitive_keys:
            sanitized[key] = "***"
        elif key == "u" and isinstance(value, str):
            sanitized[key] = mask_username(value)
        else:
            sanitized[key] = value
    return sanitized


class NavidromeClient:
    """Client for interacting with a Navidrome server."""

    def __init__(
        self,
        base_url: str,
        user: str,
        *,
        password: Optional[str],
        token: Optional[str],
        salt: Optional[str],
        api_version: str,
        client_name: str,
        timeout: float,
        retries: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user = user
        self._password = password
        self._static_token = token
        self._static_salt = salt
        self.api_version = api_version
        self.client_name = client_name
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff_base = 1.0
        self.session = requests.Session()
        self.masked_user = mask_username(user)

        if self._password:
            LOG.debug("Navidrome client configured for dynamic token authentication.")
        else:
            LOG.debug("Navidrome client configured for static token authentication.")

    def ping(self) -> None:
        """Ensure credentials and connectivity are valid."""
        response = self._request("ping")
        if response.get("status") != "ok":
            raise NavidromeError("Unexpected response from ping endpoint.")

    def search_tracks(self, query: str, song_count: int) -> List[CandidateTrack]:
        """Search for tracks matching the given query."""
        params = {"query": query, "songCount": song_count}
        payload = self._request("search3", params=params)
        search_result = payload.get("searchResult3", {})
        songs = search_result.get("song", [])
        if isinstance(songs, dict):
            songs = [songs]

        candidates: List[CandidateTrack] = []
        for entry in songs:
            song_id = str(entry.get("id"))
            title = entry.get("title")
            if not song_id or not title:
                continue
            duration = entry.get("duration")
            duration_sec = int(duration) if isinstance(duration, (int, float)) else None
            already_starred = "starred" in entry and bool(entry.get("starred"))
            candidates.append(
                CandidateTrack(
                    song_id=song_id,
                    artist=entry.get("artist"),
                    title=title,
                    album=entry.get("album"),
                    duration=duration_sec,
                    already_starred=already_starred,
                )
            )
        LOG.debug(
            "Search query '%s' returned %d candidate(s).",
            query,
            len(candidates),
        )
        return candidates

    def star(self, song_id: str) -> None:
        """Star a track by song ID."""
        self._request("star", params={"id": song_id})

    def _auth_params(self) -> Dict[str, Any]:
        if self._password:
            salt = secrets.token_hex(8)
            token = hashlib.md5((self._password + salt).encode("utf-8")).hexdigest()
            return {"u": self.user, "t": token, "s": salt}
        if self._static_token and self._static_salt:
            return {"u": self.user, "t": self._static_token, "s": self._static_salt}
        raise ConfigurationError(
            "Navidrome credentials are incomplete. Provide SUBSONIC_PASSWORD or SUBSONIC_TOKEN/SUBSONIC_SALT."
        )

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        base_params = {
            "v": self.api_version,
            "c": self.client_name,
            "f": "json",
        }
        url = f"{self.base_url}/rest/{endpoint}.view"
        attempt = 0
        max_attempts = 1 + self.retries
        last_error: Optional[Exception] = None

        while attempt < max_attempts:
            attempt += 1
            auth_params = self._auth_params()
            full_params = {**params, **base_params, **auth_params}
            try:
                LOG.debug(
                    "GET %s with params %s (attempt %d/%d)",
                    url,
                    sanitize_params(full_params),
                    attempt,
                    max_attempts,
                )
                response = self.session.get(url, params=full_params, timeout=self.timeout)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt >= max_attempts:
                    raise NavidromeError(f"Request to {endpoint} failed: {exc}") from exc
                self._sleep_with_backoff(attempt)
                continue

            if response.status_code >= 500:
                last_error = NavidromeError(
                    f"Navidrome endpoint {endpoint} returned HTTP {response.status_code}"
                )
                if attempt >= max_attempts:
                    raise last_error
                self._sleep_with_backoff(attempt)
                continue

            try:
                payload = response.json()
            except ValueError as exc:  # JSON decoding error
                raise NavidromeError(f"Invalid JSON response from {endpoint}: {exc}") from exc

            subsonic_response = payload.get("subsonic-response")
            if subsonic_response is None:
                raise NavidromeError(
                    f"Malformed response from {endpoint}: missing 'subsonic-response' payload."
                )

            status = subsonic_response.get("status")
            if status == "failed":
                error_payload = subsonic_response.get("error", {})
                code = error_payload.get("code")
                message = error_payload.get("message", "Unknown error")
                if code in {40, 41}:
                    raise NavidromeAuthError(f"Authentication failed (code {code}): {message}")
                raise NavidromeError(f"Navidrome error (code {code}): {message}")

            return subsonic_response

        assert last_error is not None  # for mypy/static reasoning
        raise NavidromeError(
            f"Request to {endpoint} failed after {max_attempts} attempts: {last_error}"
        )

    def _sleep_with_backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2 ** (attempt - 1))
        LOG.debug("Retrying after %.2f seconds.", delay)
        time.sleep(delay)


def parse_m3u(path: Path) -> List[PlaylistEntry]:
    """Parse an M3U/M3U8 playlist into entries."""
    if not path.exists():
        raise FileNotFoundError(f"Playlist not found: {path}")

    entries: List[PlaylistEntry] = []
    current_meta: Optional[Dict[str, Any]] = None
    index = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#EXTM3U"):
                continue
            if line.startswith("#"):
                match = M3U_EXTINF_PATTERN.match(line)
                if match:
                    duration_str = match.group("duration")
                    duration: Optional[int]
                    try:
                        duration = int(duration_str)
                        if duration < 0:
                            duration = None
                    except ValueError:
                        duration = None

                    meta = match.group("meta").strip()
                    artist: Optional[str]
                    title: str
                    if " - " in meta:
                        artist_part, title_part = meta.split(" - ", 1)
                        artist = artist_part.strip() or None
                        title = title_part.strip()
                    else:
                        artist = None
                        title = meta.strip()
                    current_meta = {"artist": artist, "title": title, "duration": duration}
                continue

            if current_meta is None:
                LOG.debug("Path '%s' encountered without preceding #EXTINF; skipping.", line)
                continue

            entry = PlaylistEntry(
                index=index,
                m3u_path=line,
                artist=current_meta["artist"],
                title=current_meta["title"],
                duration=current_meta["duration"],
            )
            entries.append(entry)
            index += 1
            current_meta = None

    LOG.info("Parsed %d playlist entries from %s", len(entries), path)
    return entries


def compute_match(entry: PlaylistEntry, candidates: Sequence[CandidateTrack]) -> Optional[MatchSelection]:
    """Select the best candidate for a playlist entry."""
    best: Optional[MatchSelection] = None
    best_key: Optional[tuple] = None

    for idx, candidate in enumerate(candidates):
        title_score = fuzz.token_sort_ratio(entry.title, candidate.title)
        artist_score = 0
        if entry.artist and candidate.artist:
            artist_score = fuzz.token_sort_ratio(entry.artist, candidate.artist)

        duration_delta: Optional[int] = None
        if entry.duration is not None and candidate.duration is not None:
            duration_delta = abs(entry.duration - candidate.duration)

        composite_score = title_score
        if entry.artist:
            composite_score = int(0.65 * title_score + 0.35 * artist_score)

        passes_threshold = composite_score >= 70
        if not passes_threshold and (duration_delta is None or duration_delta > 2):
            continue

        duration_priority = 0
        if duration_delta is None:
            duration_priority = 1
        elif duration_delta > 5:
            duration_priority = 2

        sort_key = (
            -title_score,
            -artist_score,
            duration_priority,
            duration_delta if duration_delta is not None else 9999,
            idx,
        )

        if best is None or sort_key < best_key:
            best = MatchSelection(candidate=candidate, title_score=title_score, artist_score=artist_score, duration_delta=duration_delta)
            best_key = sort_key

    return best


def process_entry(
    entry: PlaylistEntry,
    client: NavidromeClient,
    *,
    dry_run: bool,
    song_count: int,
) -> ProcessingResult:
    """Process a single playlist entry."""
    query = f'{entry.artist or ""} "{entry.title}"'.strip()
    candidates = client.search_tracks(query=query, song_count=song_count)

    if not candidates and entry.artist:
        LOG.debug(
            "No candidates for '%s - %s' with artist+title query; retrying with title only.",
            entry.artist,
            entry.title,
        )
        query = entry.title
        candidates = client.search_tracks(query=query, song_count=song_count)

    if not candidates:
        reason = "No candidates from Navidrome search."
        LOG.warning("No match for '%s' (%s).", entry.title, entry.m3u_path)
        return ProcessingResult(entry=entry, status="skipped_no_match", match=None, reason=reason)

    selection = compute_match(entry, candidates)
    if selection is None:
        reason = "No suitable candidate after scoring."
        LOG.warning("No acceptable match for '%s' (%s).", entry.title, entry.m3u_path)
        return ProcessingResult(entry=entry, status="skipped_no_match", match=None, reason=reason)

    candidate = selection.candidate
    LOG.debug(
        "Selected candidate %s (%s - %s) for '%s' with title_score=%d artist_score=%d duration_delta=%s.",
        candidate.song_id,
        candidate.artist,
        candidate.title,
        entry.title,
        selection.title_score,
        selection.artist_score,
        selection.duration_delta,
    )

    if candidate.already_starred:
        return ProcessingResult(entry=entry, status="already_starred", match=selection, reason=None)

    if dry_run:
        LOG.info(
            "[dry-run] Would star '%s' matched to '%s - %s'.",
            entry.title,
            candidate.artist or "Unknown Artist",
            candidate.title,
        )
        return ProcessingResult(entry=entry, status="starred", match=selection, reason="dry-run")

    client.star(candidate.song_id)
    LOG.info("Starred '%s' -> '%s - %s'.", entry.title, candidate.artist or "Unknown Artist", candidate.title)
    return ProcessingResult(entry=entry, status="starred", match=selection, reason=None)


def configure_logging(log_file: Optional[Path]) -> None:
    """Configure console and optional file logging."""
    LOG.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s %(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    LOG.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        LOG.addHandler(file_handler)
        LOG.debug("File logging enabled at %s", log_file)


def ensure_requirements(args: argparse.Namespace) -> None:
    """Ensure optional dependencies are present if requested."""
    if args.progress and tqdm is None:
        raise RuntimeError("tqdm is required for --progress but is not installed. Install tqdm and retry.")


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows to CSV with UTF-8 encoding."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    LOG.info("Wrote %s", path)


def load_settings() -> Dict[str, Any]:
    """Load and validate environment configuration."""
    load_dotenv()

    url = os.getenv("SUBSONIC_URL")
    if not url:
        raise ConfigurationError("SUBSONIC_URL is required.")

    user = os.getenv("SUBSONIC_USER")
    if not user:
        raise ConfigurationError("SUBSONIC_USER is required.")

    password = os.getenv("SUBSONIC_PASSWORD")
    token = os.getenv("SUBSONIC_TOKEN")
    salt = os.getenv("SUBSONIC_SALT")

    if password:
        token = None
        salt = None
    elif token and salt:
        password = None
    else:
        raise ConfigurationError(
            "Provide either SUBSONIC_PASSWORD or both SUBSONIC_TOKEN and SUBSONIC_SALT."
        )

    api_version = os.getenv("SUBSONIC_API_VERSION", "1.16.1")
    client_name = os.getenv("SUBSONIC_CLIENT", "nd-star-script")

    return {
        "url": url,
        "user": user,
        "password": password,
        "token": token,
        "salt": salt,
        "api_version": api_version,
        "client_name": client_name,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Star Navidrome tracks based on an M3U playlist.")
    parser.add_argument("--m3u", required=True, help="Path to the input .m3u or .m3u8 file.")
    parser.add_argument("--dry-run", action="store_true", help="Perform matching without starring tracks.")
    parser.add_argument("--concurrency", type=int, default=6, help="Number of worker threads (default: 6).")
    parser.add_argument(
        "--log-file",
        default="./star_playlist_tracks.log",
        help="Path for the log file (default: %(default)s).",
    )
    parser.add_argument("--timeout", type=int, default=15, help="Request timeout in seconds (default: 15).")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for Navidrome requests (default: 3).",
    )
    parser.add_argument("--progress", action="store_true", help="Show a progress bar while processing.")
    return parser


def run(args: argparse.Namespace) -> int:
    ensure_requirements(args)

    log_path = Path(args.log_file).expanduser() if args.log_file else None
    configure_logging(log_path)

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        LOG.error("Configuration error: %s", exc)
        return 1

    playlist_path = Path(args.m3u).expanduser()

    try:
        entries = parse_m3u(playlist_path)
    except Exception as exc:  # pylint: disable=broad-except
        LOG.error("Failed to parse playlist: %s", exc)
        return 1

    if not entries:
        LOG.warning("No entries found in playlist %s.", playlist_path)
        return 0

    concurrency = max(1, args.concurrency)
    client = NavidromeClient(
        base_url=settings["url"],
        user=settings["user"],
        password=settings["password"],
        token=settings["token"],
        salt=settings["salt"],
        api_version=settings["api_version"],
        client_name=settings["client_name"],
        timeout=float(args.timeout),
        retries=int(args.retries),
    )

    LOG.info(
        "Using Navidrome server at %s with user %s.",
        client.base_url,
        client.masked_user,
    )

    try:
        client.ping()
    except NavidromeAuthError as exc:
        LOG.error("Authentication failed: %s", exc)
        return 1
    except NavidromeError as exc:
        LOG.error("Unable to reach Navidrome server: %s", exc)
        return 1

    results: List[Optional[ProcessingResult]] = [None] * len(entries)
    total = len(entries)
    song_count = 50

    progress = None
    if args.progress and tqdm is not None:
        progress = tqdm(total=total, desc="Processing", unit="track")

    def update_progress(step: int = 1) -> None:
        if progress is not None:
            progress.update(step)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(
                process_entry,
                entry,
                client,
                dry_run=args.dry_run,
                song_count=song_count,
            ): entry.index
            for entry in entries
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
            except NavidromeError as exc:
                entry = entries[idx]
                LOG.error(
                    "Fatal Navidrome error while processing '%s' (%s): %s",
                    entry.title,
                    entry.m3u_path,
                    exc,
                )
                results[idx] = ProcessingResult(entry=entry, status="error", match=None, reason=str(exc))
            except Exception as exc:  # pylint: disable=broad-except
                entry = entries[idx]
                LOG.exception(
                    "Unexpected error while processing '%s' (%s): %s",
                    entry.title,
                    entry.m3u_path,
                    exc,
                )
                results[idx] = ProcessingResult(entry=entry, status="error", match=None, reason=str(exc))
            else:
                results[idx] = result
            finally:
                update_progress()

    if progress is not None:
        progress.close()

    assert all(result is not None for result in results)
    processed_results = [res for res in results if res is not None]

    starred = sum(1 for res in processed_results if res.status == "starred" and not args.dry_run)
    already_starred = sum(1 for res in processed_results if res.status == "already_starred")
    no_match = sum(1 for res in processed_results if res.status == "skipped_no_match")
    errors = sum(1 for res in processed_results if res.status == "error")
    matched = sum(1 for res in processed_results if res.match is not None)

    LOG.info(
        "Summary: entries=%d matched=%d newly_starred=%d already_starred=%d skipped_no_match=%d errors=%d",
        total,
        matched,
        starred,
        already_starred,
        no_match,
        errors,
    )

    # Prepare CSV rows.
    results_rows: List[Dict[str, Any]] = []
    misses_rows: List[Dict[str, Any]] = []
    already_rows: List[Dict[str, Any]] = []

    for res in processed_results:
        entry = res.entry
        match = res.match.candidate if res.match else None
        result_row = {
            "m3u_path": entry.m3u_path,
            "artist": entry.artist or "",
            "title": entry.title,
            "duration_sec": entry.duration if entry.duration is not None else "",
            "matched_song_id": match.song_id if match else "",
            "matched_artist": match.artist if match else "",
            "matched_title": match.title if match else "",
            "matched_album": match.album if match else "",
            "matched_duration_sec": match.duration if match and match.duration is not None else "",
            "status": res.status,
        }
        results_rows.append(result_row)

        if res.status == "skipped_no_match":
            misses_rows.append(
                {
                    "m3u_path": entry.m3u_path,
                    "artist": entry.artist or "",
                    "title": entry.title,
                    "duration_sec": entry.duration if entry.duration is not None else "",
                    "reason": res.reason or "",
                }
            )
        elif res.status == "already_starred" and match is not None:
            already_rows.append(
                {
                    "m3u_path": entry.m3u_path,
                    "artist": entry.artist or "",
                    "title": entry.title,
                    "matched_song_id": match.song_id,
                    "matched_artist": match.artist if match.artist else "",
                    "matched_title": match.title,
                    "matched_duration_sec": match.duration if match.duration is not None else "",
                }
            )

    write_csv(Path("results.csv"), ["m3u_path", "artist", "title", "duration_sec", "matched_song_id", "matched_artist", "matched_title", "matched_album", "matched_duration_sec", "status"], results_rows)
    write_csv(Path("misses.csv"), ["m3u_path", "artist", "title", "duration_sec", "reason"], misses_rows)
    write_csv(Path("already_starred.csv"), ["m3u_path", "artist", "title", "matched_song_id", "matched_artist", "matched_title", "matched_duration_sec"], already_rows)

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())


README = """
Star Playlist Tracks Script
===========================

Environment (.env) Schema
-------------------------
SUBSONIC_URL=https://music.example.com
SUBSONIC_USER=cody
# Choose ONE approach:
# 1) Provide password (preferred): the script generates salt+token automatically.
SUBSONIC_PASSWORD=your-password-here
# 2) Or provide a precomputed token+salt (leave PASSWORD unset):
# SUBSONIC_TOKEN=md5(password + salt)   # hex
# SUBSONIC_SALT=any-random-string

# Optional:
SUBSONIC_API_VERSION=1.16.1
SUBSONIC_CLIENT=nd-star-script

CLI Usage Examples
------------------
python3 star_playlist_tracks.py --m3u "/Volumes/NAS/Media/Music/Music_Server/_playlists/MyFavorites.m3u" --dry-run --progress
python3 star_playlist_tracks.py --m3u "/Volumes/NAS/Media/Music/Music_Server/_playlists/MyFavorites.m3u"
python3 star_playlist_tracks.py --m3u "./Top Artist Tracks.m3u" --concurrency 8 --log-file ./nd_star.log

Sample Workflow
---------------
1. Configure your `.env` using the schema above.
2. Verify connectivity: the script performs a Navidrome ping on startup and exits with a clear error if credentials are invalid.
3. Run a dry run to see which tracks would be starred:
   python3 star_playlist_tracks.py --m3u "./sample.m3u" --dry-run --progress
4. When satisfied, rerun without `--dry-run` to star the tracks.
5. Review generated CSV reports (`results.csv`, `misses.csv`, `already_starred.csv`) in the working directory.
6. Example M3U snippet expected by the matcher:
   #EXTM3U
   #EXTINF:210,Nils Frahm - Familiar
   /music/Nils Frahm/2011 - Felt [FLAC 24-44.1kHz] [ERATP033DM]/03 - Familiar.flac
   #EXTINF:236,Ólafur Arnalds - Partisans
   /music/Ólafur Arnalds/2021 - The Invisible [FLAC 24-96kHz]/01 - Partisans.flac
   #EXTINF:764,Tool - Invincible
   /music/Tool/2019 - Fear Inoculum [FLAC 24-96kHz] [G010004128304T]/04 - Invincible.flac
   #EXTINF:334,Delvon Lamarr Organ Trio - Careless Whisper
   /music/Delvon Lamarr Organ Trio/2021 - I Told You So [FLAC 24-44.1kHz]/07 - Careless Whisper.flac
   #EXTINF:187,billy woods ft. Boldy James & Gabe 'Nandez - Sauvage
   /music/billy woods/2022 - Aethiopes [FLAC 16-44.1kHz]/04 - Sauvage.flac
   #EXTINF:251,Meat Beat Manifesto - She's Unreal
   /music/Meat Beat Manifesto/1996 - Subliminal Sandwich [FLAC 16-44.1kHz]/1-07 - She's Unreal.flac
"""
