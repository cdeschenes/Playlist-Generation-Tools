#!/usr/bin/env python3
"""Generate a playlist of top-played tracks per album using Last.fm global playcounts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, List, Optional, Sequence

from config import Settings as AppSettings, get_settings
import requests
from mutagen import File as MutagenFile
from mutagen import MutagenError
from rapidfuzz import fuzz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from rich.logging import RichHandler

    HAVE_RICH = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_RICH = False

LOG = logging.getLogger("build_top_artist_tracks")

AUDIO_EXTENSIONS = {
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".alac",
    ".wav",
    ".aiff",
    ".aif",
    ".ogg",
    ".opus",
    ".wma",
}

LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"
MATCH_THRESHOLD = 85

BRACKETED_RE = re.compile(r"\(.*?\)|\[.*?\]|\{.*?\}")
VERSION_SUFFIX_RE = re.compile(r"\b(remaster(?:ed)?|live|mono mix|stereo mix|version)\b.*$")
YEAR_RE = re.compile(r"(19|20)\d{2}")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

@dataclass(frozen=True)
class AlbumKey:
    normalized_album_artist: str
    normalized_album_title: str


@dataclass
class TrackMeta:
    path: Path
    album: str
    album_artist: str
    artist: str
    title: str
    track_number: Optional[int]
    track_total: Optional[int]
    disc_number: Optional[int]
    disc_total: Optional[int]
    date: Optional[str]
    original_date: Optional[str]
    year: Optional[int]
    musicbrainz_track_id: Optional[str]
    musicbrainz_release_id: Optional[str]
    duration: Optional[int]
    normalized_album: str
    normalized_album_artist: str
    normalized_artist: str
    normalized_title: str


@dataclass
class AlbumData:
    index: int
    key: AlbumKey
    album_artist: str
    album_title: str
    year: Optional[int]
    tracks: List[TrackMeta] = field(default_factory=list)


@dataclass
class TrackMatch:
    track: TrackMeta
    playcount: int
    matched_name: str
    matched_artist: str
    match_score: int
    artist_score: int
    title_score: int
    remote_duration: Optional[int]


@dataclass
class PlaylistEntry:
    album_artist: str
    album_title: str
    normalized_album_artist: str
    normalized_album_title: str
    year: Optional[int]
    track: TrackMeta
    playcount: int
    matched_artist: str
    matched_title: str
    remote_duration: Optional[int]


class LastFMClient:
    def __init__(self, api_key: str, rate_limit_per_sec: float, cache_path: Path) -> None:
        self.api_key = api_key
        self.rate_interval = 1.0 / max(rate_limit_per_sec, 0.01)
        self.cache_path = cache_path
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0
        self._dirty = False
        self.cache_hits = 0
        self.api_calls = 0

        self.session = requests.Session()
        retry = Retry(
            total=5,
            read=5,
            connect=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                self._cache = payload
                LOG.debug("Loaded %d cached Last.fm responses", len(self._cache))
            else:
                LOG.warning("Cache file %s did not contain a JSON object; ignoring", self.cache_path)
        except Exception as exc:
            LOG.warning("Failed to read cache %s: %s", self.cache_path, exc)

    def save_cache(self) -> None:
        if not self._dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(self.cache_path.parent)) as tmp:
                json.dump(self._cache, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
                temp_name = tmp.name
            os.replace(temp_name, self.cache_path)
            LOG.debug("Persisted Last.fm cache to %s", self.cache_path)
            self._dirty = False
        except Exception as exc:
            LOG.warning("Failed to write cache %s: %s", self.cache_path, exc)

    def rate_limited_get(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_key = self._cache_key(method, params)
        with self._cache_lock:
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key]

        request_params = dict(params)
        request_params["method"] = method
        request_params["api_key"] = self.api_key
        request_params["format"] = "json"

        attempt = 0
        while True:
            attempt += 1
            with self._request_lock:
                now = time.monotonic()
                wait_for = self._last_request_ts + self.rate_interval - now
                if wait_for > 0:
                    time.sleep(wait_for)
                self._last_request_ts = time.monotonic()
            try:
                response = self.session.get(LASTFM_API_URL, params=request_params, timeout=20)
                self.api_calls += 1
            except requests.RequestException as exc:
                LOG.warning("Last.fm request error for %s: %s", method, exc)
                return None

            if response.status_code == 429:
                LOG.debug("Received 429 from Last.fm, backing off")
                time.sleep(self.rate_interval * 2)
                continue
            if response.status_code >= 500:
                if attempt >= 5:
                    LOG.warning("Last.fm server error %s for %s", response.status_code, method)
                    return None
                time.sleep(self.rate_interval)
                continue
            if response.status_code != 200:
                LOG.warning("Unexpected Last.fm status %s for %s", response.status_code, method)
                return None

            try:
                payload = response.json()
            except ValueError:
                LOG.warning("Invalid JSON response from Last.fm for %s", method)
                return None

            if "error" in payload:
                LOG.debug("Last.fm reported error %s for %s", payload.get("message", payload.get("error")), method)
                return None

            with self._cache_lock:
                self._cache[cache_key] = payload
                self._dirty = True
            return payload

    def lastfm_track_info(self, track: TrackMeta) -> Optional[TrackMatch]:
        candidate_artists: List[str] = []
        preferred = track.album_artist.strip()
        fallback = track.artist.strip()
        if preferred:
            candidate_artists.append(preferred)
        if fallback and fallback.lower() != preferred.lower():
            candidate_artists.append(fallback)
        if not candidate_artists or not track.title.strip():
            return None

        best_match: Optional[TrackMatch] = None
        seen_artists = set()
        for artist_name in candidate_artists:
            normalized_candidate = artist_name.casefold()
            if normalized_candidate in seen_artists:
                continue
            seen_artists.add(normalized_candidate)
            payload = self.rate_limited_get(
                "track.getInfo",
                {
                    "artist": artist_name,
                    "track": track.title,
                    "autocorrect": "1",
                },
            )
            if not payload:
                continue

            track_payload = payload.get("track")
            if not isinstance(track_payload, dict):
                continue

            remote_name = str(track_payload.get("name", "")).strip()
            remote_artist = ""
            artist_data = track_payload.get("artist")
            if isinstance(artist_data, dict):
                remote_artist = str(artist_data.get("name", "")).strip()

            playcount_raw = track_payload.get("playcount")
            try:
                playcount = int(playcount_raw)
            except (TypeError, ValueError):
                continue
            if playcount < 0:
                continue

            remote_title_norm = normalize_for_match(remote_name)
            remote_artist_norm = normalize_for_match(remote_artist)
            local_title_norm = normalize_for_match(track.title)
            local_artist_norm = normalize_for_match(artist_name)
            title_score = fuzz.ratio(local_title_norm, remote_title_norm)
            artist_score = fuzz.ratio(local_artist_norm, remote_artist_norm)
            match_score = min(title_score, artist_score)
            if match_score < MATCH_THRESHOLD:
                LOG.debug(
                    "Rejected Last.fm match for %s - %s (%s/%s)",
                    artist_name,
                    track.title,
                    artist_score,
                    title_score,
                )
                continue

            remote_duration = None
            if "duration" in track_payload:
                try:
                    duration_ms = int(track_payload["duration"])
                    if duration_ms > 0:
                        remote_duration = max(1, int(round(duration_ms / 1000)))
                except (TypeError, ValueError):
                    remote_duration = None

            candidate = TrackMatch(
                track=track,
                playcount=playcount,
                matched_name=remote_name or track.title,
                matched_artist=remote_artist or artist_name,
                match_score=match_score,
                artist_score=artist_score,
                title_score=title_score,
                remote_duration=remote_duration,
            )

            if best_match is None or candidate.playcount > best_match.playcount:
                best_match = candidate

        return best_match

    def _cache_key(self, method: str, params: Dict[str, Any]) -> str:
        items = tuple(sorted((k, str(v)) for k, v in params.items()))
        return json.dumps([method, items], separators=(",", ":"))


def configure_logging(settings: AppSettings, verbose: bool, debug: bool) -> None:
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    if verbose and not debug:
        level = logging.INFO
    if debug:
        level = logging.DEBUG
    if not verbose and not debug:
        level = getattr(logging, level_name, logging.INFO)

    handlers: List[logging.Handler] = []
    if HAVE_RICH:
        handlers.append(RichHandler(rich_tracebacks=False, markup=False))
    else:  # pragma: no cover - fallback path
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        handlers.append(handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

def iter_audio_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def extract_tags(path: Path) -> Optional[TrackMeta]:
    try:
        audio = MutagenFile(path)
    except (MutagenError, OSError) as exc:
        LOG.warning("Failed to read tags for %s: %s", path, exc)
        return None
    if audio is None:
        LOG.warning("Mutagen could not identify %s", path)
        return None

    tags = getattr(audio, "tags", {}) or {}

    def fetch_tag(*keys: str) -> Optional[str]:
        for key in keys:
            try:
                if hasattr(tags, "get"):
                    value = tags.get(key)
                else:
                    value = tags[key]
            except (KeyError, ValueError):
                continue
            if value is None:
                continue
            if isinstance(value, list):
                if not value:
                    continue
                candidate = value[0]
            else:
                candidate = value
            try:
                text = str(candidate)
            except Exception:
                continue
            text = text.strip()
            if text:
                return normalize_nfc(text)
        return None

    album = fetch_tag("album", "TALB", "©alb")
    album_artist = fetch_tag("albumartist", "TPE2", "aART")
    artist = fetch_tag("artist", "TPE1", "Author", "©ART")
    title = fetch_tag("title", "TIT2", "©nam") or normalize_nfc(path.stem)
    track_number_raw = fetch_tag("tracknumber", "TRCK", "trkn")
    disc_number_raw = fetch_tag("discnumber", "TPOS", "disk")
    track_number, track_total = parse_index_tag(track_number_raw)
    disc_number, disc_total = parse_index_tag(disc_number_raw)
    date = fetch_tag("date", "TDAT", "TDRC", "©day")
    original_date = fetch_tag("originaldate", "TDOR")
    year = extract_year(date, original_date)
    musicbrainz_track_id = fetch_tag("musicbrainz_trackid", "MusicBrainz Track Id")
    musicbrainz_release_id = fetch_tag("musicbrainz_albumid", "MusicBrainz Album Id")

    duration = None
    try:
        if hasattr(audio, "info") and getattr(audio.info, "length", None):
            duration = int(round(float(audio.info.length)))
    except Exception:
        duration = None

    return TrackMeta(
        path=path,
        album=album or "",
        album_artist=album_artist or "",
        artist=artist or "",
        title=title,
        track_number=track_number,
        track_total=track_total,
        disc_number=disc_number,
        disc_total=disc_total,
        date=date,
        original_date=original_date,
        year=year,
        musicbrainz_track_id=musicbrainz_track_id,
        musicbrainz_release_id=musicbrainz_release_id,
        duration=duration,
        normalized_album=normalize_simple(album or ""),
        normalized_album_artist=normalize_simple(album_artist or ""),
        normalized_artist=normalize_simple(artist or ""),
        normalized_title=normalize_simple(title),
    )


def parse_index_tag(raw: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not raw:
        return None, None
    value = raw.strip()
    if not value:
        return None, None
    parts = re.split(r"[\\/]", value)
    number = _safe_int(parts[0])
    total = _safe_int(parts[1]) if len(parts) > 1 else None
    return number, total


def _safe_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    cleaned = re.sub(r"\D", "", text)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def normalize_nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def normalize_simple(value: str) -> str:
    lowered = normalize_nfc(value).casefold()
    collapsed = " ".join(lowered.split())
    return collapsed


def normalize_for_match(value: Optional[str]) -> str:
    if not value:
        return ""
    text = normalize_nfc(value)
    text = text.replace("&", " and ")
    text = text.lower()
    text = BRACKETED_RE.sub(" ", text)
    text = VERSION_SUFFIX_RE.sub(" ", text)
    text = text.translate(PUNCT_TABLE)
    text = " ".join(word for word in text.split() if not YEAR_RE.fullmatch(word))
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return " ".join(text.split())


def extract_year(*values: Optional[str]) -> Optional[int]:
    for value in values:
        if not value:
            continue
        match = YEAR_RE.search(value)
        if not match:
            continue
        try:
            year = int(match.group())
        except ValueError:
            continue
        if 1000 <= year <= 3000:
            return year
    return None


def album_key(meta: TrackMeta) -> Optional[AlbumKey]:
    if not meta.album:
        return None
    normalized_album = meta.normalized_album
    if not normalized_album:
        return None
    if meta.normalized_album_artist:
        return AlbumKey(meta.normalized_album_artist, normalized_album)
    if meta.normalized_artist:
        return AlbumKey(meta.normalized_artist, normalized_album)
    return None


def album_group_key(track: TrackMeta, music_root: Path) -> Optional[str]:
    if not track.album:
        return None
    normalized_album = track.normalized_album
    if not normalized_album:
        return None
    base_key = album_key(track)
    if base_key and track.normalized_album_artist:
        return "::".join(["albumartist", base_key.normalized_album_artist, base_key.normalized_album_title])
    try:
        relative = track.path.relative_to(music_root)
        parts = relative.parts
    except ValueError:
        parts = track.path.parts
    if parts:
        root_key = "/".join(parts[:2]).casefold()
    else:
        root_key = track.path.parent.name.casefold()
    return "::".join(["fallback", normalized_album, root_key])


def resolve_album(tracks: List[TrackMeta]) -> Optional[tuple[AlbumKey, str, str, Optional[int]]]:
    if not tracks:
        return None
    album_titles = Counter(t.album for t in tracks if t.album)
    if not album_titles:
        return None
    album_title, _ = album_titles.most_common(1)[0]
    normalized_album = normalize_simple(album_title)

    album_artists = Counter(t.album_artist for t in tracks if t.album_artist)
    if album_artists:
        album_artist, _ = album_artists.most_common(1)[0]
        normalized_artist = normalize_simple(album_artist)
        return AlbumKey(normalized_artist, normalized_album), album_artist, album_title, extract_album_year(tracks)

    track_artists = Counter(t.artist for t in tracks if t.artist)
    if len(track_artists) == 1:
        artist = next(iter(track_artists))
        normalized_artist = normalize_simple(artist)
        return AlbumKey(normalized_artist, normalized_album), artist, album_title, extract_album_year(tracks)

    return None


def extract_album_year(tracks: Sequence[TrackMeta]) -> Optional[int]:
    years = sorted({t.year for t in tracks if t.year})
    if years:
        return years[0]
    return None


def is_compilation(tracks: Sequence[TrackMeta], album_artist: str) -> tuple[bool, Optional[str]]:
    if album_artist.lower() == "various artists":
        return True, "albumartist=Various Artists"
    artist_counts = Counter(normalize_simple(t.artist) for t in tracks if t.artist)
    distinct_artists = len(artist_counts)
    if distinct_artists > 3:
        total_tracks = sum(artist_counts.values())
        most_common = artist_counts.most_common(1)[0][1] if artist_counts else 0
        if total_tracks and (most_common / total_tracks) < 0.7:
            return True, "diverse track artists"
    return False, None


def choose_top_track_for_album(matches: List[TrackMatch]) -> Optional[TrackMatch]:
    if not matches:
        return None
    return max(matches, key=lambda m: (m.playcount, -(m.track.track_number or 0)))


def rewrite_path(path: Path, from_prefix: str, to_prefix: str) -> str:
    posix_path = path.as_posix()
    from_norm = Path(from_prefix).as_posix()
    if not from_norm.endswith("/"):
        from_norm += "/"
    to_norm = Path(to_prefix).as_posix()
    if not to_norm.endswith("/"):
        to_norm += "/"
    if posix_path.startswith(from_norm):
        suffix = posix_path[len(from_norm) :]
        rewritten = to_norm + suffix
        return rewritten
    return posix_path


def write_m3u(entries: List[PlaylistEntry], dest_path: Path, from_prefix: str, to_prefix: str) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(dest_path.parent)) as tmp:
        tmp.write("#EXTM3U\n")
        for entry in entries:
            track = entry.track
            duration = entry.remote_duration or track.duration or -1
            if duration < 0:
                duration = -1
            display_artist = track.artist or entry.album_artist
            tmp.write(f"#EXTINF:{duration},{display_artist} - {track.title}\n")
            rewritten = rewrite_path(track.path, from_prefix, to_prefix)
            tmp.write(f"{rewritten}\n")
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, dest_path)


def build_playlist(settings: AppSettings, dry_run: bool) -> int:
    start_time = time.perf_counter()
    all_tracks: List[TrackMeta] = []
    for path in iter_audio_files(settings.music_root):
        meta = extract_tags(path)
        if meta:
            all_tracks.append(meta)

    album_buckets: Dict[str, List[TrackMeta]] = defaultdict(list)
    for track in all_tracks:
        key = album_group_key(track, settings.music_root)
        if key:
            album_buckets[key].append(track)

    albums: List[AlbumData] = []
    total_albums = 0
    skipped_compilations = 0
    skipped_no_match = 0

    for bucket_tracks in album_buckets.values():
        resolved = resolve_album(bucket_tracks)
        if not resolved:
            continue
        album_key, album_artist, album_title, album_year = resolved
        total_albums += 1
        compilation, reason = is_compilation(bucket_tracks, album_artist)
        if compilation:
            LOG.info("Skipped compilation: %s - %s (%s)", album_artist, album_title, reason)
            skipped_compilations += 1
            continue
        albums.append(
            AlbumData(
                index=len(albums),
                key=album_key,
                album_artist=album_artist,
                album_title=album_title,
                year=album_year,
                tracks=bucket_tracks,
            )
        )

    if not albums:
        LOG.error("No eligible albums found.")
        return 1

    api_key = settings.lastfm_api_key
    if not api_key:
        LOG.error("Missing required LASTFM_API_KEY environment variable.")
        return 1

    client = LastFMClient(api_key, settings.lastfm_requests_per_sec, settings.lastfm_cache_file)

    album_matches: Dict[int, List[TrackMatch]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        future_map = {}
        for album in albums:
            for track in album.tracks:
                future = executor.submit(client.lastfm_track_info, track)
                future_map[future] = album.index
        for future in as_completed(future_map):
            album_index = future_map[future]
            try:
                match = future.result()
            except Exception as exc:
                LOG.warning("Error fetching Last.fm info: %s", exc)
                continue
            if match:
                album_matches[album_index].append(match)

    playlist_entries: List[PlaylistEntry] = []
    for album in albums:
        matches = album_matches.get(album.index, [])
        top_match = choose_top_track_for_album(matches)
        if not top_match:
            LOG.info("No confident Last.fm match: %s - %s", album.album_artist, album.album_title)
            skipped_no_match += 1
            continue
        playlist_entries.append(
            PlaylistEntry(
                album_artist=album.album_artist,
                album_title=album.album_title,
                normalized_album_artist=album.key.normalized_album_artist,
                normalized_album_title=album.key.normalized_album_title,
                year=album.year,
                track=top_match.track,
                playcount=top_match.playcount,
                matched_artist=top_match.matched_artist,
                matched_title=top_match.matched_name,
                remote_duration=top_match.remote_duration,
            )
        )

    playlist_entries.sort(key=playlist_sort_key)

    if playlist_entries:
        if dry_run:
            for entry in playlist_entries:
                LOG.info(
                    "[dry-run] %s - %s :: %s - %s (playcount=%d)",
                    entry.album_artist,
                    entry.album_title,
                    entry.track.artist,
                    entry.track.title,
                    entry.playcount,
                )
        else:
            write_m3u(
                playlist_entries,
                settings.playlist_path,
                settings.normalized_path_rewrite_from,
                settings.normalized_path_rewrite_to,
            )
            LOG.info("Wrote playlist to %s", settings.playlist_path)
    else:
        LOG.error("No tracks selected; cannot write playlist.")

    elapsed = time.perf_counter() - start_time
    LOG.info(
        "Summary: albums=%d, skipped_compilations=%d, skipped_unmatched=%d, tracks_written=%d, api_calls=%d, cache_hits=%d, elapsed=%.2fs",
        total_albums,
        skipped_compilations,
        skipped_no_match,
        len(playlist_entries),
        client.api_calls,
        client.cache_hits,
        elapsed,
    )

    client.save_cache()

    return 0 if playlist_entries else 1


def playlist_sort_key(entry: PlaylistEntry) -> tuple:
    artist_key = entry.normalized_album_artist or entry.track.normalized_artist
    if entry.year is not None:
        album_key = (0, entry.year, entry.normalized_album_title)
    else:
        album_key = (1, entry.normalized_album_title)
    track_number = entry.track.track_number or 0
    disc_number = entry.track.disc_number or 0
    return (artist_key, album_key, track_number, disc_number, entry.track.title.casefold())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a top-artist-tracks playlist using Last.fm global playcounts.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write the M3U file; just log selections.")
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings, verbose=args.verbose, debug=args.debug)
    exit_code = build_playlist(settings, dry_run=args.dry_run)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

# Required
# LASTFM_API_KEY=YOUR_KEY

# Defaults (can override)
# MUSIC_ROOT=/Volumes/NAS/Media/Music/Music_Server/
# PLAYLIST_DIR=/Volumes/NAS/Media/Music/Music_Server/_playlists/
# PLAYLIST_NAME=top_artist_tracks.m3u
# PATH_REWRITE_FROM=/Volumes/NAS/Media/Music/Music_Server/
# PATH_REWRITE_TO=/music/
# LASTFM_REQUESTS_PER_SEC=4
# MAX_WORKERS=8
# LASTFM_CACHE_FILE=.lastfm_cache.json

# requirements.txt
# mutagen
# python-dotenv
# rapidfuzz
# requests
# rich
