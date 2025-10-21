#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
csv_to_m3u.py
Reads a CSV with headers: artist, album, title
Builds a fuzzy-matched M3U8 playlist from your local music library.
Configuration is sourced from environment variables (see .env).

Requires:
  pip install rapidfuzz mutagen
"""

from __future__ import annotations
import csv
import sys
import unicodedata
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import Settings, get_settings

# =========================
# CONFIGURATION
# =========================

SETTINGS: Settings = get_settings()

# Root of your music files (where audio files live)
MUSIC_ROOT = SETTINGS.music_root

# Where to save the generated .m3u8 playlists
PLAYLIST_DIR = SETTINGS.playlist_dir

# Path rewriting (what to replace in file paths when writing the M3U)
PATH_REWRITE_FROM = SETTINGS.normalized_path_rewrite_from
PATH_REWRITE_TO = SETTINGS.normalized_path_rewrite_to

# CSV filename (in same folder as this script). If blank, auto-pick newest .csv next to the script
CSV_BASENAME = SETTINGS.csv_input_filename

# Matching options
FUZZ_THRESHOLD = SETTINGS.fuzzy_match_threshold  # accept if weighted score >= this
TITLE_WEIGHT = SETTINGS.fuzzy_title_weight
ARTIST_WEIGHT = SETTINGS.fuzzy_artist_weight
ALBUM_WEIGHT = SETTINGS.fuzzy_album_weight

# File types we’ll index
AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".alac", ".aac", ".ogg", ".opus", ".wav", ".aiff", ".aif"}

# Indexing performance: set to True to skip tag read for WAV/AIFF (often no tags)
SKIP_TAGS_FOR_RAW_PCM = SETTINGS.skip_raw_pcm_tags

# =========================
# Imports that may fail if deps missing
# =========================
try:
    from mutagen import File as MutagenFile
    from rapidfuzz import fuzz
except Exception as e:
    print("Missing dependencies. Please run:\n  python3 -m pip install rapidfuzz mutagen", file=sys.stderr)
    raise

# =========================
# Helpers
# =========================

def strip_diacritics(s: str) -> str:
    if not s:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    # casefold + strip diacritics + collapse spaces/punct like common “feat.” noise
    s = s or ""
    s = s.replace("&", " and ")
    s = s.replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = s.lower().strip()
    s = strip_diacritics(s)
    # remove featuring markers and content after (feat ...), ft., etc. for robust title/artist comp
    for token in ["(feat", "feat.", "feat ", " ft.", " ft "]:
        ix = s.find(token)
        if ix != -1:
            s = s[:ix].strip()
    # collapse multiple spaces
    s = " ".join(s.split())
    return s

def safe_get_tag(mutagen_audio, keys: List[str]) -> Optional[str]:
    if not mutagen_audio or not mutagen_audio.tags:
        return None
    for k in keys:
        v = mutagen_audio.tags.get(k)
        if not v:
            continue
        # Mutagen fields can be lists or text
        if isinstance(v, list):
            if v:
                return str(v[0])
        else:
            try:
                return str(v)
            except Exception:
                pass
    return None

def read_basic_tags(p: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """
    Return (artist, album, title, duration_secs) from tags, or (None, ..) if not available.
    """
    try:
        if SKIP_TAGS_FOR_RAW_PCM and p.suffix.lower() in {".wav", ".aiff", ".aif"}:
            return None, None, None, None
        m = MutagenFile(str(p))
        if not m:
            return None, None, None, None
        # Common tag keys across formats
        artist = safe_get_tag(m, ["artist", "TPE1", "Author", "©ART"])
        album  = safe_get_tag(m, ["album", "TALB", "©alb"])
        title  = safe_get_tag(m, ["title", "TIT2", "©nam"])
        duration = None
        if hasattr(m, "info") and getattr(m.info, "length", None):
            duration = int(round(m.info.length))
        return artist, album, title, duration
    except Exception:
        return None, None, None, None

def human_time(secs: Optional[int]) -> str:
    if not secs or secs <= 0:
        return "0"
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}"

# =========================
# Index music library
# =========================

class Track:
    __slots__ = ("path", "artist", "album", "title", "n_artist", "n_album", "n_title", "duration")
    def __init__(self, path: Path, artist: Optional[str], album: Optional[str],
                 title: Optional[str], duration: Optional[int]):
        self.path = path
        self.artist = artist or ""
        self.album = album or ""
        self.title = title or path.stem  # fallback to filename
        self.n_artist = normalize_text(self.artist)
        self.n_album  = normalize_text(self.album)
        self.n_title  = normalize_text(self.title)
        self.duration = duration or 0

def build_index(root: Path) -> List[Track]:
    if not root.exists():
        print(f"[ERROR] MUSIC_ROOT does not exist: {root}", file=sys.stderr)
        sys.exit(2)
    print(f"[INFO] Indexing music under: {root}")
    tracks: List[Track] = []
    t0 = time.time()
    count = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        count += 1
        if count % 1000 == 0:
            print(f"  …scanned {count} files")
        artist, album, title, duration = read_basic_tags(p)
        tracks.append(Track(p, artist, album, title, duration))
    dt = time.time() - t0
    print(f"[INFO] Indexed {len(tracks)} audio files in {dt:.1f}s")
    return tracks

# Build a quick title -> candidates map to narrow fuzzy comparisons
def title_buckets(tracks: List[Track]) -> Dict[str, List[Track]]:
    buckets: Dict[str, List[Track]] = {}
    for t in tracks:
        buckets.setdefault(t.n_title, []).append(t)
    return buckets

# =========================
# Fuzzy match
# =========================

def weighted_score(q_title: str, q_artist: str, q_album: str, cand: Track) -> float:
    # Strongest weight on title
    title_score  = fuzz.token_set_ratio(q_title,  cand.n_title)
    artist_score = fuzz.token_set_ratio(q_artist, cand.n_artist) if q_artist else 0
    album_score  = fuzz.token_set_ratio(q_album,  cand.n_album)  if q_album else 0
    score = (TITLE_WEIGHT * title_score) + (ARTIST_WEIGHT * artist_score) + (ALBUM_WEIGHT * album_score)
    return score

def best_match(tracks: List[Track],
               buckets: Dict[str, List[Track]],
               q_artist: str, q_album: str, q_title: str) -> Optional[Tuple[Track, float]]:
    """
    Return best (track, score) or None.
    """
    n_title  = normalize_text(q_title)
    n_artist = normalize_text(q_artist)
    n_album  = normalize_text(q_album)

    candidates: List[Track] = []

    # 1) exact normalized title bucket
    if n_title in buckets:
        candidates.extend(buckets[n_title])

    # 2) broaden: scan all tracks sharing first 10 normalized title chars (cheap prefix filter)
    if not candidates and len(n_title) >= 4:
        prefix = n_title[:10]
        for t in tracks:
            if t.n_title.startswith(prefix):
                candidates.append(t)

    # 3) fallback: all tracks (rarely)
    if not candidates:
        candidates = tracks

    best: Optional[Tuple[Track, float]] = None
    for t in candidates:
        s = weighted_score(n_title, n_artist, n_album, t)
        if not best or s > best[1]:
            best = (t, s)
    if best and best[1] >= FUZZ_THRESHOLD:
        return best
    return None

# =========================
# CSV handling & playlist
# =========================

def find_csv(script_dir: Path) -> Path:
    if CSV_BASENAME:
        p = (script_dir / CSV_BASENAME)
        if not p.exists():
            print(f"[ERROR] CSV_BASENAME set to '{CSV_BASENAME}', but file not found next to script.", file=sys.stderr)
            sys.exit(2)
        return p

    candidates = sorted((script_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print(f"[ERROR] No CSV found next to script. Place your CSV here or set CSV_BASENAME.", file=sys.stderr)
        sys.exit(2)
    if len(candidates) > 1:
        print(f"[INFO] Multiple CSVs found; using newest: {candidates[0].name}")
    return candidates[0]

def csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"artist", "album", "title"}
        header = {h.strip().lower() for h in reader.fieldnames or []}
        missing = expected - header
        if missing:
            print(f"[ERROR] CSV missing required headers: {', '.join(sorted(missing))}", file=sys.stderr)
            sys.exit(2)
        for row in reader:
            rows.append({
                "artist": (row.get("artist") or "").strip(),
                "album":  (row.get("album")  or "").strip(),
                "title":  (row.get("title")  or "").strip(),
            })
    print(f"[INFO] Loaded {len(rows)} rows from {csv_path.name}")
    return rows

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def rewrite_path(original: Path) -> str:
    # normalize separators; operate on POSIX form
    posix = original.as_posix()
    if PATH_REWRITE_FROM and posix.startswith(PATH_REWRITE_FROM):
        suffix = posix[len(PATH_REWRITE_FROM) :]
        return SETTINGS.normalized_path_rewrite_to + suffix.lstrip("/")
    # fallback: attempt to replace MUSIC_ROOT if PATH_REWRITE_FROM differs
    library_prefix = SETTINGS.music_root.as_posix()
    if not library_prefix.endswith("/"):
        library_prefix += "/"
    if posix.startswith(library_prefix):
        suffix = posix[len(library_prefix) :]
        return SETTINGS.normalized_path_rewrite_to + suffix.lstrip("/")
    # final fallback: leave path untouched
    return posix

def write_m3u(out_path: Path, tracks_with_meta: List[Track]):
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#EXTM3U\n")
        for t in tracks_with_meta:
            # EXTINF: duration,display title
            display_title = t.title or t.path.stem
            display_artist = t.artist or ""
            name = f"{display_artist} - {display_title}".strip(" -")
            f.write(f"#EXTINF:{t.duration if t.duration else -1},{name}\n")
            f.write(rewrite_path(t.path) + "\n")
    print(f"[OK] Wrote playlist: {out_path}")

# =========================
# Main
# =========================

def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = find_csv(script_dir)

    # Derive playlist name from CSV
    m3u_name = csv_path.stem + ".m3u8"
    out_path = PLAYLIST_DIR / m3u_name

    # Build index once
    tracks = build_index(MUSIC_ROOT)
    if not tracks:
        print("[ERROR] No audio files found to index.", file=sys.stderr)
        sys.exit(2)
    buckets = title_buckets(tracks)

    # Load rows and match
    rows = csv_rows(csv_path)

    matched: List[Track] = []
    misses: List[Tuple[int, Dict[str, str]]] = []

    for i, row in enumerate(rows, start=1):
        q_artist = row["artist"]
        q_album  = row["album"]
        q_title  = row["title"]

        if not q_title:
            print(f"[WARN] Row {i}: missing title, skipping")
            misses.append((i, row))
            continue

        result = best_match(tracks, buckets, q_artist, q_album, q_title)
        if result:
            t, score = result
            print(f"[MATCH {i:04d}] {q_artist} – {q_title}  ->  {t.artist} – {t.title}  [{score:.1f}]  ({t.path.name})")
            matched.append(t)
        else:
            print(f"[MISS  {i:04d}] {q_artist} – {q_title}")
            misses.append((i, row))

    if matched:
        write_m3u(out_path, matched)
    else:
        print("[ERROR] No matches found; nothing written.", file=sys.stderr)

    if misses:
        miss_log = PLAYLIST_DIR / (csv_path.stem + "_unmatched.tsv")
        ensure_dir(PLAYLIST_DIR)
        with miss_log.open("w", encoding="utf-8") as f:
            f.write("row\tartist\talbum\ttitle\n")
            for idx, r in misses:
                f.write(f"{idx}\t{r.get('artist','')}\t{r.get('album','')}\t{r.get('title','')}\n")
        print(f"[INFO] Wrote list of {len(misses)} unmatched rows: {miss_log}")

if __name__ == "__main__":
    main()
