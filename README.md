# Playlist Generation Scripts

Tools for building and maintaining playlists from a local music library.

## Prerequisites

- Python 3.10 or newer
- Python packages: `rapidfuzz`, `mutagen`

Install dependencies with:

```bash
python3 -m pip install rapidfuzz mutagen
```

## Configuration

Copy `.env.example` to `.env` and update the values to match your environment:

```bash
cp .env.example .env
```

Key settings:

- `MUSIC_ROOT` – root folder that contains your audio files.
- `PLAYLIST_DIR` – directory where generated M3U playlists are written.
- `PATH_REWRITE_FROM` / `PATH_REWRITE_TO` – adjust path prefixes when writing playlist entries.
- `CSV_BASENAME` – default CSV name for the CSV to M3U workflow.
- `FUZZ_THRESHOLD`, `TITLE_WEIGHT`, `ARTIST_WEIGHT`, `ALBUM_WEIGHT` – tune fuzzy match behaviour.
- `SKIP_TAGS_FOR_RAW_PCM` – skip reading tags for raw PCM formats (useful for faster scans).
- `PLAYLIST_PATH`, `PLAYLIST_PREFIX` – output file and prefix for the non-FLAC playlist script.

The helper in `config.py` loads `.env` automatically for every script and falls back to the defaults shown in `.env.example` when values are missing.

## Scripts

### csv_to_m3u.py

Creates an `.m3u8` playlist from a CSV file containing track metadata (artist, album, title). The script scans `MUSIC_ROOT`, builds a fuzzy-match index, and writes the best matches to `PLAYLIST_DIR`. Unmatched rows are logged to a TSV file in the same directory.

Run it with:

```bash
python3 csv_to_m3u.py
```

### nonFLACplaylist.py

Generates a playlist containing every non-FLAC audio file under `MUSIC_ROOT`. The playlist uses `PLAYLIST_PREFIX` so paths appear as they will on the playback system.

Run it with:

```bash
python3 nonFLACplaylist.py
```

---

As you add more playlist-related scripts, reuse `config.py` for shared settings and document new workflows in this README so the setup remains consistent.
