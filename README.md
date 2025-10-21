# Playlist Generation Scripts

Utilities for building and maintaining playlists from a local music library. All scripts target Python 3.11+ and share a unified configuration layer via `config.py`.

## Setup
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `python -m pip install -r requirements.txt`

Copy the supplied template to create your local overrides:

```bash
cp .env.example .env
```

## Configuration
Environment variables are loaded from `.env` (project root) plus any script-local `.env` alongside individual tools. Defaults cover the common NAS layout used by the repository author.

| Variable | Default | Description |
| --- | --- | --- |
| `MUSIC_ROOT` | `/Volumes/NAS/Media/Music/Music_Server/` | Root directory containing the music library. |
| `PLAYLIST_DIR` | `/Volumes/NAS/Media/Music/Music_Server/_playlists/` | Destination folder for generated playlists. |
| `PLAYLIST_NAME` | `top_artist_tracks.m3u` | Output filename for `build_top_artist_tracks.py`. |
| `NONFLAC_PLAYLIST_PATH` | `/Volumes/.../_playlists/Look For Replacment.m3u` | Destination playlist for `nonFLACplaylist.py`. |
| `PATH_REWRITE_FROM` | `/Volumes/NAS/Media/Music/Music_Server/` | Absolute prefix to replace when writing playlist entries. |
| `PATH_REWRITE_TO` | `/music/` | Replacement prefix; ensures playlists work inside containers or remote players. |
| `LASTFM_API_KEY` | _required_ | Last.fm API key used by `build_top_artist_tracks.py`. |
| `LASTFM_REQUESTS_PER_SEC` | `4` | Rate limit for Last.fm API calls. |
| `MAX_WORKERS` | `8` | Thread pool size for concurrent Last.fm lookups. |
| `LASTFM_CACHE_FILE` | `.lastfm_cache.json` | JSON cache (relative paths resolve against the project root). |
| `CSV_INPUT_FILENAME` | `Liked_Tracks_clean.csv` | Default CSV file for `csv_to_m3u.py`. |
| `FUZZY_MATCH_THRESHOLD` | `86` | Minimum RapidFuzz score to accept a match. |
| `FUZZY_TITLE_WEIGHT` | `0.60` | Weight given to the title score. |
| `FUZZY_ARTIST_WEIGHT` | `0.30` | Weight given to the artist score. |
| `FUZZY_ALBUM_WEIGHT` | `0.10` | Weight given to the album score. |
| `SKIP_RAW_PCM_TAGS` | `true` | Skip tag reads for WAV/AIFF during indexing. |
| `LOG_LEVEL` | `INFO` | Baseline log level (CLI flags still override at runtime). |

> Legacy variable names remain supported for now; when a fallback is used a `DeprecationWarning` is emitted so you can update your `.env`.

Refer to `.env.example` for a documented template.

## Usage

### build_top_artist_tracks.py
Pulls top-played tracks for each on-disk album using Last.fm global play counts and produces `PLAYLIST_DIR/PLAYLIST_NAME`. The script honours `--dry-run`, `--verbose`, and `--debug` flags.

```bash
python build_top_artist_tracks.py --dry-run
python build_top_artist_tracks.py
```

You must supply `LASTFM_API_KEY`. Results are cached in `LASTFM_CACHE_FILE` to minimise API calls.

### csv_to_m3u.py
Builds an `.m3u8` playlist by fuzzy matching a CSV of desired tracks against your library.

```bash
python csv_to_m3u.py
```

The script looks for `CSV_INPUT_FILENAME` in the same folder (falls back to the newest CSV if not specified) and writes the playlist to `PLAYLIST_DIR`.

### nonFLACplaylist.py
Enumerates every non-FLAC file under `MUSIC_ROOT` and writes them to `NONFLAC_PLAYLIST_PATH`, rewriting the prefix to `PATH_REWRITE_TO`.

```bash
python nonFLACplaylist.py
```

## Path Rewrite Behaviour
`PATH_REWRITE_FROM`/`PATH_REWRITE_TO` are applied consistently across scripts so generated playlists reference container-friendly paths (e.g., `/music/...`). Ensure `PATH_REWRITE_FROM` exactly matches the on-disk prefix you need to replace.

## Migration
Environment variables have been consolidated. The following legacy names still work but should be renamed in `.env`:

| Legacy | Canonical |
| --- | --- |
| `PLAYLIST_PATH` | `NONFLAC_PLAYLIST_PATH` |
| `PLAYLIST_PREFIX` | `PATH_REWRITE_TO` |
| `CSV_BASENAME` | `CSV_INPUT_FILENAME` |
| `FUZZ_THRESHOLD` | `FUZZY_MATCH_THRESHOLD` |
| `TITLE_WEIGHT` | `FUZZY_TITLE_WEIGHT` |
| `ARTIST_WEIGHT` | `FUZZY_ARTIST_WEIGHT` |
| `ALBUM_WEIGHT` | `FUZZY_ALBUM_WEIGHT` |
| `SKIP_TAGS_FOR_RAW_PCM` | `SKIP_RAW_PCM_TAGS` |
| `CACHE_FILE` | `LASTFM_CACHE_FILE` |
| `LASTFM_RATE_LIMIT_PER_SEC` | `LASTFM_REQUESTS_PER_SEC` |
| `TOP_ARTIST_PLAYLIST_NAME` | `PLAYLIST_NAME` |

No scripts write to `.env`; update values manually and re-run as needed.
