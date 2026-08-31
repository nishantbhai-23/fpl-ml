"""Historical backfill from the community FPL archive.

Your own capture archive starts the day you first run it. Training needs
seasons. This module vendors gameweek-level history from
`vaastav/Fantasy-Premier-League <https://github.com/vaastav/Fantasy-Premier-League>`_
(MIT licensed), pinned to an exact upstream commit so the backfill is
reproducible and cannot shift underneath you.

**This data is not point-in-time**, and that is the whole reason it lives in its
own namespace. It is assembled after each gameweek and carries no availability
information at all — no ``status``, no ``news``, no ``chance_of_playing``. Its
pre-deadline columns were snapshotted whenever the upstream collector ran, with
no recorded as-of time.

So it is kept apart from ``raw/``, which holds captures whose timing we can
verify ourselves. Never merge the two without carrying a provenance column: a
dataset is only as trustworthy as its weakest source, and once mixed you can no
longer tell which rows carry point-in-time guarantees and which do not.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from . import archive
from .client import FetchError, FplClient

UPSTREAM_REPO = "vaastav/Fantasy-Premier-League"
UPSTREAM_LICENSE = "MIT"

# Pinned deliberately. A moving target would mean today's backtest and next
# month's are run against quietly different history.
UPSTREAM_SHA = "9779cdbc0c07f6c900c2d0c181ddf6bb9c800f88"

RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_SHA}"

SEASONS: tuple[str, ...] = (
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
)

# Not every season has every file — the two earliest predate fixtures.csv and
# teams.csv upstream. The fetch records what it actually found rather than
# assuming, so a missing file is data about the season, not a crash.
SEASON_FILES: tuple[str, ...] = (
    "gws/merged_gw.csv",
    "fixtures.csv",
    "teams.csv",
    "players_raw.csv",
)

# Context that changes what a season's numbers mean. Kept as metadata rather
# than used to drop seasons at ingest: whether an anomalous season helps or
# hurts is an experiment worth running, not an assumption worth baking in.
SEASON_NOTES: dict[str, str] = {
    "2019-20": "COVID: suspended March-June, restarted behind closed doors",
    "2020-21": "COVID: behind closed doors throughout, condensed calendar",
    "2022-23": "World Cup mid-season break, November-December",
    "2025-26": "Defensive-contribution scoring introduced",
    "2026-27": "Defensive-contribution scoring; season in progress",
}

MANIFEST_NAME = "manifest.json"


API_BASE = f"https://api.github.com/repos/{UPSTREAM_REPO}/contents"


def file_url(season: str, relative: str) -> str:
    return f"{RAW_BASE}/data/{season}/{relative}"


def api_url(season: str, relative: str) -> str:
    return f"{API_BASE}/data/{season}/{relative}?ref={UPSTREAM_SHA}"


def _api_headers() -> dict[str, str]:
    """Ask the API for raw bytes rather than a base64-wrapped JSON envelope."""
    headers = {"Accept": "application/vnd.github.raw"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        # Optional. Unauthenticated is 60 requests/hour, and the fallback is
        # only reached for the handful of files raw.githubusercontent refuses.
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _fetch_file(
    client: FplClient, season: str, relative: str
) -> tuple[bytes, int, str]:
    """Fetch one upstream file, returning ``(body, status, route)``.

    Two routes, because ``raw.githubusercontent.com`` returns a persistent 400
    for a few specific paths in this repo that are perfectly ordinary blobs and
    that the API serves without complaint. It is a CDN quirk rather than
    anything wrong with the files, so we fall back rather than lose the season.
    The route is recorded per file so the oddity stays visible instead of being
    silently papered over.
    """
    try:
        response = await client.fetch(file_url(season, relative))
    except FetchError as raw_error:
        try:
            response = await client.fetch(
                api_url(season, relative), headers=_api_headers()
            )
        except FetchError:
            raise raw_error from None
        return response.body, response.status, "api"
    return response.body, response.status, "raw"


async def fetch(
    dest: Path,
    *,
    seasons: tuple[str, ...] = SEASONS,
    client: FplClient | None = None,
) -> dict[str, object]:
    """Vendor the pinned upstream CSVs into ``dest`` and write a manifest.

    A missing upstream file is recorded and skipped rather than raising: the
    early seasons genuinely lack some of these, and that absence is itself
    something the normaliser needs to know.
    """
    owns_client = client is None
    client = client or FplClient()
    started_at = archive.utc_now()
    entries: list[dict[str, object]] = []

    try:
        for season in seasons:
            for relative in SEASON_FILES:
                url = file_url(season, relative)
                try:
                    body, status, route = await _fetch_file(client, season, relative)
                except FetchError as exc:
                    entries.append(
                        {
                            "season": season,
                            "file": relative,
                            "url": url,
                            "status": exc.status,
                            "path": None,
                            "error": str(exc),
                        }
                    )
                    continue

                destination = dest / season / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(body)
                entries.append(
                    {
                        "season": season,
                        "file": relative,
                        "url": url,
                        "status": status,
                        "route": route,
                        "path": str(destination.relative_to(dest)),
                        "bytes": len(body),
                        "sha256": archive.sha256_hex(body),
                    }
                )
    finally:
        if owns_client:
            await client.aclose()

    missing = [e for e in entries if e.get("path") is None]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "fetched_at": started_at.isoformat(),
        "upstream": {
            "repo": UPSTREAM_REPO,
            "sha": UPSTREAM_SHA,
            "license": UPSTREAM_LICENSE,
            "url": f"https://github.com/{UPSTREAM_REPO}",
        },
        "point_in_time": False,
        "note": (
            "Post-hoc reconstruction, not a point-in-time capture. No player "
            "availability data. Do not merge with raw/ without a provenance "
            "column."
        ),
        "season_notes": {s: SEASON_NOTES[s] for s in seasons if s in SEASON_NOTES},
        "counts": {"ok": len(entries) - len(missing), "missing": len(missing)},
        "entries": entries,
    }

    dest.mkdir(parents=True, exist_ok=True)
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(dest: Path, *, seasons: tuple[str, ...] = SEASONS) -> dict[str, object]:
    return asyncio.run(fetch(dest, seasons=seasons))


def read_manifest(dest: Path) -> dict[str, object]:
    return json.loads((dest / MANIFEST_NAME).read_text())
