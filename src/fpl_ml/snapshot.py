"""Capture one point-in-time snapshot of the FPL API.

The distinction this module exists to hold onto: we parse ``bootstrap-static``
**for control flow** -- to learn which player IDs to fetch and which gameweek we
are in -- while still storing the original bytes untouched. Parsing to decide
what to do next is fine. Parsing to decide what to *store* is the thing we are
avoiding.

A capture never raises on a failed endpoint. It records the failure in the
manifest and carries on, because a partial snapshot is worth far more than no
snapshot: you cannot come back tomorrow and re-capture today.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from . import archive, config
from .client import FetchError, FplClient

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def gameweek_context(bootstrap_body: bytes | None) -> dict[str, object]:
    """Which gameweek was live, and when is the next deadline.

    Recorded in the manifest so that later, when building point-in-time
    features, you can tell where a snapshot sits relative to a deadline without
    re-parsing a megabyte of JSON.
    """
    empty: dict[str, object] = {
        "current_event": None,
        "next_event": None,
        "next_deadline": None,
    }
    if bootstrap_body is None:
        return empty

    try:
        events = json.loads(bootstrap_body).get("events", [])
    except (ValueError, AttributeError):
        return empty

    current = next((e for e in events if e.get("is_current")), None)
    following = next((e for e in events if e.get("is_next")), None)
    return {
        "current_event": current.get("id") if current else None,
        "next_event": following.get("id") if following else None,
        "next_deadline": following.get("deadline_time") if following else None,
    }


def player_ids(bootstrap_body: bytes) -> list[int]:
    """Every player ID in the current game, from the stored bootstrap payload."""
    elements = json.loads(bootstrap_body).get("elements", [])
    return [int(element["id"]) for element in elements]


async def _fetch_into(
    client: FplClient,
    url: str,
    run_dir: Path,
    relative_path: str,
    entries: list[dict[str, object]],
) -> bytes | None:
    """Fetch one URL, store it, and append an honest manifest entry."""
    try:
        response = await client.fetch(url)
    except FetchError as exc:
        entries.append(
            {"url": url, "status": exc.status, "path": None, "error": str(exc)}
        )
        return None

    stored = archive.write_payload(run_dir, relative_path, response.body)
    entries.append({"url": url, "status": response.status, **stored})
    return response.body


async def capture(
    client: FplClient,
    *,
    root: Path = config.RAW_ROOT,
    include_players: bool = False,
) -> Path:
    """Capture a snapshot into a new run directory and return its path."""
    started_at = archive.utc_now()
    run_dir = archive.new_run(root, started_at)
    entries: list[dict[str, object]] = []

    bootstrap = await _fetch_into(
        client, config.BOOTSTRAP_URL, run_dir, "bootstrap-static.json", entries
    )
    await _fetch_into(client, config.FIXTURES_URL, run_dir, "fixtures.json", entries)

    if include_players and bootstrap is not None:
        # The client's semaphore caps how many of these actually run at once,
        # so it is safe to hand the whole fan-out to the event loop at once.
        await asyncio.gather(
            *(
                _fetch_into(
                    client,
                    config.ELEMENT_SUMMARY_URL.format(element_id=element_id),
                    run_dir,
                    f"element-summary/{element_id}.json",
                    entries,
                )
                for element_id in player_ids(bootstrap)
            )
        )

    failed = [entry for entry in entries if entry.get("path") is None]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": started_at.isoformat(),
        "completed_at": archive.utc_now().isoformat(),
        "run_directory": run_dir.name,
        "include_players": include_players,
        "gameweek": gameweek_context(bootstrap),
        "counts": {"ok": len(entries) - len(failed), "failed": len(failed)},
        # Concurrent fan-out finishes in arbitrary order; sort so that two
        # captures of the same thing produce comparable manifests.
        "entries": sorted(entries, key=lambda entry: str(entry["url"])),
    }
    (run_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    return run_dir


async def run(
    *,
    root: Path = config.RAW_ROOT,
    include_players: bool = False,
) -> Path:
    """Capture a snapshot using a freshly created client."""
    async with FplClient() as client:
        return await capture(client, root=root, include_players=include_players)


def read_manifest(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / MANIFEST_NAME).read_text())


def is_usable(manifest: dict[str, object]) -> bool:
    """Did the capture get the one payload everything else depends on?

    Not all failures are equal. Losing a single player summary out of 630 is a
    blemish -- the data is still there next week, and bootstrap already carries
    most of what that player's row needs. Losing ``bootstrap-static`` means the
    capture recorded nothing about who existed, what they cost, or which
    gameweek it was, and there is no way to reconstruct it.

    Callers use this to decide whether a capture is worth alerting about, so
    that a transient blip does not train you to ignore red builds.
    """
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if entry.get("url") == config.BOOTSTRAP_URL:
            return entry.get("path") is not None
    return False
