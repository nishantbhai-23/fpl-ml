"""The prediction log: what we said, before we could know.

This is the smallest component in the project and the most important one. A
prediction written down before a deadline, never edited afterwards, is the only
thing that makes any claim about this system checkable. Everything else -- the
backtest, the metrics, the models -- can be fooled by a mistake nobody notices.
A committed prediction cannot.

The log is append-only for the same reason the capture archive is. A prediction
you can edit after the results arrive is not evidence of anything.

Layout::

    predictions/2026-27/gw03/
      manifest.json     when, from which capture, at which code version
      <baseline>.csv    one row per player, with the rank

Every entry records the capture it was built from and the git commit of the
code that built it, so any prediction can be reproduced exactly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl

from . import archive

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def code_version() -> str | None:
    """The git commit that produced a prediction, when one is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def gameweek_directory(root: Path, season: str, gameweek: int) -> Path:
    return root / season / f"gw{int(gameweek):02d}"


def write(
    root: Path,
    *,
    season: str,
    gameweek: int,
    deadline: str | None,
    capture: str,
    predictions: dict[str, pl.DataFrame],
    context: dict[str, object] | None = None,
) -> Path:
    """Write one gameweek's predictions. Refuses to overwrite an existing set.

    ``predictions`` maps a model name to a frame holding at least ``code``,
    ``name``, ``position`` and ``prediction``.
    """
    target = gameweek_directory(root, season, gameweek)
    target.mkdir(parents=True, exist_ok=True)

    # What makes a prediction evidence is that it was fixed before the answer
    # existed -- not that its folder was empty. So the guard is the deadline,
    # and adding another model before that deadline is legitimate. Rewriting a
    # model already on disk never is.
    if deadline is not None and archive.utc_now().isoformat() > deadline:
        raise ValueError(
            f"the deadline for {season} gw{gameweek} ({deadline}) has passed; "
            "writing a prediction now would not be a prediction."
        )

    existing = read_manifest(target).get("models", []) if (target / MANIFEST_NAME).exists() else []
    already = {entry["model"] for entry in existing}
    clashes = already & set(predictions)
    if clashes:
        raise FileExistsError(
            f"{', '.join(sorted(clashes))} already logged for {season} gw{gameweek}. "
            "A prediction that can be rewritten proves nothing."
        )

    entries = list(existing)
    for name, frame in predictions.items():
        ordered = frame.sort("prediction", descending=True).with_columns(
            pl.int_range(1, frame.height + 1).alias("rank")
        )
        columns = [
            c
            for c in ("code", "name", "position", "team", "value", "prediction", "rank")
            if c in ordered.columns
        ]
        body = ordered.select(columns).write_csv().encode("utf-8")
        (target / f"{name}.csv").write_bytes(body)
        entries.append(
            {
                "model": name,
                "file": f"{name}.csv",
                "rows": ordered.height,
                "sha256": archive.sha256_hex(body),
                # Per model, because models can be added at different times
                # before the same deadline.
                "made_at": archive.utc_now().isoformat(),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "gameweek": int(gameweek),
        "deadline": deadline,
        # Each model carries its own timestamp; compare against `deadline`.
        "first_written_at": (
            existing[0]["made_at"] if existing and "made_at" in existing[0]
            else archive.utc_now().isoformat()
        ),
        "capture": capture,
        "code_version": code_version(),
        "models": entries,
        "context": context or {},
    }
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def read_manifest(directory: Path) -> dict[str, object]:
    return json.loads((directory / MANIFEST_NAME).read_text())


def list_logged(root: Path) -> list[Path]:
    """Every gameweek directory that holds a prediction, oldest first."""
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/gw*") if (p / MANIFEST_NAME).exists())
