"""Normalise vendored seasons into one tidy panel table.

One row per player per gameweek, eleven seasons wide. This is the table the
walk-forward harness will train on, so two properties matter more than
convenience:

**Every column is classified.** Ingest fails loudly on a column nobody has
placed in :mod:`fpl_ml.schema`. FPL adds columns when it changes the game — the
defensive-contribution stats arrived exactly that way in 2025-26 — and an
unclassified column flowing into a feature matrix is how leakage gets in.

**Nothing is dropped for being awkward.** Anomalous seasons are labelled, not
excluded. Whether the COVID seasons help or hurt is an experiment to run later,
and you cannot run it against data you threw away at ingest.

A caution the table cannot express on its own: ``element`` is FPL's player ID
*within a season*. It is reassigned between seasons, so it is not a
cross-season key. Linking a player across seasons means matching on name, which
is genuinely unreliable — surnames collide, spellings change, and accents come
and go. Phase 02 will need to solve that properly; until then, treat
``(season, element)`` as the only trustworthy player key.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from . import backfill, schema

PANEL_NAME = "panel.parquet"
SUMMARY_NAME = "panel_summary.json"

GAMEWEEK_FILE = "gws/merged_gw.csv"

# Encoding actually used per season, recorded during the read for the summary.
_ENCODINGS: dict[str, str] = {}


def _decode(raw: bytes) -> tuple[str, str]:
    """Decode a vendored CSV, returning ``(text, encoding)``.

    The three earliest seasons are latin-1; everything from 2019-20 on is
    UTF-8. Detected rather than hardcoded, because the boundary is an accident
    of how the upstream collector changed over the years and could move again.

    Latin-1 decodes any byte sequence without error, so it is a safe last
    resort — though if a file were really cp1252 it would quietly mangle smart
    quotes. Accented player names, the reason any of this matters, live in a
    range where the two agree.
    """
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def _read_season(source: Path, season: str) -> pl.DataFrame | None:
    """Read one season's gameweek CSV, or None if it was not vendored."""
    path = source / season / GAMEWEEK_FILE
    if not path.exists():
        return None

    text, encoding = _decode(path.read_bytes())

    # Every column as string first: dtypes drift across seasons (integers that
    # became floats, booleans written as True/False vs 1/0), and inferring
    # per-season then reconciling is far more painful than casting once, later,
    # under our own control.
    frame = pl.read_csv(
        text.encode("utf-8"),
        infer_schema_length=0,
        truncate_ragged_lines=True,
        ignore_errors=True,
    )
    _ENCODINGS[season] = encoding
    frame.columns = [c.strip() for c in frame.columns]

    # Fail on anything unclassified before it can reach a feature matrix.
    schema.check_all_known(frame.columns)

    return frame.with_columns(pl.lit(season).alias("season"))


def build(
    source: Path,
    *,
    seasons: tuple[str, ...] = backfill.SEASONS,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Read every vendored season and stack them into one panel."""
    frames: list[pl.DataFrame] = []
    per_season: dict[str, object] = {}

    for season in seasons:
        frame = _read_season(source, season)
        if frame is None:
            per_season[season] = {"rows": 0, "columns": 0, "status": "not vendored"}
            continue
        frames.append(frame)
        per_season[season] = {
            "rows": frame.height,
            "columns": frame.width - 1,  # excluding the season column we added
            "gameweeks": frame["round"].n_unique() if "round" in frame.columns else None,
            "encoding": _ENCODINGS.get(season),
            "note": backfill.SEASON_NOTES.get(season),
        }

    if not frames:
        raise FileNotFoundError(f"no vendored seasons found under {source}")

    # Diagonal: seasons genuinely have different columns, and a column absent
    # from a season must land as null rather than silently aligning to the
    # wrong field.
    panel = pl.concat(frames, how="diagonal_relaxed")

    present = set(panel.columns)
    summary: dict[str, object] = {
        "rows": panel.height,
        "columns": panel.width,
        "seasons": per_season,
        "provenance": "backfill",
        "point_in_time": False,
        "upstream_sha": backfill.UPSTREAM_SHA,
        "column_classes": {
            "identity": sorted(present & schema.IDENTITY),
            "pre_deadline": sorted(present & schema.PRE_DEADLINE),
            "outcome": sorted(present & schema.OUTCOME),
        },
        "coverage": _coverage(panel),
    }
    return panel, summary


def _coverage(panel: pl.DataFrame) -> dict[str, list[str]]:
    """Which seasons each column actually has data for.

    Useful precisely because the answer is uncomfortable: expected-goals columns
    only exist from a certain season onward, and the defensive-action stats
    exist at both ends of the range but not in the middle.
    """
    coverage: dict[str, list[str]] = {}
    for column in panel.columns:
        if column == "season":
            continue
        seasons = (
            panel.filter(pl.col(column).is_not_null())
            .select("season")
            .unique()
            .to_series()
            .sort()
            .to_list()
        )
        coverage[column] = seasons
    return coverage


def write(panel: pl.DataFrame, summary: dict[str, object], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / PANEL_NAME
    panel.write_parquet(target, compression="zstd")
    (dest / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return target
