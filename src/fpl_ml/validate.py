"""Check the community backfill against FPL's own record.

Every ``element-summary`` payload we capture carries ``history_past``: FPL's
official per-season totals for that player. That gives us a source of truth,
straight from the game, to audit third-party history against — and it costs
nothing extra, because we already captured it.

The join is by normalised name and season, and **that is the weak link**.
Player IDs are reassigned between seasons, so name matching is the only route,
and names are genuinely unreliable: accents come and go, clubs record players
differently, and surnames collide. A disagreement found here is therefore
evidence about *either* the backfill or the match, and the match is usually the
likelier culprit. Treat the result as a smoke alarm, not a proof.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import polars as pl

from . import archive


# Letters that NFKD cannot fold, because the diacritic is part of the glyph
# rather than a combining mark: a stroke through the letter, or a ligature.
# Without these, "Đorđe" and "Dorde" are different players.
_UNDECOMPOSABLE = str.maketrans(
    {
        "đ": "d",
        "ð": "d",
        "ł": "l",
        "ø": "o",
        "þ": "th",
        "æ": "ae",
        "œ": "oe",
        "ß": "ss",
        "ı": "i",
    }
)


def normalise_name(name: str) -> str:
    """Fold a player name into something joinable across sources.

    Handles the known format differences: early seasons separate names with
    underscores, some append the element ID, and accents are inconsistent
    between the API and the community CSVs.

    Two passes are needed for accents. NFKD splits most accented letters into a
    base letter plus a combining mark, which we then drop — that turns "ć" into
    "c". But letters whose diacritic is drawn *through* the glyph have no such
    decomposition and survive NFKD untouched, so they are transliterated
    explicitly afterwards.

    This is still only a good approximation. Name matching is inherently
    lossy — surnames collide, clubs record players differently, and some
    players are known by a single name — so treat matches as evidence rather
    than identity. Phase 02 will need a real player-linking table.
    """
    name = name.replace("_", " ")
    name = re.sub(r"\s+\d+$", "", name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().translate(_UNDECOMPOSABLE)
    return re.sub(r"\s+", " ", name).strip()


def official_season_totals(run_dir: Path) -> pl.DataFrame:
    """FPL's own per-season points totals, from a full capture's payloads.

    Requires a capture taken with ``--players``; the per-player summaries are
    where ``history_past`` lives.
    """
    bootstrap = json.loads(archive.read_payload(run_dir, "bootstrap-static.json"))
    records: list[dict[str, object]] = []

    for element in bootstrap["elements"]:
        try:
            summary = json.loads(
                archive.read_payload(run_dir, f"element-summary/{element['id']}.json")
            )
        except (FileNotFoundError, OSError):
            continue
        name = normalise_name(f"{element['first_name']} {element['second_name']}")
        for past in summary.get("history_past", []):
            records.append(
                {
                    "name_n": name,
                    # FPL writes 2021/22; the community archive writes 2021-22.
                    "season": str(past["season_name"]).replace("/", "-"),
                    "official_points": int(past["total_points"]),
                }
            )

    return pl.DataFrame(records)


def backfill_season_totals(panel: pl.DataFrame) -> pl.DataFrame:
    """Per-season points totals implied by the backfill panel."""
    return (
        panel.filter(pl.col("total_points").is_not_null())
        .with_columns(pl.col("total_points").cast(pl.Int64, strict=False))
        .group_by(["season", "name"])
        .agg(pl.col("total_points").sum().alias("backfill_points"))
        .with_columns(
            pl.col("name")
            .map_elements(normalise_name, return_dtype=pl.Utf8)
            .alias("name_n")
        )
    )


def compare(panel: pl.DataFrame, run_dir: Path) -> dict[str, object]:
    """Compare backfill season totals against FPL's official record."""
    official = official_season_totals(run_dir)
    if official.is_empty():
        return {"matched": 0, "note": "no history_past found; needs a --players capture"}

    joined = official.join(backfill_season_totals(panel), on=["name_n", "season"], how="inner")
    if joined.is_empty():
        return {"matched": 0, "note": "no player-seasons matched"}

    disagreements = joined.filter(
        pl.col("official_points") != pl.col("backfill_points")
    ).sort("name")

    return {
        "matched": joined.height,
        "agreed": joined.height - disagreements.height,
        "agreement_rate": round(
            (joined.height - disagreements.height) / joined.height, 4
        ),
        "disagreements": [
            {
                "name": row["name"],
                "season": row["season"],
                "official": row["official_points"],
                "backfill": row["backfill_points"],
            }
            for row in disagreements.head(50).iter_rows(named=True)
        ],
    }
