"""Turn our own captures into panel rows for the season in progress.

The backfill stops where the community collector last ran. The season in
progress moves on, and the only record of it that we control is the archive in
``raw/``. This module reads the newest full capture and produces rows in the
same shape as the panel, so the features and the baselines work unchanged.

Two kinds of row come out:

*Completed gameweeks* come from each player's ``history``, which carries the
same fields as the community CSVs — minutes, points, price, transfers.

*The upcoming gameweek* comes from ``bootstrap-static``, and holds only what is
knowable before the deadline: price, ownership, the transfer market, the
availability status, and FPL's own forecast. Every outcome field is null,
because the matches have not been played.

That second kind is the point. A stub row for the upcoming gameweek lets the
lagged features in :mod:`fpl_ml.features` compute exactly as they do in the
backtest — the rolling means look back at the completed gameweeks and stop at
the deadline, with no special case anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from . import archive

PROVENANCE = "capture"

# Fields on a history row that map straight onto panel columns.
HISTORY_FIELDS = (
    "round",
    "minutes",
    "total_points",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "opponent_team",
    "was_home",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
    "starts",
    "kickoff_time",
)


def _bootstrap(run_dir: Path) -> dict:
    return json.loads(archive.read_payload(run_dir, "bootstrap-static.json"))


def season_name(bootstrap: dict) -> str:
    """Work out the season label, e.g. 2026-27, from the fixture calendar.

    FPL does not publish the season name anywhere in bootstrap-static, so it is
    derived from the first gameweek's deadline: a season that starts in August
    2026 is 2026-27.
    """
    deadlines = [e["deadline_time"] for e in bootstrap["events"] if e.get("deadline_time")]
    if not deadlines:
        raise ValueError("no gameweek deadlines in bootstrap-static")
    start_year = int(min(deadlines)[:4])
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def next_gameweek(bootstrap: dict) -> dict | None:
    """The gameweek whose deadline has not passed yet."""
    return next((e for e in bootstrap["events"] if e.get("is_next")), None)


def _player_index(bootstrap: dict) -> dict[int, dict[str, object]]:
    positions = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    index = {}
    for element in bootstrap["elements"]:
        index[element["id"]] = {
            "code": str(element["code"]),
            "name": f"{element['first_name']} {element['second_name']}".strip(),
            "position": positions.get(element["element_type"]),
            "team": teams.get(element["team"]),
            "element": str(element["id"]),
        }
    return index


def completed_gameweeks(run_dir: Path) -> pl.DataFrame:
    """Rows for every gameweek already played this season."""
    bootstrap = _bootstrap(run_dir)
    season = season_name(bootstrap)
    index = _player_index(bootstrap)

    records: list[dict[str, object]] = []
    for element_id, meta in index.items():
        try:
            summary = json.loads(
                archive.read_payload(run_dir, f"element-summary/{element_id}.json")
            )
        except (FileNotFoundError, OSError):
            continue
        for row in summary.get("history", []):
            record: dict[str, object] = {
                "season": season,
                "provenance": PROVENANCE,
                **meta,
            }
            for field in HISTORY_FIELDS:
                if field in row:
                    value = row[field]
                    record[field] = str(value) if value is not None else None
            records.append(record)

    if not records:
        raise ValueError(
            f"no per-player history in {run_dir}; this needs a --players capture"
        )
    return pl.DataFrame(records, infer_schema_length=None)


def upcoming_gameweek(run_dir: Path) -> pl.DataFrame:
    """Pre-deadline stub rows for the gameweek that has not started.

    Only columns knowable before the deadline are filled. Every outcome column
    is absent, which is what makes these rows safe to hand to the feature
    builder next to completed ones.
    """
    bootstrap = _bootstrap(run_dir)
    event = next_gameweek(bootstrap)
    if event is None:
        raise ValueError("no upcoming gameweek; the season may be over")

    season = season_name(bootstrap)
    index = _player_index(bootstrap)
    total_players = bootstrap.get("total_players") or 0

    records = []
    for element in bootstrap["elements"]:
        meta = index[element["id"]]
        owned_pct = float(element.get("selected_by_percent") or 0.0)
        records.append(
            {
                "season": season,
                "provenance": PROVENANCE,
                **meta,
                "round": str(event["id"]),
                "value": str(element["now_cost"]),
                # History stores an owner count; bootstrap stores a percentage.
                # Converted here so the column means one thing everywhere.
                "selected": str(int(owned_pct / 100.0 * total_players)),
                "transfers_in": str(element.get("transfers_in_event") or 0),
                "transfers_out": str(element.get("transfers_out_event") or 0),
                "transfers_balance": str(
                    (element.get("transfers_in_event") or 0)
                    - (element.get("transfers_out_event") or 0)
                ),
                # FPL's own forecast for the next gameweek. Same quantity the
                # backfill calls xP, under a different name.
                "xP": str(element.get("ep_next") or 0.0),
                # Availability, which the backfill has no equivalent for at all.
                "status": element.get("status"),
                "chance_of_playing": (
                    str(element["chance_of_playing_next_round"])
                    if element.get("chance_of_playing_next_round") is not None
                    else None
                ),
                "news": element.get("news") or None,
            }
        )

    return pl.DataFrame(records, infer_schema_length=None)


def build(run_dir: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    """Completed rows plus upcoming stubs, ready for the feature builder."""
    bootstrap = _bootstrap(run_dir)
    event = next_gameweek(bootstrap)
    past = completed_gameweeks(run_dir)
    upcoming = upcoming_gameweek(run_dir)

    frame = pl.concat([past, upcoming], how="diagonal_relaxed").sort(
        ["season", "code", "round"]
    )

    context = {
        "season": season_name(bootstrap),
        "next_gameweek": event["id"] if event else None,
        "deadline": event["deadline_time"] if event else None,
        "capture": run_dir.name,
        "completed_rows": past.height,
        "upcoming_rows": upcoming.height,
    }
    return frame, context
