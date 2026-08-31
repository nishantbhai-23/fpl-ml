"""What each column means, and when you were allowed to know it.

The discipline that separates a real ML system from a demo: a feature for
gameweek *k* may only use information that existed before gameweek *k*'s
deadline. Everything in this module exists so that rule is checkable by a
machine rather than remembered by a person.

Three classes:

``IDENTITY``
    Who, when, and against whom. Fixed when fixtures are released, so always
    safe to use.

``PRE_DEADLINE``
    State that existed before the deadline — price, ownership, the transfer
    market, FPL's own published forecast. Safe as same-gameweek features.

``OUTCOME``
    What happened during or after the match. These are labels. Using gameweek
    *k*'s outcome to predict gameweek *k* is leakage; using gameweek *k-1*'s is
    an ordinary lagged feature and is fine.

A caveat specific to the historical backfill: its ``PRE_DEADLINE`` columns were
snapshotted whenever the upstream collector ran, not at the deadline, and carry
no as-of timestamp. They are *probably* deadline-accurate — ``transfers_in``
for a gameweek is finalised at that gameweek's deadline — but this cannot be
verified from the data itself. Treat them as suggestive for historical seasons,
and trust only your own captures for the current one.
"""

from __future__ import annotations

IDENTITY: frozenset[str] = frozenset(
    {
        "season",
        "name",
        "element",
        "id",
        "position",
        "team",
        "round",
        # Added by the upstream merger when it concatenates gw*.csv into
        # merged_gw.csv. Verified identical to `round` across all 27,605 rows
        # of 2024-25. Kept rather than dropped: deduplication is the feature
        # layer's call, not the ingest layer's.
        "GW",
        "fixture",
        "opponent_team",
        "was_home",
        "kickoff_time",
        "kickoff_time_formatted",
        "modified",
    }
)

PRE_DEADLINE: frozenset[str] = frozenset(
    {
        "value",  # price for the gameweek
        "selected",  # ownership
        "transfers_in",
        "transfers_out",
        "transfers_balance",
        "xP",  # FPL's own expected points, published before the deadline
        "loaned_in",  # a long-removed FPL feature, present in early seasons
        "loaned_out",
    }
)

OUTCOME: frozenset[str] = frozenset(
    {
        # Appearance
        "minutes",
        "starts",
        # Scoring events
        "goals_scored",
        "assists",
        "own_goals",
        "penalties_missed",
        "penalties_saved",
        "yellow_cards",
        "red_cards",
        "saves",
        "clean_sheets",
        "goals_conceded",
        "winning_goals",
        # Points
        "total_points",
        "bonus",
        "bps",
        # FPL's post-match indices
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "ea_index",
        # Post-match expected stats
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        # Match result
        "team_h_score",
        "team_a_score",
        # Defensive actions. Present in 2016-17, dropped, then reinstated in
        # 2025-26 when the defensive-contribution scoring rule was introduced.
        "defensive_contribution",
        "tackles",
        "recoveries",
        "clearances_blocks_interceptions",
        # Manager scoring. Managers became a pickable asset in recent seasons
        # and appear as their own rows, with a distinct position code. A points
        # model must not train on manager rows mixed in with footballer rows —
        # they score under entirely different rules.
        "mng_win",
        "mng_draw",
        "mng_loss",
        "mng_clean_sheets",
        "mng_goals_scored",
        "mng_underdog_win",
        "mng_underdog_draw",
        # Detailed match stats, 2016-17 era only
        "attempted_passes",
        "completed_passes",
        "key_passes",
        "big_chances_created",
        "big_chances_missed",
        "dribbles",
        "fouls",
        "offside",
        "open_play_crosses",
        "penalties_conceded",
        "tackled",
        "target_missed",
        "errors_leading_to_goal",
        "errors_leading_to_goal_attempt",
    }
)

_ALL = IDENTITY | PRE_DEADLINE | OUTCOME


class UnknownColumnError(ValueError):
    """A column appeared that nobody has classified yet.

    Deliberately fatal. FPL adds columns when it changes the game — the
    defensive-contribution stats arrived exactly this way in 2025-26 — and a
    new column silently flowing into a feature matrix is precisely how leakage
    gets in. Classify it, then carry on.
    """


def classify(column: str) -> str:
    if column in IDENTITY:
        return "identity"
    if column in PRE_DEADLINE:
        return "pre_deadline"
    if column in OUTCOME:
        return "outcome"
    raise UnknownColumnError(column)


def check_all_known(columns: object) -> None:
    """Raise if any column is unclassified. Call this on every ingest."""
    unknown = sorted(set(columns) - _ALL)
    if unknown:
        raise UnknownColumnError(
            f"unclassified column(s): {', '.join(unknown)}. "
            "Add each to IDENTITY, PRE_DEADLINE or OUTCOME in schema.py."
        )


def safe_for_same_gameweek(columns: object) -> list[str]:
    """The columns usable as features for the gameweek they describe."""
    return sorted(c for c in columns if c in IDENTITY or c in PRE_DEADLINE)


def must_be_lagged(columns: object) -> list[str]:
    """Columns that may only be used from *earlier* gameweeks."""
    return sorted(c for c in columns if c in OUTCOME)
