"""Turn the panel into a feature table that cannot leak.

The leakage rule from :mod:`fpl_ml.schema` says an OUTCOME column for gameweek
*k* may never be a feature for gameweek *k*. This module enforces that by
construction rather than by care: an outcome column is reachable only through
:func:`lagged`, and :func:`lagged` always shifts by one gameweek first.

That shift is the whole point, and it is easy to get wrong. A rolling mean of
the last five gameweeks that *includes* the current one looks almost identical
in code and is completely wrong -- it contains the answer. So the shift happens
inside the helper, where a caller cannot forget it.

Every feature this module produces is therefore safe to use when predicting the
gameweek it sits on.
"""

from __future__ import annotations

import polars as pl

from . import schema

# Rows that are not footballers. Managers became pickable assets in 2024-25 and
# score under entirely different rules, so they must never train a points model.
MANAGER_POSITION = "AM"

# One season wrote GKP where every other season writes GK.
POSITION_FIXES = {"GKP": "GK"}

NUMERIC = (
    "round",
    "minutes",
    "total_points",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "xP",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
)

PLAYER_KEY = ("season", "code")


def prepare(panel: pl.DataFrame) -> pl.DataFrame:
    """Clean the panel into rows a model may learn from.

    Drops manager rows, normalises the position codes, casts the numeric
    columns, and sorts so that every rolling calculation runs in time order.
    """
    frame = panel
    if "position" in frame.columns:
        frame = frame.filter(
            pl.col("position").is_not_null() & (pl.col("position") != MANAGER_POSITION)
        ).with_columns(
            pl.col("position").replace(POSITION_FIXES).alias("position")
        )

    present = [c for c in NUMERIC if c in frame.columns]
    frame = frame.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in present])

    # A player with no permanent code cannot be followed between gameweeks.
    frame = frame.filter(pl.col("code").is_not_null() & pl.col("round").is_not_null())

    return frame.sort(["season", "code", "round"])


def lagged(column: str, window: int, *, statistic: str = "mean") -> pl.Expr:
    """A rolling statistic over the gameweeks *before* this one.

    The ``shift(1)`` is not optional and is not a parameter. It is what makes
    the result legal as a same-gameweek feature.
    """
    if column in schema.PRE_DEADLINE or column in schema.IDENTITY:
        raise ValueError(
            f"{column!r} is already knowable before the deadline; use it directly "
            "instead of lagging it."
        )
    schema.classify(column)  # raises for an unclassified column

    base = pl.col(column).shift(1).over(list(PLAYER_KEY))
    rolling = {
        "mean": base.rolling_mean(window, min_samples=1),
        "sum": base.rolling_sum(window, min_samples=1),
    }[statistic]
    return rolling.over(list(PLAYER_KEY)).alias(f"{column}_{statistic}{window}")


def expanding(column: str, *, statistic: str = "sum") -> pl.Expr:
    """A season-to-date total over the gameweeks before this one."""
    schema.classify(column)
    base = pl.col(column).shift(1).over(list(PLAYER_KEY))
    result = {"sum": base.cum_sum(), "count": base.is_not_null().cum_sum()}[statistic]
    return result.over(list(PLAYER_KEY)).alias(f"{column}_todate_{statistic}")


def build(panel: pl.DataFrame) -> pl.DataFrame:
    """Produce the feature table used by the baselines and the backtest."""
    frame = prepare(panel)

    frame = frame.with_columns(
        [
            lagged("total_points", 5),
            lagged("total_points", 10),
            lagged("minutes", 3),
            lagged("minutes", 5),
            expanding("total_points"),
            expanding("minutes"),
            # How often the player finished 60 minutes recently. The strongest
            # single clue about whether they will start the next match.
            (pl.col("minutes") >= 60)
            .cast(pl.Float64)
            .shift(1)
            .over(list(PLAYER_KEY))
            .rolling_mean(5, min_samples=1)
            .over(list(PLAYER_KEY))
            .alias("started_rate5"),
            (pl.col("minutes") > 0)
            .cast(pl.Float64)
            .shift(1)
            .over(list(PLAYER_KEY))
            .cum_sum()
            .over(list(PLAYER_KEY))
            .alias("appearances_todate"),
        ]
    )

    # Points for each 90 minutes played so far this season. Undefined until the
    # player has actually played, so it stays null rather than dividing by zero.
    frame = frame.with_columns(
        pl.when(pl.col("minutes_todate_sum") > 0)
        .then(pl.col("total_points_todate_sum") / pl.col("minutes_todate_sum") * 90)
        .otherwise(None)
        .alias("points_per90_todate")
    )

    return frame
