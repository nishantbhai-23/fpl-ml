"""The four baselines that any real model must beat.

A baseline is a simple rule you compare a model against. If the model cannot
beat all four of these, it has no value, however sophisticated it looks.

The fourth one is the interesting one. Eleven million people make transfer
decisions every week, and the sum of those decisions carries a great deal of
information about who is about to do well. It is free, it updates continuously,
and it already prices in most of the injury news a model would work hard to
discover. Being forced to beat it is the most useful humbling available.

Every baseline here reads only lagged or pre-deadline features, so each one is
legal to compute before the deadline of the gameweek it predicts.
"""

from __future__ import annotations

import polars as pl

# A baseline maps a feature frame to a prediction column.
# `points_scale` says whether the output is in units of FPL points. Two of the
# baselines rank players without predicting a point total, so an error metric
# against actual points would be meaningless for them.
BASELINES: dict[str, dict[str, object]] = {}


def _register(name: str, *, points_scale: bool, description: str):
    def wrap(fn):
        BASELINES[name] = {
            "fn": fn,
            "points_scale": points_scale,
            "description": description,
        }
        return fn

    return wrap


@_register(
    "ppm90_x_minutes",
    points_scale=True,
    description="Season points per 90 so far, times the minutes we expect them to play.",
)
def ppm90_times_minutes(frame: pl.DataFrame) -> pl.Series:
    """Rate times volume — the most honest simple model of a points total.

    Expected minutes here is just the player's recent mean, which is a crude
    stand-in for the real minutes model that Phase 02 builds.
    """
    return (
        pl.when(pl.col("points_per90_todate").is_not_null() & pl.col("minutes_mean5").is_not_null())
        .then(pl.col("points_per90_todate") * pl.col("minutes_mean5") / 90.0)
        .otherwise(0.0)
        .alias("prediction")
    )


@_register(
    "rolling_mean5",
    points_scale=True,
    description="Mean points over the player's last five gameweeks.",
)
def rolling_mean5(frame: pl.DataFrame) -> pl.Expr:
    return pl.col("total_points_mean5").fill_null(0.0).alias("prediction")


@_register(
    "fpl_xp",
    points_scale=True,
    description="The game's own published forecast, available before the deadline.",
)
def fpl_xp(frame: pl.DataFrame) -> pl.Expr:
    return pl.col("xP").fill_null(0.0).alias("prediction")


@_register(
    "crowd",
    points_scale=False,
    description="Ownership plus this week's net transfers — what eleven million managers think.",
)
def crowd(frame: pl.DataFrame) -> pl.Expr:
    """A ranking signal, not a points estimate.

    Ownership sets the level and net transfers supply the momentum: a player
    many managers are buying right now is one the crowd has just learned
    something good about. The two are on wildly different scales, so each is
    ranked within the gameweek before they are combined.
    """
    owned = pl.col("selected").fill_null(0.0).rank("average").over("_gw_key")
    moving = pl.col("transfers_balance").fill_null(0.0).rank("average").over("_gw_key")
    return (owned + 0.5 * moving).alias("prediction")


def predict(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Attach a ``prediction`` column produced by the named baseline."""
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(BASELINES)}")
    frame = frame.with_columns(
        pl.concat_str([pl.col("season"), pl.col("round").cast(pl.Utf8)], separator="|").alias("_gw_key")
    )
    return frame.with_columns(BASELINES[name]["fn"](frame))
