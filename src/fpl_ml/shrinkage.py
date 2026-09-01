"""Shrink a small-sample average toward a prior.

The naive baselines fail in a specific, diagnosable way. After two gameweeks a
player who scored 14 then 6 is predicted at 10.0, ahead of established
performers, because a two-match average carries almost no information and the
formula has no way to know that. It chases outliers.

The fix is the oldest idea in statistics that still earns its keep: pull a noisy
estimate toward what you believed before you saw it, and pull harder when you
have seen less.

    prediction = (n · observed_mean + k · prior_mean) / (n + k)

``n`` is how many gameweeks the player has behind them, so the observed average
takes over as evidence accumulates. ``k`` behaves like a number of imaginary
gameweeks already spent agreeing with the prior. It is not chosen by hand: with

    k = within-player variance / between-player variance

the formula is the posterior mean of a normal-normal model, and it is the same
partial pooling described for promoted clubs, applied one level down.

The prior comes from **position and price**. Price is the single most useful
thing to condition on, because it is the market's own estimate of a player's
worth, set before a ball is kicked and updated by eleven million people. A
£4.0m defender and a £15.5m forward should not start from the same guess.

Everything here is fitted on completed gameweeks only, so a prediction never
sees its own answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

PRICE_BUCKETS = 10

# Guards for the degenerate cases: no history at all, or a variance estimate
# that collapses. Both give maximum shrinkage, which is the safe direction.
MIN_ROWS_FOR_FIT = 200
MIN_PLAYER_GAMEWEEKS = 5
FALLBACK_K = 10.0


@dataclass(frozen=True)
class Prior:
    """A fitted prior: expected points by group, and how hard to shrink."""

    table: pl.DataFrame  # position, price_bucket, prior_mean
    k: float
    grand_mean: float

    def describe(self) -> str:
        return f"k={self.k:.2f} (higher means more shrinkage), grand mean={self.grand_mean:.2f}"


def _with_price_bucket(frame: pl.DataFrame) -> pl.DataFrame:
    """Bucket price by rank within each gameweek.

    Rank rather than absolute price, because prices inflate across seasons: a
    £10m midfielder in 2016 is not the same asset as a £10m midfielder now.
    Rank keeps the buckets comparable.
    """
    return frame.with_columns(
        (
            pl.col("value").rank("average").over(["season", "round"])
            / pl.len().over(["season", "round"])
            * PRICE_BUCKETS
        )
        .ceil()
        .clip(1, PRICE_BUCKETS)
        .alias("price_bucket")
    )


def fit(history: pl.DataFrame) -> Prior:
    """Estimate the group priors and the shrinkage strength from completed rows."""
    usable = history.drop_nulls(["total_points", "value", "position"])
    grand_mean = float(usable["total_points"].mean()) if usable.height else 0.0

    if usable.height < MIN_ROWS_FOR_FIT:
        empty = pl.DataFrame(
            schema={"position": pl.Utf8, "price_bucket": pl.Float64, "prior_mean": pl.Float64}
        )
        return Prior(table=empty, k=FALLBACK_K, grand_mean=grand_mean)

    bucketed = _with_price_bucket(usable)
    table = bucketed.group_by(["position", "price_bucket"]).agg(
        pl.col("total_points").mean().alias("prior_mean")
    )

    # Empirical Bayes. The spread of observed player averages contains both the
    # real spread of ability and the noise of measuring it, so the noise term is
    # subtracted out before the two are compared.
    per_player = (
        bucketed.group_by(["season", "code"])
        .agg(
            [
                pl.col("total_points").mean().alias("player_mean"),
                pl.col("total_points").var().alias("player_var"),
                pl.len().alias("n"),
            ]
        )
        .filter(pl.col("n") >= MIN_PLAYER_GAMEWEEKS)
        .drop_nulls(["player_var"])
    )

    k = FALLBACK_K
    if per_player.height > MIN_PLAYER_GAMEWEEKS:
        within = float(per_player["player_var"].mean())
        observed_spread = float(per_player["player_mean"].var())
        mean_n = float(per_player["n"].mean())
        between = observed_spread - within / mean_n
        if between > 1e-6 and within > 0:
            k = within / between

    return Prior(table=table, k=float(k), grand_mean=grand_mean)


def predict(prior: Prior, target: pl.DataFrame) -> pl.Series:
    """Shrink each player's season-to-date average toward their group prior."""
    frame = _with_price_bucket(target).join(
        prior.table, on=["position", "price_bucket"], how="left"
    )

    prior_mean = pl.col("prior_mean").fill_null(prior.grand_mean)

    # Gameweeks elapsed, not appearances: a gameweek missed really did score
    # zero, and that is information about what next week is worth.
    gameweeks = (pl.col("round") - 1).clip(0, None).cast(pl.Float64)
    observed_total = pl.col("total_points_todate_sum").fill_null(0.0)

    shrunk = (observed_total + prior.k * prior_mean) / (gameweeks + prior.k)

    return frame.select(shrunk.alias("prediction"))["prediction"]


def predictor(history_seasons: pl.DataFrame | None = None):
    """Build a predictor for the walk-forward harness.

    If ``history_seasons`` is given (completed past seasons), the prior is fitted
    once from it. Otherwise the prior is refitted from the history the harness
    supplies at each gameweek, which is slower but uses only same-season data.
    """
    fixed = fit(history_seasons) if history_seasons is not None else None

    def run(history: pl.DataFrame, target: pl.DataFrame) -> pl.Series:
        prior = fixed if fixed is not None else fit(history)
        return predict(prior, target)

    return run
