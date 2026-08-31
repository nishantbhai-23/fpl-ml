"""Walk-forward evaluation.

A normal machine learning split takes random rows for the test set. On this
data that is a lie: it lets a model learn from gameweek 20 and then predict
gameweek 10, which is impossible in real use, and it returns a score far better
than anything the system could really achieve.

So the harness walks forward instead. For each gameweek *k* it hands the
predictor everything before *k*, asks for a prediction of *k*, and scores it
against what actually happened. Then it moves on. Every prediction is made from
the past only, exactly as it would be on a Friday afternoon.

The metrics deliberately lead with **rank correlation within position**, not
with error. You never need a player's absolute point total to be right. You
need to know that this midfielder will out-score that midfielder, because the
decision you actually make is a choice between them.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from . import baselines

# A predictor sees the history and the rows to predict, and returns one
# prediction per target row. Baselines ignore the history, because their
# features are already lagged; a trained model will fit on it.
Predictor = Callable[[pl.DataFrame, pl.DataFrame], pl.Series]

# Rolling features need history before they mean anything.
DEFAULT_MIN_GAMEWEEK = 6

# A gameweek-position group needs enough players for a rank correlation to say
# anything. Five is already generous.
MIN_GROUP = 5


def baseline_predictor(name: str) -> Predictor:
    def predict(history: pl.DataFrame, target: pl.DataFrame) -> pl.Series:
        return baselines.predict(target, name)["prediction"]

    return predict


def run(
    frame: pl.DataFrame,
    predictor: Predictor,
    *,
    min_gameweek: int = DEFAULT_MIN_GAMEWEEK,
) -> pl.DataFrame:
    """Walk forward through every gameweek and collect the predictions."""
    chunks: list[pl.DataFrame] = []

    for (season,), season_rows in frame.group_by(["season"], maintain_order=True):
        gameweeks = sorted(season_rows["round"].unique().drop_nulls().to_list())
        for gameweek in gameweeks:
            if gameweek < min_gameweek:
                continue
            target = season_rows.filter(pl.col("round") == gameweek)
            if target.height == 0:
                continue
            history = season_rows.filter(pl.col("round") < gameweek)
            prediction = predictor(history, target)
            chunks.append(
                target.select(["season", "round", "code", "name", "position"]).with_columns(
                    [
                        pl.Series("prediction", prediction).cast(pl.Float64),
                        target["total_points"].alias("actual"),
                        target["minutes"].alias("minutes"),
                    ]
                )
            )

    if not chunks:
        raise ValueError("no gameweeks were predicted; check min_gameweek")
    return pl.concat(chunks)


def _rank_correlation(results: pl.DataFrame) -> float | None:
    """Mean Spearman correlation, computed inside each gameweek and position.

    Spearman is Pearson applied to ranks, which is why this ranks first and
    then correlates. Grouping by position matters: comparing a goalkeeper's
    points against a forward's would reward a model for rediscovering that
    forwards score more, which is not a useful skill.
    """
    grouped = (
        results.drop_nulls(["prediction", "actual"])
        .with_columns(
            [
                pl.col("prediction").rank("average").over(["season", "round", "position"]).alias("_pr"),
                pl.col("actual").rank("average").over(["season", "round", "position"]).alias("_ar"),
            ]
        )
        .group_by(["season", "round", "position"])
        .agg([pl.corr("_pr", "_ar").alias("rho"), pl.len().alias("n")])
        # NaN and null are different things in polars, and a correlation over a
        # group with no variance returns NaN, not null. A baseline that predicts
        # the same value for every player in a group -- xP does exactly that,
        # giving 0.0 to everyone it expects not to play -- would otherwise poison
        # the mean with NaN and report no score at all.
        .filter(
            (pl.col("n") >= MIN_GROUP)
            & pl.col("rho").is_not_null()
            & pl.col("rho").is_not_nan()
        )
    )
    if grouped.height == 0:
        return None
    return float(grouped["rho"].mean())


def _top_overlap(results: pl.DataFrame, k: int = 20) -> float | None:
    """Of the k highest-predicted players each gameweek, how many land in the actual top k?

    Closer to the real decision than any error metric: you pick a small squad,
    so what matters is whether the best players appear near the top of the list.
    """
    hits: list[float] = []
    for _, gw in results.drop_nulls(["prediction", "actual"]).group_by(["season", "round"]):
        if gw.height < k * 2:
            continue
        predicted = set(gw.sort("prediction", descending=True).head(k)["code"].to_list())
        actual = set(gw.sort("actual", descending=True).head(k)["code"].to_list())
        hits.append(len(predicted & actual) / k)
    return float(sum(hits) / len(hits)) if hits else None


def score(results: pl.DataFrame, *, points_scale: bool = True) -> dict[str, object]:
    """Metrics for one set of walk-forward predictions."""
    played = results.filter(pl.col("minutes") > 0)

    out: dict[str, object] = {
        "gameweeks": results.select(["season", "round"]).unique().height,
        "rows": results.height,
        "rank_corr": _rank_correlation(results),
        "rank_corr_played": _rank_correlation(played),
        "top20_overlap": _top_overlap(results),
    }

    # An error against actual points is meaningless for a baseline that only
    # ranks players, so it is reported only where the units line up.
    if points_scale:
        err = (results["prediction"] - results["actual"]).abs()
        out["mae"] = float(err.mean())
        out["mae_played"] = float((played["prediction"] - played["actual"]).abs().mean())
    else:
        out["mae"] = None
        out["mae_played"] = None

    return out


def compare_baselines(
    frame: pl.DataFrame, *, min_gameweek: int = DEFAULT_MIN_GAMEWEEK
) -> pl.DataFrame:
    """Run every baseline through the harness and gather the scores."""
    rows = []
    for name, spec in baselines.BASELINES.items():
        results = run(frame, baseline_predictor(name), min_gameweek=min_gameweek)
        metrics = score(results, points_scale=bool(spec["points_scale"]))
        rows.append({"baseline": name, **metrics})
    return pl.DataFrame(rows).sort("rank_corr", descending=True)
