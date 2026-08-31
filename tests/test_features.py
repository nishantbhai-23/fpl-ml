"""Leakage safety: the property the whole harness rests on."""

from __future__ import annotations

import polars as pl
import pytest

from fpl_ml import backtest, features


def panel_frame() -> pl.DataFrame:
    # One player, six gameweeks, points 1..6 so a leak is obvious by value.
    return pl.DataFrame(
        {
            "season": ["2024-25"] * 6,
            "code": ["999"] * 6,
            "element": ["1"] * 6,
            "name": ["Test Player"] * 6,
            "position": ["MID"] * 6,
            "round": ["1", "2", "3", "4", "5", "6"],
            "minutes": ["90"] * 6,
            "total_points": ["1", "2", "3", "4", "5", "6"],
            "value": ["50"] * 6,
            "selected": ["100"] * 6,
            "transfers_balance": ["0"] * 6,
            "xP": ["2.0"] * 6,
        }
    )


def test_lagged_feature_excludes_the_current_gameweek():
    built = features.build(panel_frame()).sort("round")
    rolling = built["total_points_mean5"].to_list()

    # Gameweek 1 has no past at all.
    assert rolling[0] is None
    # Gameweek 2 sees only gameweek 1, so the mean is 1.
    assert rolling[1] == pytest.approx(1.0)
    # Gameweek 4 sees 1, 2, 3 -> mean 2. If the current gameweek leaked in it
    # would be 2.5, which is exactly the bug this test exists to catch.
    assert rolling[3] == pytest.approx(2.0)


def test_season_to_date_totals_exclude_the_current_gameweek():
    built = features.build(panel_frame()).sort("round")
    # By gameweek 5 the player has scored 1+2+3+4 = 10 in completed gameweeks.
    assert built["total_points_todate_sum"].to_list()[4] == pytest.approx(10.0)


def test_lagging_a_pre_deadline_column_is_refused():
    # `value` is already knowable before the deadline. Lagging it would throw
    # away information for no reason, so the helper refuses.
    with pytest.raises(ValueError, match="knowable before the deadline"):
        features.lagged("value", 5)


def test_lagging_an_unclassified_column_is_refused():
    from fpl_ml import schema

    with pytest.raises(schema.UnknownColumnError):
        features.lagged("invented_stat", 5)


def test_manager_rows_are_dropped():
    frame = panel_frame().with_columns(pl.lit("AM").alias("position"))
    assert features.prepare(frame).height == 0


def test_goalkeeper_position_codes_are_normalised():
    frame = panel_frame().with_columns(pl.lit("GKP").alias("position"))
    assert features.prepare(frame)["position"].unique().to_list() == ["GK"]


def test_zero_variance_groups_do_not_poison_the_rank_correlation():
    # A baseline predicting one value for everyone yields a NaN correlation,
    # not a null. NaN must be excluded or it destroys the mean.
    results = pl.DataFrame(
        {
            "season": ["2024-25"] * 12,
            "round": [1] * 6 + [2] * 6,
            "position": ["MID"] * 12,
            "code": [str(i) for i in range(12)],
            "prediction": [0.0] * 6 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "actual": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
            "minutes": [90.0] * 12,
        }
    )
    rho = backtest._rank_correlation(results)
    # Gameweek 1 is all-ties and is skipped; gameweek 2 is a perfect match.
    assert rho == pytest.approx(1.0)
