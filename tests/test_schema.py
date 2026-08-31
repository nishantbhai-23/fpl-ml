"""Column provenance: the foundation of the leakage test."""

from __future__ import annotations

import pytest

from fpl_ml import schema


def test_classifies_the_three_kinds():
    assert schema.classify("round") == "identity"
    assert schema.classify("value") == "pre_deadline"
    assert schema.classify("total_points") == "outcome"


def test_unknown_column_is_fatal():
    # The whole point: FPL adds columns when it changes the game, and an
    # unclassified one reaching a feature matrix is how leakage gets in.
    with pytest.raises(schema.UnknownColumnError):
        schema.classify("some_stat_fpl_added_last_tuesday")

    with pytest.raises(schema.UnknownColumnError, match="brand_new"):
        schema.check_all_known(["round", "value", "brand_new"])


def test_check_all_known_passes_for_known_columns():
    schema.check_all_known(["round", "value", "total_points", "GW"])


def test_outcomes_are_never_same_gameweek_safe():
    columns = ["round", "value", "total_points", "minutes", "expected_goals"]
    safe = schema.safe_for_same_gameweek(columns)
    lagged = schema.must_be_lagged(columns)

    assert safe == ["round", "value"]
    assert lagged == ["expected_goals", "minutes", "total_points"]
    # Every column lands in exactly one bucket.
    assert set(safe) | set(lagged) == set(columns)
    assert not set(safe) & set(lagged)


def test_defensive_and_manager_stats_are_outcomes():
    # Both arrived with rule changes; both are post-match.
    for column in ("defensive_contribution", "tackles", "mng_win", "mng_clean_sheets"):
        assert schema.classify(column) == "outcome"


def test_fpl_own_forecast_is_pre_deadline():
    # xP is FPL's published expectation, available before the deadline, and is
    # one of the four baselines Phase 01 requires beating.
    assert schema.classify("xP") == "pre_deadline"
