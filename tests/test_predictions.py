"""The append-only guarantees on the prediction log."""

from __future__ import annotations

import polars as pl
import pytest

from fpl_ml import predictions

FUTURE = "2099-01-01T00:00:00Z"
PAST = "2000-01-01T00:00:00Z"


def frame(value: float = 5.0) -> pl.DataFrame:
    return pl.DataFrame(
        {"code": ["1", "2"], "name": ["A", "B"], "position": ["MID", "DEF"],
         "prediction": [value, value / 2]}
    )


def test_writes_and_ranks(tmp_path):
    target = predictions.write(
        tmp_path, season="2026-27", gameweek=3, deadline=FUTURE,
        capture="cap", predictions={"m1": frame()},
    )
    rows = pl.read_csv(target / "m1.csv")
    assert rows["rank"].to_list() == [1, 2]
    assert predictions.read_manifest(target)["gameweek"] == 3


def test_a_second_model_may_be_added_before_the_deadline(tmp_path):
    predictions.write(tmp_path, season="2026-27", gameweek=3, deadline=FUTURE,
                      capture="cap", predictions={"m1": frame()})
    target = predictions.write(tmp_path, season="2026-27", gameweek=3, deadline=FUTURE,
                               capture="cap", predictions={"m2": frame(9.0)})

    logged = {m["model"] for m in predictions.read_manifest(target)["models"]}
    assert logged == {"m1", "m2"}


def test_rewriting_a_logged_model_is_refused(tmp_path):
    predictions.write(tmp_path, season="2026-27", gameweek=3, deadline=FUTURE,
                      capture="cap", predictions={"m1": frame()})

    with pytest.raises(FileExistsError, match="m1"):
        predictions.write(tmp_path, season="2026-27", gameweek=3, deadline=FUTURE,
                          capture="cap", predictions={"m1": frame(99.0)})


def test_writing_after_the_deadline_is_refused(tmp_path):
    # The guard that actually matters: after the deadline the answer exists,
    # so anything written is not a prediction.
    with pytest.raises(ValueError, match="has passed"):
        predictions.write(tmp_path, season="2026-27", gameweek=3, deadline=PAST,
                          capture="cap", predictions={"m1": frame()})


def test_every_model_records_its_own_timestamp(tmp_path):
    target = predictions.write(tmp_path, season="2026-27", gameweek=3, deadline=FUTURE,
                               capture="cap", predictions={"m1": frame()})
    entry = predictions.read_manifest(target)["models"][0]
    assert entry["made_at"] < FUTURE
    assert entry["sha256"]
