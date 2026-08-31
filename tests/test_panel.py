"""Panel normalisation: encoding, schema drift, and the unknown-column guard."""

from __future__ import annotations

import pytest

from fpl_ml import panel, schema, validate

GW_HEADER = "name,element,round,value,minutes,total_points"


def write_season(root, season, body, *, encoding="utf-8"):
    path = root / season / panel.GAMEWEEK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body.encode(encoding))
    return path


def test_reads_utf8_and_latin1_seasons(tmp_path):
    # The three earliest upstream seasons are latin-1; the rest are UTF-8.
    write_season(tmp_path, "2016-17", f"{GW_HEADER}\nJosé Fonte,1,1,55,90,6\n", encoding="latin-1")
    write_season(tmp_path, "2024-25", f"{GW_HEADER}\nJosé Fonte,1,1,55,90,6\n")

    frame, summary = panel.build(tmp_path, seasons=("2016-17", "2024-25"))

    assert summary["seasons"]["2016-17"]["encoding"] == "latin-1"
    assert summary["seasons"]["2024-25"]["encoding"] == "utf-8"
    # The accent survives both routes -- that is the entire point of detecting.
    assert frame["name"].to_list() == ["José Fonte", "José Fonte"]


def test_missing_season_is_recorded_not_fatal(tmp_path):
    write_season(tmp_path, "2024-25", f"{GW_HEADER}\nA,1,1,55,90,6\n")

    _, summary = panel.build(tmp_path, seasons=("2016-17", "2024-25"))

    assert summary["seasons"]["2016-17"]["status"] == "not vendored"
    assert summary["seasons"]["2024-25"]["rows"] == 1


def test_unknown_column_stops_the_build(tmp_path):
    write_season(tmp_path, "2024-25", f"{GW_HEADER},mystery_stat\nA,1,1,55,90,6,3\n")

    with pytest.raises(schema.UnknownColumnError, match="mystery_stat"):
        panel.build(tmp_path, seasons=("2024-25",))


def test_columns_absent_from_a_season_become_null(tmp_path):
    # Seasons genuinely differ; a missing column must land as null rather than
    # silently aligning to whatever column happens to sit in that position.
    write_season(tmp_path, "2019-20", f"{GW_HEADER}\nA,1,1,55,90,6\n")
    write_season(tmp_path, "2024-25", f"{GW_HEADER},expected_goals\nB,2,1,60,90,8,0.4\n")

    frame, summary = panel.build(tmp_path, seasons=("2019-20", "2024-25"))

    assert frame.height == 2
    assert summary["coverage"]["expected_goals"] == ["2024-25"]
    older = frame.filter(frame["season"] == "2019-20")
    assert older["expected_goals"].to_list() == [None]


def test_no_vendored_seasons_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        panel.build(tmp_path, seasons=("2024-25",))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Aaron_Cresswell", "aaron cresswell"),  # early-season underscores
        ("Aaron_Cresswell_402", "aaron cresswell"),  # trailing element id
        ("Đorđe Petrović", "dorde petrovic"),  # NFKD folds c-acute...
        ("Łukasz Fabiański", "lukasz fabianski"),  # ...but l-stroke needs the table
        ("Ole Gunnar Solskjær", "ole gunnar solskjaer"),
        ("  Alex   Scott ", "alex scott"),
    ],
)
def test_name_normalisation(raw, expected):
    assert validate.normalise_name(raw) == expected
