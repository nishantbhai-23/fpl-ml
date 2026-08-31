"""Capture tests: layout, verbatim storage, and honest manifests."""

from __future__ import annotations

import json

import httpx

from fpl_ml import archive, config, snapshot
from fpl_ml.client import FplClient

# Deliberately ugly: irregular whitespace and an escaped unicode sequence.
# Parsing and re-serialising this would normalise both, so an equality check
# against the stored bytes is a real test that we store what we received.
BOOTSTRAP_BODY = (
    b'{"events":[{"id":1,"is_current":false,"is_next":false},'
    b'{"id":2,"is_current":true,"is_next":false},'
    b'{"id":3,"is_current":false,"is_next":true,'
    b'"deadline_time":"2026-08-28T17:30:00Z"}],'
    b'   "elements":[ {"id":1,"web_name":"S\\u00e1nchez"},'
    b'{"id":7,"web_name":"Rodri"} ]  }'
)
FIXTURES_BODY = b'[{"id":1,"event":3,"team_h":1,"team_a":2}]'


def make_client(*, fixtures_status: int = 200) -> FplClient:
    """A client whose transport serves canned FPL responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/bootstrap-static/"):
            return httpx.Response(200, content=BOOTSTRAP_BODY)
        if path.endswith("/fixtures/"):
            return httpx.Response(fixtures_status, content=FIXTURES_BODY)
        if "/element-summary/" in path:
            element_id = int(path.rstrip("/").rsplit("/", 1)[-1])
            body = json.dumps({"history": [], "element": element_id}).encode()
            return httpx.Response(200, content=body)
        return httpx.Response(404)

    return FplClient(transport=httpx.MockTransport(handler), backoff_base=0.0)


async def test_capture_writes_verbatim_bytes(tmp_path):
    async with make_client() as client:
        run_dir = await snapshot.capture(client, root=tmp_path)

    assert archive.read_payload(run_dir, "bootstrap-static.json") == BOOTSTRAP_BODY
    assert archive.read_payload(run_dir, "fixtures.json") == FIXTURES_BODY


async def test_manifest_records_hashes_and_gameweek(tmp_path):
    async with make_client() as client:
        run_dir = await snapshot.capture(client, root=tmp_path)

    manifest = snapshot.read_manifest(run_dir)

    assert manifest["counts"] == {"ok": 2, "failed": 0}
    assert manifest["gameweek"] == {
        "current_event": 2,
        "next_event": 3,
        "next_deadline": "2026-08-28T17:30:00Z",
    }

    bootstrap_entry = next(
        entry for entry in manifest["entries"] if "bootstrap" in entry["url"]
    )
    assert bootstrap_entry["sha256"] == archive.sha256_hex(BOOTSTRAP_BODY)
    assert bootstrap_entry["bytes"] == len(BOOTSTRAP_BODY)
    assert bootstrap_entry["path"] == "bootstrap-static.json.gz"


async def test_partial_failure_is_recorded_not_raised(tmp_path):
    async with make_client(fixtures_status=404) as client:
        run_dir = await snapshot.capture(client, root=tmp_path)

    manifest = snapshot.read_manifest(run_dir)
    assert manifest["counts"] == {"ok": 1, "failed": 1}

    failure = next(entry for entry in manifest["entries"] if "fixtures" in entry["url"])
    assert failure["status"] == 404
    assert failure["path"] is None
    assert "404" in failure["error"]

    # The bootstrap payload still landed. A partial capture beats no capture,
    # because today cannot be re-captured tomorrow.
    assert archive.read_payload(run_dir, "bootstrap-static.json") == BOOTSTRAP_BODY


async def test_captures_never_overwrite_each_other(tmp_path):
    async with make_client() as client:
        first = await snapshot.capture(client, root=tmp_path)
        second = await snapshot.capture(client, root=tmp_path)

    assert first != second
    assert len(archive.list_runs(tmp_path)) == 2


async def test_player_sweep_fetches_every_element(tmp_path):
    async with make_client() as client:
        run_dir = await snapshot.capture(client, root=tmp_path, include_players=True)

    stored = json.loads(archive.read_payload(run_dir, "element-summary/7.json"))
    assert stored["element"] == 7

    manifest = snapshot.read_manifest(run_dir)
    assert manifest["include_players"] is True
    assert manifest["counts"] == {"ok": 4, "failed": 0}  # bootstrap, fixtures, 2 players


async def test_capture_is_usable_when_bootstrap_landed(tmp_path):
    # Fixtures failed, but bootstrap is what everything else depends on.
    async with make_client(fixtures_status=404) as client:
        run_dir = await snapshot.capture(client, root=tmp_path)

    assert snapshot.is_usable(snapshot.read_manifest(run_dir)) is True


def test_capture_is_not_usable_without_bootstrap():
    assert snapshot.is_usable({"entries": []}) is False
    assert (
        snapshot.is_usable(
            {
                "entries": [
                    {"url": config.BOOTSTRAP_URL, "status": 503, "path": None},
                ]
            }
        )
        is False
    )


def test_player_ids_reads_elements():
    assert snapshot.player_ids(BOOTSTRAP_BODY) == [1, 7]


def test_gameweek_context_tolerates_missing_bootstrap():
    assert snapshot.gameweek_context(None)["current_event"] is None
    assert snapshot.gameweek_context(b"not json")["next_deadline"] is None
