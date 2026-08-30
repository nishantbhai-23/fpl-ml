"""Where captured bytes go on disk.

One rule, and everything downstream depends on it: **a run directory is created
once and never written to twice.** The archive is append-only. Nothing in this
project may edit or delete a past capture, because the archive is the only
record of what the API said before a deadline -- and that record cannot be
rebuilt from any other source.

Layout::

    data/raw/2026-08-31T14-03-22Z/
      bootstrap-static.json.gz
      fixtures.json.gz
      element-summary/1.json.gz, 2.json.gz, ...
      manifest.json

Payloads are gzipped because daily JSON captures add up quickly. The manifest
stays plain text so the archive can be inspected without any tooling.
"""

from __future__ import annotations

import gzip
import hashlib
from datetime import UTC, datetime
from pathlib import Path

RUN_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%SZ"


def utc_now() -> datetime:
    return datetime.now(UTC)


def run_directory_name(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(RUN_TIMESTAMP_FORMAT)


def sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def new_run(root: Path, moment: datetime | None = None) -> Path:
    """Create and return a fresh run directory.

    Directory names are timestamped to the second, so two captures started
    within the same second would collide. Rather than overwrite -- which the
    append-only rule forbids -- we suffix the name and carry on.
    """
    moment = moment or utc_now()
    base = root / run_directory_name(moment)

    candidate = base
    attempt = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}.{attempt}")
        attempt += 1

    candidate.mkdir(parents=True)
    return candidate


def write_payload(run_dir: Path, relative_path: str, body: bytes) -> dict[str, object]:
    """Store ``body`` verbatim at ``relative_path`` + ``.gz`` inside ``run_dir``.

    The bytes written are exactly the bytes received -- no parse, no re-encode.
    Round-tripping through a JSON parser can quietly reorder keys, change float
    precision and alter unicode escaping, which would make the archive a record
    of our interpretation rather than of what the server actually said.

    The gzip container is written with ``mtime=0`` and an empty stored filename
    so that identical payloads produce byte-identical files -- otherwise the
    capture time and path would leak into the compressed bytes and two
    identical payloads would look different on disk.

    Returns the manifest entry describing what was written.
    """
    destination = run_dir / f"{relative_path}.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(body)

    return {
        "path": f"{relative_path}.gz",
        "bytes": len(body),
        "sha256": sha256_hex(body),
    }


def read_payload(run_dir: Path, relative_path: str) -> bytes:
    """Read back a stored payload. Accepts the path with or without ``.gz``."""
    name = relative_path if relative_path.endswith(".gz") else f"{relative_path}.gz"
    with gzip.open(run_dir / name, "rb") as handle:
        return handle.read()


def list_runs(root: Path) -> list[Path]:
    """Every run directory under ``root``, oldest first."""
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())
