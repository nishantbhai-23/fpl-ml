# fpl-ml archive

Point-in-time captures of the Fantasy Premier League API. **This branch holds
data only** — the code that produces it lives on `main`.

Written by the `Snapshot` workflow on `main`: a light capture
(`bootstrap-static` + `fixtures`) every six hours, and a full sweep including
per-player summaries on Tuesday mornings, after the gameweek has finished.

## Layout

```
raw/2026-08-30T20-01-30Z/
  manifest.json              what was fetched, with statuses, sizes, SHA-256s
  bootstrap-static.json.gz   players, teams, gameweeks, prices, ownership
  fixtures.json.gz           fixtures and difficulty ratings
  element-summary/<id>.json.gz   per-player history (full sweeps only)
```

Directory names are the UTC capture time. Payloads are stored as the exact
bytes the API returned, gzipped.

## Append-only

Nothing here is ever edited or deleted. The FPL API serves only current state,
so once a deadline passes there is no way to recover what it said beforehand —
not from the API, not from anywhere. A wrong parser can be fixed and re-run
against this archive; a missing capture cannot be recovered at any price.

## Reading it

```bash
git worktree add archive data
```

Then from `main`:

```python
from pathlib import Path
from fpl_ml import archive

runs = archive.list_runs(Path("archive/raw"))
bootstrap = archive.read_payload(runs[-1], "bootstrap-static.json")
```
