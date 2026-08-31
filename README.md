# fpl-ml archive

Point-in-time captures of the Fantasy Premier League API. **This branch holds
data only** — the code that produces it lives on `main`.

Written by the `Snapshot` workflow on `main`: a light capture
(`bootstrap-static` + `fixtures`) every six hours, and a full sweep including
per-player summaries on Tuesday mornings, after the gameweek has finished.

## Two namespaces, deliberately kept apart

| Path | What it is | Point-in-time? |
| --- | --- | --- |
| `raw/` | Our own API captures | **Yes** — we know exactly when each byte was observed |
| `backfill/` | Vendored community history | **No** — post-hoc reconstruction |

Never merge them without carrying a provenance column. A dataset is only as
trustworthy as its weakest source, and once mixed you can no longer tell which
rows carry point-in-time guarantees and which don't.

## Layout

```
raw/2026-08-30T20-01-30Z/
  manifest.json              what was fetched, with statuses, sizes, SHA-256s
  bootstrap-static.json.gz   players, teams, gameweeks, prices, ownership
  fixtures.json.gz           fixtures and difficulty ratings
  element-summary/<id>.json.gz   per-player history (full sweeps only)

backfill/
  manifest.json              upstream repo, pinned SHA, per-file hashes
  <season>/gws/merged_gw.csv one row per player per gameweek
  <season>/{fixtures,teams,players_raw}.csv
```

Directory names under `raw/` are the UTC capture time. Payloads are stored as
the exact bytes the API returned, gzipped.

`backfill/` is vendored from
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
(MIT), pinned to one upstream commit so history cannot shift underneath a
backtest. It has **no player availability data at all** — no `status`, no
`news`, no `chance_of_playing` — which is exactly the gap `raw/` exists to fill.

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
