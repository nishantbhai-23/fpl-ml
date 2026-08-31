# fpl-ml

Expected Points — a season-length ML engineering build using Fantasy Premier
League as the domain. Point-in-time data, a walk-forward backtest harness,
decomposed points models, a constrained optimizer, and (later) a news-retrieval
layer and tool-using agent.

**Status:** Phase 00 — point-in-time capture running on a schedule.

## Two branches

| Branch | Holds |
| --- | --- |
| `main` | Code. |
| `data` | The capture archive, append-only. Written only by CI. |

They are kept apart so a year of automated capture commits doesn't bury the
code history, and so working on `main` never means pulling first.

To read the archive locally, check it out beside the code:

```bash
git worktree add archive data
```

`archive/` is gitignored on `main`, so the two never collide.

## Capturing

```bash
uv run fpl-ml snapshot
```

Writes a timestamped run directory under `data/raw/` (scratch, gitignored).
Add `--players` for the per-player sweep — around 630 requests, roughly 20
seconds. `uv run fpl-ml runs` lists what's on disk.

In CI the same command writes to `archive/raw/` and commits to the `data`
branch: a light capture every six hours, a full sweep on Tuesday mornings once
the gameweek has finished. See
[.github/workflows/snapshot.yml](.github/workflows/snapshot.yml).

## Why capture comes before everything else

The FPL API serves only *current* state — prices, ownership, form and injury
news as of the moment you ask. There is no historical endpoint. Once a gameweek
deadline passes, what the API said beforehand is gone permanently.

So the archive is the one asset in this project that cannot be rebuilt. A bad
parser, a wrong feature, a broken model: all fixable, and re-runnable against
the archive. A missing capture is not recoverable at any price. That is why
Phase 00 is scheduled capture and nothing else, and why the raw bytes are
stored exactly as received rather than parsed on the way in.

## Historical backfill

Your own archive starts the day you run it; training needs seasons. Eleven of
them are vendored from
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
(MIT), pinned to one upstream commit:

```bash
uv run fpl-ml backfill
uv run fpl-ml panel --validate-against archive/raw/<a --players capture>
```

`backfill` vendors the CSVs into `archive/backfill/`. `panel` normalises them
into one tidy table — 254,510 rows, one per player per gameweek — and writes
`archive/panel/panel.parquet`, which is derived and therefore gitignored.

`--validate-against` audits the result against FPL's own record: every
`element-summary` capture carries `history_past`, the official per-season
totals, so third-party history can be checked against the game itself.
Currently **1,790 of 1,793 player-season totals agree (99.83%)**.

### What the backfill cannot do

It is a post-hoc reconstruction, so it carries **no availability data at all** —
no `status`, no `news`, no `chance_of_playing`. Its pre-deadline columns were
snapshotted whenever the upstream collector ran, with no as-of timestamp. It is
excellent for outcomes and lagged rolling features; it cannot tell you what was
knowable at a past deadline. That is the gap `raw/` fills, and why the two live
in separate namespaces.

Three things the data will not warn you about:

- **`total_points` is not comparable across seasons.** Defensive-contribution
  scoring arrived in 2025-26 and is worth ~11% of defender points. Model the
  components and compose them under current rules.
- **The stats behind that rule have a six-season hole.** `tackles`,
  `recoveries` and `clearances_blocks_interceptions` exist in 2016-17→2018-19
  and again from 2025-26, but not in between — so those seasons' defensive
  points cannot be reconstructed.
- **`element` is a per-season ID**, reassigned between seasons. `(season,
  element)` is the only trustworthy player key; cross-season linking needs name
  matching, which is lossy.

## Layout

```
src/fpl_ml/
  config.py     endpoints, paths, tunables
  client.py     async HTTP: backoff, retry policy, concurrency cap
  archive.py    append-only run directories, verbatim payload storage
  snapshot.py   orchestrates a capture, writes the manifest
  backfill.py   vendors pinned community history
  schema.py     column provenance: identity / pre-deadline / outcome
  panel.py      normalises seasons into one tidy table
  validate.py   audits the backfill against FPL's own history_past
  cli.py        fpl-ml snapshot | runs | backfill | panel
```

## Development

```bash
uv sync
uv run pytest
```
