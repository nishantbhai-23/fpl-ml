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

## Layout

```
src/fpl_ml/
  config.py     endpoints, paths, tunables
  client.py     async HTTP: backoff, retry policy, concurrency cap
  archive.py    append-only run directories, verbatim payload storage
  snapshot.py   orchestrates a capture, writes the manifest
  cli.py        fpl-ml snapshot | runs
```

## Development

```bash
uv sync
uv run pytest
```
