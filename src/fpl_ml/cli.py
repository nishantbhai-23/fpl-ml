"""Command line entry point: ``fpl-ml snapshot``."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from . import archive, backfill, config, snapshot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpl-ml",
        description="Point-in-time capture of the Fantasy Premier League API.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    snap = subcommands.add_parser(
        "snapshot",
        help="Capture the current API state into a new immutable run directory.",
    )
    snap.add_argument(
        "--players",
        action="store_true",
        help="Also fetch per-player summaries (~700 requests, roughly 20s).",
    )
    snap.add_argument(
        "--data-root",
        type=Path,
        default=config.RAW_ROOT,
        help=f"Where run directories are written (default: {config.RAW_ROOT}).",
    )

    runs = subcommands.add_parser("runs", help="List captures already on disk.")
    runs.add_argument("--data-root", type=Path, default=config.RAW_ROOT)

    back = subcommands.add_parser(
        "backfill",
        help="Vendor historical seasons from the pinned community archive.",
    )
    back.add_argument(
        "--dest",
        type=Path,
        default=config.BACKFILL_ROOT,
        help=f"Where vendored CSVs are written (default: {config.BACKFILL_ROOT}).",
    )
    back.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help="Fetch only this season; repeatable. Default: all.",
    )

    pan = subcommands.add_parser(
        "panel",
        help="Normalise vendored seasons into one tidy panel table.",
    )
    pan.add_argument("--source", type=Path, default=config.BACKFILL_ROOT)
    pan.add_argument("--dest", type=Path, default=config.ARCHIVE_ROOT / "panel")
    pan.add_argument(
        "--validate-against",
        type=Path,
        default=None,
        help="A --players capture directory to audit season totals against.",
    )

    return parser


def _panel(args: argparse.Namespace) -> int:
    # Imported here, not at module scope: these pull in polars, and the
    # scheduled capture path should not depend on a dataframe library it never
    # uses. Capture is the one step whose failure is unrecoverable.
    from . import panel, validate

    frame, summary = panel.build(args.source)

    if args.validate_against:
        summary["validation"] = validate.compare(frame, args.validate_against)

    target = panel.write(frame, summary, args.dest)
    print(target)
    print(f"  rows       {summary['rows']:,} x {summary['columns']} columns")
    print(f"  seasons    {len([s for s, i in summary['seasons'].items() if i['rows']])}")
    classes = summary["column_classes"]
    print(
        f"  classified identity={len(classes['identity'])} "
        f"pre_deadline={len(classes['pre_deadline'])} "
        f"outcome={len(classes['outcome'])}"
    )
    v = summary.get("validation")
    if v and v.get("matched"):
        print(
            f"  validated  {v['agreed']:,}/{v['matched']:,} season totals match "
            f"FPL's own record ({v['agreement_rate'] * 100:.2f}%)"
        )
    return 0


def _backfill(args: argparse.Namespace) -> int:
    seasons = tuple(args.seasons) if args.seasons else backfill.SEASONS
    manifest = backfill.run(args.dest, seasons=seasons)
    counts = manifest["counts"]

    print(f"{args.dest}")
    print(f"  upstream   {backfill.UPSTREAM_REPO} @ {backfill.UPSTREAM_SHA[:12]}")
    print(f"  fetched    {counts['ok']} files, {counts['missing']} missing")
    for entry in manifest["entries"]:
        if entry.get("path") is None:
            print(f"  MISSING    {entry['season']}/{entry['file']}")

    # Missing files are expected for the earliest seasons, so this is not a
    # failure -- only a total washout is.
    return 0 if counts["ok"] else 1


def _snapshot(args: argparse.Namespace) -> int:
    run_dir = asyncio.run(
        snapshot.run(root=args.data_root, include_players=args.players)
    )
    manifest = snapshot.read_manifest(run_dir)
    counts = manifest["counts"]
    gameweek = manifest["gameweek"]

    print(run_dir)
    print(f"  captured   {counts['ok']} ok, {counts['failed']} failed")
    print(f"  gameweek   current={gameweek['current_event']} next={gameweek['next_event']}")
    print(f"  deadline   {gameweek['next_deadline']}")

    for entry in manifest["entries"]:
        if entry.get("path") is None:
            print(f"  FAILED     {entry['url']} ({entry.get('error')})")

    # Exit non-zero only when the capture is worthless, not merely imperfect --
    # a scheduler that goes red for one blipped player request out of 630 will
    # train you to ignore it, which is worse than no alerting at all.
    return 0 if snapshot.is_usable(manifest) else 1


def _runs(args: argparse.Namespace) -> int:
    directories = archive.list_runs(args.data_root)
    if not directories:
        print(f"no captures under {args.data_root}")
        return 0

    for directory in directories:
        try:
            manifest = snapshot.read_manifest(directory)
        except (OSError, ValueError):
            print(f"{directory.name}  (no readable manifest)")
            continue
        counts = manifest["counts"]
        gameweek = manifest["gameweek"]
        print(
            f"{directory.name}  gw={gameweek['current_event']}  "
            f"ok={counts['ok']}  failed={counts['failed']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "snapshot":
        return _snapshot(args)
    if args.command == "runs":
        return _runs(args)
    if args.command == "backfill":
        return _backfill(args)
    if args.command == "panel":
        return _panel(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
