"""Command line entry point: ``fpl-ml snapshot``."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from . import archive, config, snapshot


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

    return parser


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

    # A failed capture is worth a non-zero exit so a scheduler can notice.
    return 1 if counts["failed"] else 0


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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
