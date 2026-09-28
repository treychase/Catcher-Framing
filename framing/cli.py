"""Command line entry point: `python -m framing run`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import config as C
from . import pipeline, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="framing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fetch, model, grade and build the page")
    r.add_argument("--seasons", type=int, nargs="+", default=list(C.SEASONS))
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out", default="results")
    r.add_argument("--site", default="site/catcher-framing.html")
    r.add_argument(
        "--synthetic",
        action="store_true",
        help="use generated data instead of Statcast (dry run, never published)",
    )
    r.add_argument(
        "--quick", action="store_true", help="tiny grid and fewer folds, for smoke tests"
    )
    f = sub.add_parser("fetch", help="only pull and cache the data")
    f.add_argument("--seasons", type=int, nargs="+", default=list(C.SEASONS))
    f.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.cmd == "fetch":
        from .data import build_dataset

        df = build_dataset(args.seasons, args.data_dir)
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(Path(args.data_dir) / "called_pitches.parquet", index=False)
        return 0

    if args.synthetic:
        from .synthetic import make_called_pitches

        df = make_called_pitches(n=120_000, n_catchers=40, n_umpires=60)
    else:
        cached = Path(args.data_dir) / "called_pitches.parquet"
        if cached.exists():
            df = pd.read_parquet(cached)
        else:
            from .data import build_dataset

            df = build_dataset(args.seasons, args.data_dir)
            df.to_parquet(cached, index=False)

    kw = {}
    if args.quick:
        kw = dict(
            param_grid={
                "num_leaves": [15, 31],
                "learning_rate": [0.1],
                "n_estimators": [100],
                "min_child_samples": [50],
            },
            cv_folds=2,
            oof_folds=2,
        )
    payload = pipeline.run(df, args.out, tuple(args.seasons), synthetic=args.synthetic, **kw)
    report.write(payload, args.site)
    print(json.dumps({k: payload["model"]["test"][k] for k in ("auc", "log_loss", "brier")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
