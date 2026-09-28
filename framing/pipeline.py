"""End to end: data -> strike model -> shadow grades -> validation -> outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import grading, model, plots, validate
from .zone import add_zone_columns

log = logging.getLogger(__name__)


def _records(df: pd.DataFrame, cols: list[str], digits: int = 4) -> list[dict]:
    out = df[cols].copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")


LEADER_COLS = [
    "catcher_id",
    "catcher",
    "called",
    "shadow",
    "stolen",
    "lost",
    "frame_score",
    "frame_score_100",
    "grade",
    "strikes_added_100",
    "strikes_added",
    "framing_runs",
    "model_grade",
    "stolen_rate",
    "lost_rate",
]


def run(
    df: pd.DataFrame,
    out_dir: Path | str = "results",
    seasons=C.SEASONS,
    param_grid: dict | None = None,
    grid_sample: int = C.GRID_SAMPLE,
    cv_folds: int = C.CV_FOLDS,
    oof_folds: int = C.OOF_FOLDS,
    min_shadow: int = C.MIN_SHADOW_PITCHES,
    min_shadow_season: int = C.MIN_SHADOW_PITCHES_SEASON,
    synthetic: bool = False,
    figures: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    param_grid = param_grid or C.PARAM_GRID

    df = add_zone_columns(df).reset_index(drop=True)
    spec = model.FeatureSpec().fit(df)
    X = spec.transform(df)
    y = df["is_strike"].to_numpy()
    groups = df["game_pk"].to_numpy()

    # ---- model selection and held-out diagnostics -------------------------
    tr, te = model.split_by_game(df)
    best, grid_table = model.grid_search(
        X.iloc[tr], y[tr], groups[tr], param_grid, grid_sample, cv_folds
    )
    final = model.fit(X.iloc[tr], y[tr], best)
    p_test = final.predict_proba(X.iloc[te])[:, 1]
    test_metrics = model.metrics(y[te], p_test)
    shadow_te = (df.loc[te, "zone_dist"] >= C.HEART_EDGE) & (
        df.loc[te, "zone_dist"] < C.SHADOW_OUTER
    )
    shadow_metrics = model.metrics(y[te][shadow_te.to_numpy()], p_test[shadow_te.to_numpy()])

    spec_nu = spec.without("umpire")
    no_ump = model.fit(spec_nu.transform(df.iloc[tr]), y[tr], best)
    base_metrics = model.metrics(y[te], no_ump.predict_proba(spec_nu.transform(df.iloc[te]))[:, 1])

    cal = model.calibration_table(y[te], p_test)
    roc = model.roc_points(y[te], p_test)
    imp = model.feature_importance(final, spec.columns)
    shadow_ref = df[(df["zone_dist"] >= C.HEART_EDGE) & (df["zone_dist"] < C.SHADOW_OUTER)]
    ref = shadow_ref.sample(min(len(shadow_ref), 20_000), random_state=C.RANDOM_STATE)
    ump_games = df.groupby("umpire")["game_pk"].nunique()
    umps = model.umpire_effects(final, spec, ref, ump_games)
    umps = umps[umps["games"] >= max(5, int(ump_games.quantile(0.1)))].reset_index(drop=True)
    template = df.iloc[0].copy()
    template[["sz_top", "sz_bot"]] = [df["sz_top"].median(), df["sz_bot"].median()]
    template[["pfx_x", "pfx_z", "balls", "strikes"]] = [0.0, df["pfx_z"].median(), 0, 0]
    template[["stand", "p_throws"]] = ["R", "R"]
    surface = model.probability_surface(final, spec, template)

    # ---- framing ---------------------------------------------------------
    df["p_strike"] = model.oof_predict(X, y, groups, best, oof_folds)
    oof_metrics = model.metrics(y, df["p_strike"].to_numpy())
    scored = grading.pitch_scores(df)

    scopes = {"all": (scored, min_shadow)}
    for s in seasons:
        part = scored[scored["game_year"] == s]
        if len(part):
            scopes[str(s)] = (part, min_shadow_season)
    tables = {k: grading.catcher_table(v, m) for k, (v, m) in scopes.items()}
    main = tables["all"]

    prof = validate.distance_profile(scored)
    validation = {
        "pitch": validate.pitch_level(scored),
        "catcher": validate.catcher_level(main),
        "stability": validate.stability(scored, tuple(int(s) for s in seasons), min_shadow_season),
    }

    # Heatmaps: every qualified catcher, every scope.
    heat = {}
    league = {}
    for k, (part, _) in scopes.items():
        league[k] = grading.heatmap(part, "frame_score", only_shadow=True, min_n=30)["values"]
        heat[k] = {}
        for row in tables[k][tables[k]["qualified"]].itertuples():
            mine = part[part["catcher_id"] == row.catcher_id]
            heat[k][str(row.catcher_id)] = {
                "fs": grading.heatmap(mine, "frame_score", only_shadow=True)["values"],
                "sa": grading.heatmap(mine, "strikes_added")["values"],
            }

    # ---- files -------------------------------------------------------------
    grid_table.to_csv(out_dir / "grid_search.csv", index=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    umps.to_csv(out_dir / "umpire_zones.csv", index=False)
    cal.to_csv(out_dir / "calibration.csv", index=False)
    prof.to_csv(out_dir / "shadow_distance_profile.csv", index=False)
    for k, t in tables.items():
        t.to_csv(out_dir / f"catcher_framing_{k}.csv", index=False)

    metrics = {
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "synthetic": synthetic,
        "seasons": [int(s) for s in seasons],
        "data": {
            "called_pitches": int(len(df)),
            "games": int(df["game_pk"].nunique()),
            "catchers": int(df["catcher_id"].nunique()),
            "umpires": int(df["umpire"].nunique()),
            "first_date": str(df["game_date"].min()),
            "last_date": str(df["game_date"].max()),
            "shadow_pitches": int(scored["shadow"].sum()),
            "unknown_umpire_share": float((df["umpire"] == "Unknown").mean()),
            "by_season": {str(k): int(v) for k, v in df.groupby("game_year").size().items()},
        },
        "model": {
            "features": spec.columns,
            "best_params": best,
            "test": test_metrics,
            "test_shadow": shadow_metrics,
            "test_without_umpire": base_metrics,
            "oof": oof_metrics,
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
        },
        "validation": validation,
        "grading": {
            "max_weight": C.MAX_WEIGHT,
            "heart_edge": C.HEART_EDGE,
            "shadow_outer": C.SHADOW_OUTER,
            "min_shadow": min_shadow,
            "min_shadow_season": min_shadow_season,
            "runs_per_strike": C.RUNS_PER_STRIKE,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if figures:
        plots.calibration(cal, fig_dir / "calibration.png")
        plots.roc(roc, test_metrics["auc"], fig_dir / "roc.png")
        plots.importance(imp, fig_dir / "feature_importance.png")
        plots.grid(grid_table, fig_dir / "grid_search.png")
        plots.umpires(umps, fig_dir / "umpire_zones.png")
        plots.probability_surface(surface, fig_dir / "strike_probability.png")
        plots.distance_profile(prof, fig_dir / "shadow_distance_profile.png")
        r = validation["catcher"]["frame_score_vs_model"]["pearson"] or 0.0
        plots.validation_scatter(main, r, fig_dir / "grade_vs_model.png")
        q = main[main["qualified"]]
        L = np.array(league["all"], dtype=float)

        def rel(cid):
            m = np.array(heat["all"][str(cid)]["fs"], dtype=float)
            return [[None if np.isnan(v) else float(v) for v in row] for row in m - L]

        rels = {str(c): rel(c) for c in q["catcher_id"]}
        vals = [abs(v) for g in rels.values() for row in g for v in row if v is not None]
        vmax = float(np.percentile(vals, 95)) if vals else 1.0
        for label, rows in (("leaders", q.head(5)), ("laggards", q.tail(5).iloc[::-1])):
            maps = [(r.catcher, rels[str(r.catcher_id)]) for r in rows.itertuples()]
            if maps:
                plots.catcher_heatmaps(
                    maps,
                    f"Shadow-zone frame score vs league: {label}",
                    fig_dir / f"heatmaps_{label}.png",
                    vmax,
                )

    payload = {
        "meta": {k: metrics[k] for k in ("built", "synthetic", "seasons", "data", "grading")},
        "model": {
            **metrics["model"],
            "grid": _records(grid_table, list(grid_table.columns)),
            "calibration": _records(cal, list(cal.columns)),
            "roc": _records(roc, ["fpr", "tpr"], 3),
            "importance": _records(imp, ["feature", "share"]),
            "umpires": _records(umps, ["umpire", "expected_strike_rate", "vs_average", "games"]),
            "surface": surface,
        },
        "validation": {**validation, "profile": _records(prof, list(prof.columns))},
        "leaders": {k: _records(t[t["qualified"]], LEADER_COLS, 3) for k, t in tables.items()},
        "heat": {"extent": C.HEAT_EXTENT, "bins": C.HEAT_BINS, "league": league, "catchers": heat},
    }
    log.info("done: %d qualified catchers", int(main["qualified"].sum()))
    return payload
