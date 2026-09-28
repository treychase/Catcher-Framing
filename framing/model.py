"""The predicted called-strike model.

A LightGBM classifier on location, movement, the batter's zone, the count,
handedness and the home plate umpire (as a native categorical, so each umpire
gets his own zone shape rather than a single intercept). Hyperparameters come
from a grid search scored on log loss with folds grouped by game, so pitches
from one game never sit on both sides of a split. The catcher is deliberately
not a feature: whatever the model cannot explain about a call is what framing
gets credited with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

from . import config as C

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    numeric: list[str] = field(default_factory=lambda: list(C.NUMERIC_FEATURES))
    categorical: list[str] = field(default_factory=lambda: list(C.CATEGORICAL_FEATURES))
    categories: dict[str, list] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical

    def fit(self, df: pd.DataFrame) -> FeatureSpec:
        self.categories = {c: sorted(df[c].astype(str).unique()) for c in self.categorical}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.numeric].astype(float).copy()
        for c in self.categorical:
            vals = df[c].astype(str)
            # Unseen levels (a new umpire) become missing rather than an error.
            vals = vals.where(vals.isin(self.categories[c]))
            X[c] = pd.Categorical(vals, categories=self.categories[c])
        return X

    def without(self, name: str) -> FeatureSpec:
        return FeatureSpec(
            [c for c in self.numeric if c != name],
            [c for c in self.categorical if c != name],
            {k: v for k, v in self.categories.items() if k != name},
        )


def make_estimator(**params) -> LGBMClassifier:
    base = dict(
        objective="binary",
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        cat_smooth=20,
        min_data_per_group=100,
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    base.update(params)
    return LGBMClassifier(**base)


# --------------------------------------------------------------------------
# Splits and search
# --------------------------------------------------------------------------
def split_by_game(
    df: pd.DataFrame, test_fraction: float = C.TEST_FRACTION, seed: int = C.RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["game_pk"]))
    return train_idx, test_idx


def grid_search(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: dict = C.PARAM_GRID,
    sample: int = C.GRID_SAMPLE,
    folds: int = C.CV_FOLDS,
    seed: int = C.RANDOM_STATE,
) -> tuple[dict, pd.DataFrame]:
    """Grid search on a game-grouped subsample; returns best params and the full table."""
    if len(X) > sample:
        rng = np.random.default_rng(seed)
        games = np.unique(groups)
        rng.shuffle(games)
        # Take whole games until the sample is full.
        counts = pd.Series(groups).value_counts().reindex(games).cumsum()
        chosen = set(counts.index[counts <= sample])
        mask = np.isin(groups, list(chosen))
        X, y, groups = X[mask], y[mask], groups[mask]
    search = GridSearchCV(
        make_estimator(),
        param_grid,
        scoring="neg_log_loss",
        cv=GroupKFold(n_splits=folds),
        n_jobs=1,
        refit=False,
        return_train_score=True,
    )
    search.fit(X, y, groups=groups)
    res = pd.DataFrame(search.cv_results_)
    keys = list(param_grid)
    table = pd.DataFrame({k: res[f"param_{k}"].astype(float) for k in keys})
    table["cv_log_loss"] = -res["mean_test_score"]
    table["cv_log_loss_sd"] = res["std_test_score"]
    table["train_log_loss"] = -res["mean_train_score"]
    table["fit_seconds"] = res["mean_fit_time"]
    table["rank"] = res["rank_test_score"].astype(int)
    table = table.sort_values("rank").reset_index(drop=True)
    best = {k: _as_param(search.best_params_[k]) for k in keys}
    log.info(
        "grid search on %d rows: best %s (log loss %.4f)",
        len(X),
        best,
        table["cv_log_loss"].iloc[0],
    )
    return best, table


def _as_param(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def fit(X: pd.DataFrame, y: np.ndarray, params: dict) -> LGBMClassifier:
    return make_estimator(**params).fit(X, y)


def oof_predict(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, params: dict, folds: int = C.OOF_FOLDS
) -> np.ndarray:
    """Out-of-fold strike probability for every pitch, folds grouped by game.

    Framing is the residual of this prediction, so each pitch is scored by a
    model that never saw it (or anything else from its game).
    """
    p = np.zeros(len(X))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        m = fit(X.iloc[tr], y[tr], params)
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
        log.info("oof fold %d/%d done", k + 1, folds)
    return p


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    base = float(np.mean(y))
    ll = log_loss(y, p)
    ll0 = log_loss(y, np.full_like(p, base))
    return {
        "n": int(len(y)),
        "strike_rate": base,
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(ll),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "mcfadden_r2": float(1 - ll / ll0),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    out = df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True)


def roc_points(y: np.ndarray, p: np.ndarray, n: int = 120) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return pd.DataFrame({"fpr": fpr[keep], "tpr": tpr[keep]})


def feature_importance(model: LGBMClassifier, columns: list[str]) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    out = pd.DataFrame({"feature": columns, "gain": gain})
    out["share"] = out["gain"] / out["gain"].sum()
    return out.sort_values("share", ascending=False).reset_index(drop=True)


def umpire_effects(
    model: LGBMClassifier, spec: FeatureSpec, ref: pd.DataFrame, min_games: pd.Series | None = None
) -> pd.DataFrame:
    """Each umpire's zone, holding the pitches fixed.

    Every reference pitch is scored as if each umpire had called it; the mean
    probability is that umpire's expected strike rate on an identical set of
    borderline pitches, so the ranking is about the umpire rather than about
    which pitchers he happened to draw.
    """
    X = spec.transform(ref)
    rows = []
    for ump in spec.categories["umpire"]:
        if ump == "Unknown":
            continue
        X["umpire"] = pd.Categorical([ump] * len(X), categories=spec.categories["umpire"])
        rows.append(
            {"umpire": ump, "expected_strike_rate": float(model.predict_proba(X)[:, 1].mean())}
        )
    out = pd.DataFrame(rows)
    out["vs_average"] = out["expected_strike_rate"] - out["expected_strike_rate"].mean()
    if min_games is not None:
        out = out.merge(min_games.rename("games"), left_on="umpire", right_index=True, how="left")
    return out.sort_values("vs_average", ascending=False).reset_index(drop=True)


def probability_surface(
    model: LGBMClassifier, spec: FeatureSpec, template: pd.Series, n: int = 41, extent: float = 1.6
) -> dict:
    """League-average strike probability over the plate for one typical take."""
    g = np.linspace(-extent, extent, n)
    nx, nz = np.meshgrid(g, g)
    mid = (template["sz_top"] + template["sz_bot"]) / 2
    half_h = (template["sz_top"] - template["sz_bot"]) / 2 + C.BALL_RADIUS
    grid = pd.DataFrame(
        {
            "plate_x": (nx * C.ZONE_HALF_WIDTH).ravel(),
            "plate_z": (mid + nz * half_h).ravel(),
        }
    )
    for c in spec.columns:
        if c not in grid:
            grid[c] = template[c]
    probs = []
    umps = [u for u in spec.categories["umpire"] if u != "Unknown"]
    # Average over umpires so the surface is the league's zone, not one man's.
    for ump in umps[:: max(1, len(umps) // 20)]:
        grid["umpire"] = ump
        probs.append(model.predict_proba(spec.transform(grid))[:, 1])
    p = np.mean(probs, axis=0).reshape(n, n)
    return {"axis": g.round(3).tolist(), "p": p.round(4).tolist()}
