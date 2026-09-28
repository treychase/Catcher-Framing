"""Shadow-zone framing grades.

Only pitches in the shadow - the band either side of the zone edge, from 67%
to 133% of the zone's half size, which is Baseball Savant's definition - are
graded, because that is the only place a catcher can change a call. Inside it:

* A ball called a strike (a stolen strike) earns credit that grows with how far
  outside the zone the pitch was: 1 point for a pitch grazing the edge, up to
  `MAX_WEIGHT` at the shadow's outer edge.
* A strike called a ball (a lost strike) costs the mirror image: 1 point at the
  edge, up to `MAX_WEIGHT` for a pitch at the shadow's inner edge.
* Correct calls score zero.

A catcher's Frame Score is the net per 100 shadow takes, and his grade places
that on the 20-80 scouting scale against the qualified catchers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def pitch_scores(df: pd.DataFrame, max_weight: float = C.MAX_WEIGHT) -> pd.DataFrame:
    """Per-pitch framing columns. Needs `zone_dist`, `is_strike` and, optionally, `p_strike`."""
    d = df["zone_dist"].to_numpy(dtype=float)
    strike = df["is_strike"].to_numpy().astype(bool)
    shadow = (d >= C.HEART_EDGE) & (d < C.SHADOW_OUTER)
    outside = d > 1.0

    w_out = 1 + (max_weight - 1) * np.clip((d - 1) / (C.SHADOW_OUTER - 1), 0, 1)
    w_in = 1 + (max_weight - 1) * np.clip((1 - d) / (1 - C.HEART_EDGE), 0, 1)

    stolen = shadow & outside & strike
    lost = shadow & ~outside & ~strike
    out = df.copy()
    out["shadow"] = shadow
    out["stolen"] = stolen
    out["lost"] = lost
    out["weight"] = np.where(outside, w_out, w_in)
    out["frame_score"] = np.where(stolen, w_out, 0.0) - np.where(lost, w_in, 0.0)
    if "p_strike" in df:
        out["strikes_added"] = strike - df["p_strike"].to_numpy(dtype=float)
    return out


def to_grade(values: pd.Series) -> pd.Series:
    """20-80 scale: 50 is the qualified mean, every 10 points one standard deviation."""
    sd = values.std(ddof=0)
    z = (values - values.mean()) / (sd if sd > 0 else 1.0)
    return (50 + 10 * z).clip(20, 80).round().astype(int)


def catcher_table(scored: pd.DataFrame, min_shadow: int = C.MIN_SHADOW_PITCHES) -> pd.DataFrame:
    g = scored.groupby(["catcher_id", "catcher"], sort=False)
    t = g.agg(
        called=("is_strike", "size"),
        shadow=("shadow", "sum"),
        stolen=("stolen", "sum"),
        lost=("lost", "sum"),
        frame_score=("frame_score", "sum"),
    )
    shadow_only = scored[scored["shadow"]].groupby(["catcher_id", "catcher"], sort=False)
    if "strikes_added" in scored:
        t["strikes_added"] = g["strikes_added"].sum()
        t["shadow_strikes_added"] = shadow_only["strikes_added"].sum()
        t["expected_shadow_strikes"] = shadow_only["p_strike"].sum()
        t["shadow_strikes"] = shadow_only["is_strike"].sum()
    t = t.reset_index()
    t["frame_score_100"] = 100 * t["frame_score"] / t["shadow"].clip(lower=1)
    t["stolen_rate"] = t["stolen"] / t["shadow"].clip(lower=1)
    t["lost_rate"] = t["lost"] / t["shadow"].clip(lower=1)
    if "strikes_added" in t:
        t["strikes_added_100"] = 100 * t["shadow_strikes_added"] / t["shadow"].clip(lower=1)
        t["framing_runs"] = C.RUNS_PER_STRIKE * t["strikes_added"]
    t["qualified"] = t["shadow"] >= min_shadow
    q = t["qualified"]
    t["grade"] = pd.NA
    t["model_grade"] = pd.NA
    if q.sum() >= 2:
        t.loc[q, "grade"] = to_grade(t.loc[q, "frame_score_100"])
        if "strikes_added_100" in t:
            t.loc[q, "model_grade"] = to_grade(t.loc[q, "strikes_added_100"])
    t = t.sort_values(["qualified", "frame_score_100"], ascending=[False, False]).reset_index(
        drop=True
    )
    t["rank"] = np.where(t["qualified"], np.arange(1, len(t) + 1), np.nan)
    return t


def heatmap(
    scored: pd.DataFrame,
    value: str,
    bins: int = C.HEAT_BINS,
    extent: float = C.HEAT_EXTENT,
    min_n: int = C.HEAT_MIN_PITCHES,
    only_shadow: bool = False,
) -> dict:
    """Mean of `value` on a bins x bins grid over the normalized zone (catcher's view).

    Rows run top to bottom (high pitches first) so the grid reads like a picture.
    Cells with fewer than `min_n` pitches are None.
    """
    d = scored[scored["shadow"]] if only_shadow else scored
    edges = np.linspace(-extent, extent, bins + 1)
    ix = np.digitize(d["nx"].to_numpy(), edges) - 1
    iz = np.digitize(d["nz"].to_numpy(), edges) - 1
    ok = (ix >= 0) & (ix < bins) & (iz >= 0) & (iz < bins)
    v = d[value].to_numpy(dtype=float)[ok]
    flat = iz[ok] * bins + ix[ok]
    s = np.bincount(flat, weights=v, minlength=bins * bins)
    n = np.bincount(flat, minlength=bins * bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    m = np.where(n >= min_n, m, np.nan).reshape(bins, bins)[::-1]
    return {
        "values": [[None if np.isnan(x) else round(float(x), 4) for x in row] for row in m],
        "counts": n.reshape(bins, bins)[::-1].astype(int).tolist(),
    }
