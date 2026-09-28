"""Checking the rule-based Frame Score against the strike probability model.

The Frame Score is transparent but blunt: it knows where a pitch was, not who
called it or what it looked like on the way in. The model knows both. If the
distance weights are sensible, three things should hold:

1. Pitch level - the farther outside the zone a stolen strike is, the lower the
   model's strike probability for it, so the extra credit is paid for pitches
   that really were less likely to be called.
2. Catcher level - Frame Score per 100 should rank catchers much as the model's
   strikes added per 100 shadow takes does.
3. Stability - a skill should repeat from one season to the next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .grading import catcher_table


def _corr(a: pd.Series, b: pd.Series) -> dict:
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(float), b[ok].astype(float)
    if len(a) < 3:
        return {"n": int(len(a)), "pearson": None, "spearman": None}
    return {
        "n": int(len(a)),
        "pearson": float(a.corr(b)),
        "spearman": float(a.rank().corr(b.rank())),
    }


def distance_profile(scored: pd.DataFrame, n_bins: int = 16) -> pd.DataFrame:
    """Called-strike rate, model probability and grade weight across the shadow."""
    s = scored[scored["shadow"]]
    edges = np.linspace(C.HEART_EDGE, C.SHADOW_OUTER, n_bins + 1)
    idx = np.clip(np.digitize(s["zone_dist"], edges) - 1, 0, n_bins - 1)
    out = (
        s.assign(bin=idx)
        .groupby("bin")
        .agg(
            n=("is_strike", "size"),
            strike_rate=("is_strike", "mean"),
            p_strike=("p_strike", "mean"),
            weight=("weight", "mean"),
        )
    )
    out["distance"] = ((edges[:-1] + edges[1:]) / 2)[out.index]
    return out.reset_index(drop=True)


def pitch_level(scored: pd.DataFrame) -> dict:
    s = scored[scored["shadow"]]
    out_zone = s[s["zone_dist"] > 1]
    in_zone = s[s["zone_dist"] <= 1]
    stolen = out_zone[out_zone["is_strike"] == 1]
    lost = in_zone[in_zone["is_strike"] == 0]
    res = {
        "shadow_pitches": int(len(s)),
        "stolen": int(len(stolen)),
        "lost": int(len(lost)),
        "p_strike_stolen": float(stolen["p_strike"].mean()),
        "p_strike_out_of_zone_shadow": float(out_zone["p_strike"].mean()),
        "p_strike_lost": float(lost["p_strike"].mean()),
        "p_strike_in_zone_shadow": float(in_zone["p_strike"].mean()),
        # Credit should rise as the probability of the call falls.
        "weight_vs_surprise_stolen": _corr(stolen["weight"], 1 - stolen["p_strike"]),
        "weight_vs_surprise_lost": _corr(lost["weight"], lost["p_strike"]),
    }
    return res


def catcher_level(table: pd.DataFrame) -> dict:
    q = table[table["qualified"]]
    top = set(q.nlargest(10, "frame_score_100")["catcher_id"])
    top_m = set(q.nlargest(10, "strikes_added_100")["catcher_id"])
    bot = set(q.nsmallest(10, "frame_score_100")["catcher_id"])
    bot_m = set(q.nsmallest(10, "strikes_added_100")["catcher_id"])
    return {
        "frame_score_vs_model": _corr(q["frame_score_100"], q["strikes_added_100"]),
        "frame_score_vs_model_all_takes": _corr(
            q["frame_score_100"], 100 * q["strikes_added"] / q["called"]
        ),
        "top10_overlap": len(top & top_m),
        "bottom10_overlap": len(bot & bot_m),
    }


def stability(
    scored: pd.DataFrame, seasons=C.SEASONS, min_shadow: int = C.MIN_SHADOW_PITCHES_SEASON
) -> dict:
    if len(seasons) < 2:
        return {}
    a, b = seasons[0], seasons[1]
    ta = catcher_table(scored[scored["game_year"] == a], min_shadow)
    tb = catcher_table(scored[scored["game_year"] == b], min_shadow)
    ta, tb = ta[ta["qualified"]], tb[tb["qualified"]]
    m = ta.merge(tb, on="catcher_id", suffixes=("_a", "_b"))
    res = {
        "seasons": [int(a), int(b)],
        "frame_score": _corr(m["frame_score_100_a"], m["frame_score_100_b"]),
    }
    if "strikes_added_100_a" in m:
        res["model"] = _corr(m["strikes_added_100_a"], m["strikes_added_100_b"])
    return res
