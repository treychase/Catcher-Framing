import numpy as np

from framing import validate
from framing.grading import catcher_table, pitch_scores


def scored(zoned, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    # A stand-in "model": a smooth edge plus noise, no catcher knowledge.
    p = 1 / (1 + np.exp(14 * (zoned["zone_dist"] - 1))) + rng.normal(0, noise, len(zoned))
    return pitch_scores(zoned.assign(p_strike=np.clip(p, 0.001, 0.999)))


def test_pitch_level(zoned):
    res = validate.pitch_level(scored(zoned))
    assert res["p_strike_stolen"] < 0.5 < res["p_strike_lost"]
    assert res["p_strike_out_of_zone_shadow"] < res["p_strike_in_zone_shadow"]
    # farther out = less likely = more credit
    assert res["weight_vs_surprise_stolen"]["pearson"] > 0.5


def test_catcher_level_agrees(zoned):
    t = catcher_table(scored(zoned), min_shadow=50)
    res = validate.catcher_level(t)
    assert res["frame_score_vs_model"]["pearson"] > 0.8
    assert 0 <= res["top10_overlap"] <= 10


def test_distance_profile(zoned):
    prof = validate.distance_profile(scored(zoned), n_bins=8)
    assert len(prof) == 8
    assert prof["strike_rate"].iloc[0] > prof["strike_rate"].iloc[-1]
    assert prof["distance"].is_monotonic_increasing


def test_stability(zoned):
    res = validate.stability(scored(zoned), seasons=(2025, 2026), min_shadow=30)
    assert res["seasons"] == [2025, 2026]
    assert res["frame_score"]["n"] > 3
    assert validate.stability(scored(zoned), seasons=(2025,)) == {}


def test_corr_handles_small_samples():
    import pandas as pd

    out = validate._corr(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
    assert out["pearson"] is None and out["n"] == 2
