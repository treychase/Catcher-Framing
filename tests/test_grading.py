import numpy as np
import pandas as pd
import pytest

from framing import config as C
from framing.grading import catcher_table, heatmap, pitch_scores, to_grade


def pitches(dist, strike, p=None):
    df = pd.DataFrame({"zone_dist": dist, "is_strike": strike})
    if p is not None:
        df["p_strike"] = p
    return df


def test_stolen_strike_weight_grows_with_distance():
    s = pitch_scores(pitches([1.0001, 1.165, 1.3299], [1, 1, 1]))
    assert s["stolen"].all()
    np.testing.assert_allclose(s["frame_score"], [1.0, 1.5, 2.0], atol=1e-2)


def test_lost_strike_penalty_grows_with_depth():
    s = pitch_scores(pitches([1.0, 0.835, 0.67], [0, 0, 0]))
    assert s["lost"].all()
    np.testing.assert_allclose(s["frame_score"], [-1.0, -1.5, -2.0], atol=1e-2)


def test_correct_calls_and_non_shadow_score_zero():
    # correct calls in the shadow, then a stolen strike in the chase and a lost one in the heart
    s = pitch_scores(pitches([0.9, 1.2, 1.5, 0.3], [1, 0, 1, 0]))
    assert (s["frame_score"] == 0).all()
    assert s["shadow"].tolist() == [True, True, False, False]


def test_max_weight_is_configurable():
    s = pitch_scores(pitches([1.3299], [1]), max_weight=3.0)
    assert s["frame_score"].iloc[0] == pytest.approx(3.0, abs=1e-2)


def test_strikes_added_is_residual():
    s = pitch_scores(pitches([1.1, 0.9], [1, 0], p=[0.3, 0.8]))
    np.testing.assert_allclose(s["strikes_added"], [0.7, -0.8])


def test_to_grade_scale():
    g = to_grade(pd.Series(np.linspace(-5, 5, 101)))
    assert g.between(20, 80).all()
    assert abs(g.mean() - 50) < 1
    assert to_grade(pd.Series([1.0, 1.0])).tolist() == [50, 50]


def test_catcher_table(zoned):
    df = zoned.assign(p_strike=0.5)
    t = catcher_table(pitch_scores(df), min_shadow=50)
    assert t["catcher_id"].is_unique
    q = t[t["qualified"]]
    assert q["frame_score_100"].is_monotonic_decreasing
    assert q["grade"].astype(int).between(20, 80).all()
    assert (t["shadow"] <= t["called"]).all()
    assert q["rank"].tolist() == list(range(1, len(q) + 1))
    assert np.allclose(t["framing_runs"], C.RUNS_PER_STRIKE * t["strikes_added"])


def test_unqualified_catchers_get_no_grade(zoned):
    t = catcher_table(pitch_scores(zoned.assign(p_strike=0.5)), min_shadow=10**9)
    assert not t["qualified"].any()
    assert t["grade"].isna().all()


def test_grades_recover_true_skill(zoned):
    t = catcher_table(pitch_scores(zoned), min_shadow=50)
    truth = pd.Series(zoned.attrs["true_catcher_skill"])
    r = t.set_index("catcher_id")["frame_score_100"].corr(truth)
    assert r > 0.7


def test_heatmap_orientation_and_masking():
    df = pd.DataFrame(
        {
            "nx": [0.1, 0.1, 5.0],
            "nz": [1.4, 1.4, 0.0],
            "v": [2.0, 4.0, 9.0],
            "shadow": [True, True, True],
        }
    )
    h = heatmap(df, "v", bins=6, extent=1.5, min_n=2)
    vals = np.array([[np.nan if x is None else x for x in r] for r in h["values"]], dtype=float)
    assert vals.shape == (6, 6)
    assert vals[0, 3] == pytest.approx(3.0)  # high pitch lands in the top row
    assert np.isnan(vals).sum() == 35  # everything else masked; out-of-range dropped
    h = heatmap(df, "v", bins=6, extent=1.5, min_n=3)
    assert all(x is None for r in h["values"] for x in r)
