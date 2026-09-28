import json

import pandas as pd
from conftest import QUICK_GRID

from framing import pipeline, report
from framing.cli import main
from framing.synthetic import make_called_pitches


def test_end_to_end(tmp_path):
    df = make_called_pitches(n=24_000, n_catchers=14, n_umpires=16, seed=3)
    payload = pipeline.run(
        df,
        tmp_path,
        (2025, 2026),
        param_grid=QUICK_GRID,
        grid_sample=10_000,
        cv_folds=2,
        oof_folds=2,
        min_shadow=150,
        min_shadow_season=60,
        synthetic=True,
    )
    for f in [
        "metrics.json",
        "grid_search.csv",
        "catcher_framing_all.csv",
        "catcher_framing_2025.csv",
        "umpire_zones.csv",
        "figures/calibration.png",
        "figures/heatmaps_leaders.png",
        "figures/grade_vs_model.png",
    ]:
        assert (tmp_path / f).exists(), f
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["synthetic"] is True and m["model"]["test"]["auc"] > 0.9

    leaders = payload["leaders"]["all"]
    assert leaders and all(r["grade"] is not None for r in leaders)
    ids = {str(r["catcher_id"]) for r in leaders}
    assert ids == set(payload["heat"]["catchers"]["all"])
    grid = payload["heat"]["catchers"]["all"][next(iter(ids))]["fs"]
    assert len(grid) == payload["heat"]["bins"] == len(grid[0])

    # The framing measure should track the skill the generator planted.
    t = pd.read_csv(tmp_path / "catcher_framing_all.csv").set_index("catcher_id")
    truth = pd.Series(df.attrs["true_catcher_skill"])
    assert t["strikes_added_100"].corr(truth) > 0.6
    assert t["frame_score_100"].corr(truth) > 0.6

    html = report.render(payload)
    assert "Synthetic data" in html and len(html) > 10_000


def test_cli_synthetic(tmp_path, monkeypatch):
    import framing.synthetic as S

    real = S.make_called_pitches
    monkeypatch.setattr(
        S, "make_called_pitches", lambda **kw: real(n=15_000, n_catchers=10, n_umpires=12, seed=5)
    )
    site = tmp_path / "site" / "page.html"
    rc = main(
        ["run", "--synthetic", "--quick", "--out", str(tmp_path / "res"), "--site", str(site)]
    )
    assert rc == 0 and site.exists()
