import numpy as np
import pandas as pd
import pytest
from conftest import QUICK_GRID

from framing import model


@pytest.fixture(scope="module")
def fitted(synth):
    spec = model.FeatureSpec().fit(synth)
    X = spec.transform(synth)
    y = synth["is_strike"].to_numpy()
    tr, te = model.split_by_game(synth)
    m = model.fit(
        X.iloc[tr],
        y[tr],
        {"num_leaves": 15, "n_estimators": 150, "learning_rate": 0.1, "min_child_samples": 40},
    )
    return spec, X, y, tr, te, m


def test_feature_spec_handles_unseen_categories(synth):
    spec = model.FeatureSpec().fit(synth)
    new = synth.head(3).assign(umpire="Never Seen")
    X = spec.transform(new)
    assert X["umpire"].isna().all()
    assert list(X.columns) == spec.columns
    assert "umpire" not in spec.without("umpire").columns


def test_split_by_game_is_disjoint(synth):
    tr, te = model.split_by_game(synth, test_fraction=0.25)
    assert set(synth["game_pk"].iloc[tr]).isdisjoint(synth["game_pk"].iloc[te])
    assert 0.15 < len(te) / len(synth) < 0.35


def test_grid_search_returns_ranked_table(synth):
    spec = model.FeatureSpec().fit(synth)
    X = spec.transform(synth)
    best, table = model.grid_search(
        X,
        synth["is_strike"].to_numpy(),
        synth["game_pk"].to_numpy(),
        QUICK_GRID,
        sample=8_000,
        folds=2,
    )
    assert len(table) == 2 and table["rank"].iloc[0] == 1
    assert table["cv_log_loss"].is_monotonic_increasing
    assert set(best) == set(QUICK_GRID)
    assert isinstance(best["num_leaves"], int)


def test_model_is_accurate_and_calibrated(fitted):
    spec, X, y, tr, te, m = fitted
    p = m.predict_proba(X.iloc[te])[:, 1]
    met = model.metrics(y[te], p)
    assert met["auc"] > 0.95
    assert met["mcfadden_r2"] > 0.5
    cal = model.calibration_table(y[te], p, bins=10)
    big = cal[cal["n"] > 100]
    assert (big["predicted"] - big["observed"]).abs().max() < 0.1


def test_metrics_values():
    y = np.array([0, 1, 1, 0])
    met = model.metrics(y, np.array([0.1, 0.9, 0.8, 0.2]))
    assert met["auc"] == 1.0 and met["accuracy"] == 1.0 and met["strike_rate"] == 0.5
    assert met["brier"] == pytest.approx(np.mean([0.01, 0.01, 0.04, 0.04]))


def test_umpire_effects_recover_zone_size(fitted, synth):
    spec, X, y, tr, te, m = fitted
    ref = synth[(synth["plate_x"].abs() < 1.2)].sample(3000, random_state=0)
    eff = model.umpire_effects(m, spec, ref)
    assert eff["vs_average"].abs().sum() > 0
    assert eff["vs_average"].mean() == pytest.approx(0, abs=1e-9)
    assert eff["vs_average"].is_monotonic_decreasing


def test_umpire_feature_helps(fitted, synth):
    spec, X, y, tr, te, m = fitted
    params = {"num_leaves": 15, "n_estimators": 150, "learning_rate": 0.1, "min_child_samples": 40}
    nu = spec.without("umpire")
    m0 = model.fit(nu.transform(synth.iloc[tr]), y[tr], params)
    with_u = model.metrics(y[te], m.predict_proba(X.iloc[te])[:, 1])["log_loss"]
    without = model.metrics(y[te], m0.predict_proba(nu.transform(synth.iloc[te]))[:, 1])["log_loss"]
    assert with_u < without


def test_oof_predict(synth):
    spec = model.FeatureSpec().fit(synth)
    X = spec.transform(synth)
    p = model.oof_predict(
        X,
        synth["is_strike"].to_numpy(),
        synth["game_pk"].to_numpy(),
        {"num_leaves": 7, "n_estimators": 40},
        folds=2,
    )
    assert p.shape == (len(synth),) and ((p > 0) & (p < 1)).all()


def test_probability_surface_shape(fitted, synth):
    spec, X, y, tr, te, m = fitted
    t = synth.iloc[0].copy()
    s = model.probability_surface(m, spec, t, n=11, extent=1.5)
    p = np.array(s["p"])
    assert p.shape == (11, 11)
    assert p[5, 5] > 0.9 and p[0, 0] < 0.1  # middle is a strike, the corner is not


def test_feature_importance_sums_to_one(fitted):
    spec, X, y, tr, te, m = fitted
    imp = model.feature_importance(m, spec.columns)
    assert imp["share"].sum() == pytest.approx(1)
    assert set(imp["feature"].head(2)) == {"plate_x", "plate_z"}
    assert isinstance(imp, pd.DataFrame)
