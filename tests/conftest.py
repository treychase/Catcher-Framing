import pytest

from framing.synthetic import make_called_pitches
from framing.zone import add_zone_columns

QUICK_GRID = {
    "num_leaves": [7, 15],
    "learning_rate": [0.1],
    "n_estimators": [60],
    "min_child_samples": [40],
}


@pytest.fixture(scope="session")
def synth():
    return make_called_pitches(n=30_000, n_catchers=16, n_umpires=20, seed=11)


@pytest.fixture(scope="session")
def zoned(synth):
    return add_zone_columns(synth)
