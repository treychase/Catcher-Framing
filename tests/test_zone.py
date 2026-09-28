import numpy as np
import pandas as pd
import pytest

from framing import config as C
from framing.zone import add_zone_columns, attack_zone, normalized_coords, zone_distance


def frame(x, z, top=3.5, bot=1.5):
    return pd.DataFrame({"plate_x": x, "plate_z": z, "sz_top": top, "sz_bot": bot})


def test_centre_is_origin():
    nx, nz = normalized_coords(frame([0.0], [2.5]))
    assert nx[0] == pytest.approx(0) and nz[0] == pytest.approx(0)


def test_ball_adjusted_edges_are_one():
    half_h = 1.0 + C.BALL_RADIUS
    d = zone_distance(frame([C.ZONE_HALF_WIDTH, 0, 0], [2.5, 2.5 + half_h, 2.5 - half_h]))
    np.testing.assert_allclose(d, 1.0)


def test_distance_is_chebyshev():
    d = zone_distance(frame([0.5 * C.ZONE_HALF_WIDTH], [2.5 + 0.9 * (1 + C.BALL_RADIUS)]))
    assert d[0] == pytest.approx(0.9)


def test_zone_scales_to_batter_height():
    tall = zone_distance(frame([0.0], [3.9], top=4.0, bot=2.0))
    short = zone_distance(frame([0.0], [3.4], top=3.5, bot=1.5))
    assert tall[0] == pytest.approx(short[0])


@pytest.mark.parametrize(
    "d,label",
    [
        (0.0, "heart"),
        (0.66, "heart"),
        (0.67, "shadow"),
        (1.0, "shadow"),
        (1.32, "shadow"),
        (1.33, "chase"),
        (1.99, "chase"),
        (2.0, "waste"),
    ],
)
def test_attack_zones(d, label):
    assert attack_zone(np.array([d]))[0] == label


def test_add_zone_columns_does_not_mutate():
    df = frame([0.0, 2.0], [2.5, 2.5])
    out = add_zone_columns(df)
    assert "zone_dist" not in df
    assert out["in_zone"].tolist() == [True, False]
