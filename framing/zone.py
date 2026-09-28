"""Strike zone geometry: normalized coordinates and Savant-style attack zones."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def normalized_coords(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Location in zone units: 0 is the centre, +-1 is the ball-adjusted edge.

    The vertical axis uses the batter's own sz_top / sz_bot, so a pitch at the
    knees of a short hitter and one at the knees of a tall one land together.
    """
    x = df["plate_x"].to_numpy(dtype=float)
    z = df["plate_z"].to_numpy(dtype=float)
    top = df["sz_top"].to_numpy(dtype=float)
    bot = df["sz_bot"].to_numpy(dtype=float)
    mid = (top + bot) / 2.0
    half_h = (top - bot) / 2.0 + C.BALL_RADIUS
    nx = x / C.ZONE_HALF_WIDTH
    nz = (z - mid) / half_h
    return nx, nz


def zone_distance(df: pd.DataFrame) -> np.ndarray:
    """Chebyshev distance in zone units; <= 1 is a rulebook strike."""
    nx, nz = normalized_coords(df)
    return np.maximum(np.abs(nx), np.abs(nz))


def attack_zone(distance: np.ndarray) -> np.ndarray:
    """Heart / shadow / chase / waste, as Baseball Savant draws them."""
    d = np.asarray(distance, dtype=float)
    out = np.full(d.shape, "waste", dtype=object)
    out[d < C.CHASE_OUTER] = "chase"
    out[d < C.SHADOW_OUTER] = "shadow"
    out[d < C.HEART_EDGE] = "heart"
    return out


def in_rulebook_zone(distance: np.ndarray) -> np.ndarray:
    return np.asarray(distance, dtype=float) <= 1.0


def add_zone_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nx, nz = normalized_coords(df)
    df["nx"] = nx
    df["nz"] = nz
    df["zone_dist"] = np.maximum(np.abs(nx), np.abs(nz))
    df["attack_zone"] = attack_zone(df["zone_dist"].to_numpy())
    df["in_zone"] = in_rulebook_zone(df["zone_dist"].to_numpy())
    return df
