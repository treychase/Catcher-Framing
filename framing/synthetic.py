"""A synthetic called-pitch sample in Statcast's schema.

Used by the unit tests and for dry runs of the whole pipeline without network
access. The generating process is deliberately simple but has the structure a
framing model has to recover: a soft zone edge, umpires whose zones differ in
size, a count effect, a small movement effect, and catchers with a known true
framing skill that only acts near the edge. Nothing built from it is ever
published as a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def make_called_pitches(
    n: int = 40_000,
    n_catchers: int = 24,
    n_umpires: int = 30,
    n_games: int | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_games = n_games or max(n // 150, 10)

    catcher_ids = np.arange(600_000, 600_000 + n_catchers)
    catcher_skill = rng.normal(0, 0.35, n_catchers)  # logit shift near the edge
    ump_names = np.array([f"Umpire {chr(65 + i % 26)}{i // 26}" for i in range(n_umpires)])
    ump_size = rng.normal(0, 0.05, n_umpires)  # zone edge shift, zone units

    game_pk = rng.integers(700_000, 700_000 + n_games, n)
    game_ump = rng.integers(0, n_umpires, n_games)
    game_catcher = rng.integers(0, n_catchers, n_games)
    g = game_pk - 700_000
    u = game_ump[g]
    c = game_catcher[g]

    sz_bot = rng.normal(1.6, 0.08, n)
    sz_top = sz_bot + rng.normal(1.8, 0.1, n)
    # Takes cluster around the edge, as real takes do.
    plate_x = rng.normal(0, 0.75, n)
    plate_z = (sz_top + sz_bot) / 2 + rng.normal(0, 0.8, n)
    pfx_x = rng.normal(0, 0.7, n)
    pfx_z = rng.normal(0.8, 0.6, n)
    balls = rng.integers(0, 4, n)
    strikes = rng.integers(0, 3, n)
    stand = rng.choice(["R", "L"], n, p=[0.58, 0.42])
    p_throws = rng.choice(["R", "L"], n, p=[0.72, 0.28])

    df = pd.DataFrame({"plate_x": plate_x, "plate_z": plate_z, "sz_top": sz_top, "sz_bot": sz_bot})
    half_h = (sz_top - sz_bot) / 2 + C.BALL_RADIUS
    d = np.maximum(
        np.abs(plate_x / C.ZONE_HALF_WIDTH), np.abs((plate_z - (sz_top + sz_bot) / 2) / half_h)
    )
    edge = 1.0 + ump_size[u] + 0.03 * (balls - strikes)  # hitter's counts widen the zone
    near_edge = np.exp(-(((d - 1.0) / 0.25) ** 2))
    logit = -14.0 * (d - edge) + catcher_skill[c] * near_edge * 2.0 - 0.15 * pfx_z
    p = 1 / (1 + np.exp(-logit))
    is_strike = rng.random(n) < p

    dates = pd.to_datetime("2025-04-01") + pd.to_timedelta(g % 360, unit="D")
    df = df.assign(
        game_pk=game_pk,
        game_date=dates.strftime("%Y-%m-%d"),
        game_year=np.where(g % 2 == 0, 2025, 2026),
        game_type="R",
        at_bat_number=rng.integers(1, 80, n),
        pitch_number=np.arange(n),
        pitcher=rng.integers(500_000, 500_300, n),
        batter=rng.integers(400_000, 400_400, n),
        fielder_2=catcher_ids[c],
        catcher_id=catcher_ids[c],
        catcher=[f"Catcher {i:02d}" for i in c],
        description=np.where(is_strike, "called_strike", "ball"),
        pitch_type="FF",
        stand=stand,
        p_throws=p_throws,
        balls=balls,
        strikes=strikes,
        pfx_x=pfx_x,
        pfx_z=pfx_z,
        release_speed=rng.normal(93, 3, n),
        umpire=ump_names[u],
        umpire_id=u,
        is_strike=is_strike.astype(np.int8),
    )
    df.attrs["true_catcher_skill"] = dict(
        zip(catcher_ids.tolist(), catcher_skill.tolist(), strict=True)
    )
    return df
