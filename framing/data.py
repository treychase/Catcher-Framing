"""Pulling and cleaning the data.

Pitch-level Statcast comes from Baseball Savant's CSV search endpoint, in small
date chunks so no request hits Savant's 25,000-row cap. Statcast's own `umpire`
column has been empty for years, so the home plate umpire is joined on
`game_pk` from the MLB Stats API schedule, and catcher names come from the same
API's people endpoint. Every pull is cached as parquet under `data/`, so a
rerun only fetches the days it has not seen.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
STATSAPI = "https://statsapi.mlb.com/api/v1"
SAVANT_ROW_CAP = 25_000
HEADERS = {"User-Agent": "catcher-framing research (github.com/treychase/Catcher-Framing)"}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _get(
    url: str, params: dict | None = None, tries: int = 5, timeout: int = 120
) -> requests.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}")
        except requests.RequestException as exc:  # network blip: back off and retry
            last = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"giving up on {url}: {last}")


# --------------------------------------------------------------------------
# Statcast
# --------------------------------------------------------------------------
def date_chunks(start: str, end: str, days: int = 2) -> list[tuple[str, str]]:
    """Inclusive [start, end] windows of at most `days` days."""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    while s <= e:
        stop = min(s + timedelta(days=days - 1), e)
        out.append((s.isoformat(), stop.isoformat()))
        s = stop + timedelta(days=1)
    return out


def savant_params(season: int, start: str, end: str) -> dict:
    return {
        "all": "true",
        "hfGT": "R|",
        "hfSea": f"{season}|",
        "player_type": "pitcher",
        "game_date_gt": start,
        "game_date_lt": end,
        "min_pitches": 0,
        "min_results": 0,
        "min_pas": 0,
        "group_by": "name",
        "sort_col": "pitches",
        "sort_order": "desc",
        "type": "details",
    }


def parse_savant_csv(text: str) -> pd.DataFrame:
    """Parse one Savant CSV response down to the columns the pipeline uses."""
    text = text.lstrip("﻿")
    if not text.strip() or text.lstrip().startswith("<"):
        return pd.DataFrame(columns=C.STATCAST_COLUMNS)
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    keep = [c for c in C.STATCAST_COLUMNS if c in df.columns]
    return df[keep]


def fetch_statcast_chunk(season: int, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"statcast_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    r = _get(SAVANT_URL, savant_params(season, start, end))
    df = parse_savant_csv(r.text)
    if len(df) >= SAVANT_ROW_CAP - 1:
        # Truncated: split the window in two and try again.
        s = datetime.strptime(start, "%Y-%m-%d").date()
        e = datetime.strptime(end, "%Y-%m-%d").date()
        if s == e:
            raise RuntimeError(f"Savant row cap hit for a single day {start}")
        mid = s + (e - s) // 2
        df = pd.concat(
            [
                fetch_statcast_chunk(season, start, mid.isoformat(), cache_dir),
                fetch_statcast_chunk(season, (mid + timedelta(days=1)).isoformat(), end, cache_dir),
            ],
            ignore_index=True,
        )
    # Only cache windows that are fully in the past, so today's partial day refetches.
    if datetime.strptime(end, "%Y-%m-%d").date() < date.today():
        df.to_parquet(path, index=False)
    return df


def fetch_statcast(
    seasons: Iterable[int] = C.SEASONS, cache_dir: Path | str = "data/raw", workers: int = 6
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for season in seasons:
        start, end = C.SEASON_WINDOWS[season]
        end = min(end, (date.today() - timedelta(days=1)).isoformat())
        jobs += [(season, s, e) for s, e in date_chunks(start, end)]
    log.info("fetching %d Statcast windows", len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda j: fetch_statcast_chunk(j[0], j[1], j[2], cache_dir), jobs))
    frames = [f for f in frames if len(f)]
    if not frames:
        raise RuntimeError("Savant returned no pitches for the requested seasons")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])


# --------------------------------------------------------------------------
# Umpires and names (MLB Stats API)
# --------------------------------------------------------------------------
def parse_schedule_officials(payload: dict) -> pd.DataFrame:
    rows = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            for off in game.get("officials", []) or []:
                if off.get("officialType") == "Home Plate":
                    o = off.get("official", {})
                    rows.append(
                        {
                            "game_pk": int(game["gamePk"]),
                            "umpire_id": o.get("id"),
                            "umpire": o.get("fullName"),
                        }
                    )
    return pd.DataFrame(rows, columns=["game_pk", "umpire_id", "umpire"])


def parse_boxscore_officials(game_pk: int, payload: dict) -> dict | None:
    for off in payload.get("officials", []) or []:
        if off.get("officialType") == "Home Plate":
            o = off.get("official", {})
            return {"game_pk": int(game_pk), "umpire_id": o.get("id"), "umpire": o.get("fullName")}
    return None


def fetch_umpires(
    game_pks: Iterable[int],
    seasons: Iterable[int] = C.SEASONS,
    cache_path: Path | str = "data/umpires.parquet",
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    have = pd.read_parquet(cache_path) if cache_path.exists() else parse_schedule_officials({})
    need = set(int(g) for g in game_pks) - set(have["game_pk"].astype(int))
    if need:
        frames = [have]
        for season in seasons:
            start, end = C.SEASON_WINDOWS[season]
            for s, e in date_chunks(start, end, days=31):
                r = _get(
                    f"{STATSAPI}/schedule",
                    {
                        "sportId": 1,
                        "gameType": "R",
                        "startDate": s,
                        "endDate": e,
                        "hydrate": "officials",
                    },
                )
                frames.append(parse_schedule_officials(r.json()))
        have = pd.concat(frames, ignore_index=True).drop_duplicates("game_pk", keep="last")
        # Anything the schedule did not carry, ask the boxscore for.
        missing = sorted(need - set(have["game_pk"].astype(int)))
        log.info("schedule gave %d umpires; %d games need the boxscore", len(have), len(missing))

        def one(pk: int) -> dict | None:
            try:
                return parse_boxscore_officials(pk, _get(f"{STATSAPI}/game/{pk}/boxscore").json())
            except RuntimeError:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            extra = [x for x in pool.map(one, missing) if x]
        if extra:
            have = pd.concat([have, pd.DataFrame(extra)], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(cache_path, index=False)
    return have


def fetch_player_names(
    ids: Iterable[int], cache_path: Path | str = "data/players.parquet"
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    have = (
        pd.read_parquet(cache_path)
        if cache_path.exists()
        else pd.DataFrame(columns=["player_id", "name"])
    )
    need = sorted(set(int(i) for i in ids) - set(have["player_id"].astype(int)))
    rows = []
    for i in range(0, len(need), 100):
        batch = need[i : i + 100]
        r = _get(f"{STATSAPI}/people", {"personIds": ",".join(map(str, batch))})
        rows += [
            {"player_id": int(p["id"]), "name": p["fullName"]} for p in r.json().get("people", [])
        ]
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("player_id")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(cache_path, index=False)
    return have


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def called_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep takes the umpire called, with everything the model needs present."""
    desc = df["description"].astype(str)
    called = (desc == C.CALLED_STRIKE) | desc.isin(C.CALLED_BALL)
    out = df.loc[called & ~desc.isin(C.EXCLUDED_DESCRIPTIONS)].copy()
    if "game_type" in out:
        out = out[out["game_type"].fillna("R") == "R"]
    need = ["plate_x", "plate_z", "sz_top", "sz_bot", "pfx_x", "pfx_z", "fielder_2"]
    out = out.dropna(subset=need)
    # Guard against the odd broken zone (sz_top below sz_bot) in the feed.
    out = out[(out["sz_top"] - out["sz_bot"]) > 0.8]
    out["is_strike"] = (out["description"] == C.CALLED_STRIKE).astype(np.int8)
    out["catcher_id"] = out["fielder_2"].astype(np.int64)
    out["game_year"] = pd.to_datetime(out["game_date"]).dt.year.astype(int)
    return out.reset_index(drop=True)


def attach_umpires(df: pd.DataFrame, umps: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in ("umpire", "umpire_id") if c in df.columns])
    out = df.merge(umps[["game_pk", "umpire_id", "umpire"]], on="game_pk", how="left")
    out["umpire"] = out["umpire"].fillna("Unknown")
    return out


def build_dataset(
    seasons: Iterable[int] = C.SEASONS, data_dir: Path | str = "data"
) -> pd.DataFrame:
    """Fetch (or load from cache) and clean every called pitch for `seasons`."""
    data_dir = Path(data_dir)
    raw = fetch_statcast(seasons, data_dir / "raw")
    df = called_pitches(raw)
    umps = fetch_umpires(df["game_pk"].unique(), seasons, data_dir / "umpires.parquet")
    df = attach_umpires(df, umps)
    names = fetch_player_names(df["catcher_id"].unique(), data_dir / "players.parquet")
    df = df.merge(
        names.rename(columns={"player_id": "catcher_id", "name": "catcher"}),
        on="catcher_id",
        how="left",
    )
    df["catcher"] = df["catcher"].fillna(df["catcher_id"].astype(str))
    log.info(
        "%d called pitches, %d games, %d catchers, %d umpires",
        len(df),
        df["game_pk"].nunique(),
        df["catcher_id"].nunique(),
        df["umpire"].nunique(),
    )
    return df
