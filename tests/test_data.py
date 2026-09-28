import pandas as pd
import pytest

from framing import data as D

CSV = """pitch_type,game_date,release_speed,plate_x,plate_z,description,stand,p_throws,balls,strikes,pfx_x,pfx_z,sz_top,sz_bot,game_pk,at_bat_number,pitch_number,pitcher,batter,fielder_2,game_year,game_type,home_team,away_team,extra
FF,2025-05-01,95.1,0.1,2.5,called_strike,R,R,0,0,-0.5,1.3,3.4,1.6,1,1,1,10,20,30,2025,R,NYY,BOS,x
SL,2025-05-01,85.0,1.2,1.1,ball,L,R,1,0,0.3,0.1,3.3,1.5,1,1,2,10,20,30,2025,R,NYY,BOS,x
CH,2025-05-01,84.0,0.0,0.3,blocked_ball,L,R,2,0,0.3,0.1,3.3,1.5,1,1,3,10,20,30,2025,R,NYY,BOS,x
FF,2025-05-01,95.0,0.0,2.3,swinging_strike,L,R,3,0,0.3,0.1,3.3,1.5,1,1,4,10,20,30,2025,R,NYY,BOS,x
FF,2025-05-01,95.0,,2.3,ball,L,R,3,0,0.3,0.1,3.3,1.5,1,2,1,10,20,30,2025,R,NYY,BOS,x
FF,2025-05-01,95.0,0.0,2.3,pitchout,L,R,3,0,0.3,0.1,3.3,1.5,1,2,2,10,20,30,2025,R,NYY,BOS,x
FF,2025-05-01,95.0,0.0,2.3,ball,L,R,3,0,0.3,0.1,1.0,1.5,1,2,3,10,20,30,2025,R,NYY,BOS,x
"""


def test_date_chunks_cover_range():
    ch = D.date_chunks("2025-03-30", "2025-04-04", days=2)
    assert ch == [
        ("2025-03-30", "2025-03-31"),
        ("2025-04-01", "2025-04-02"),
        ("2025-04-03", "2025-04-04"),
    ]
    assert D.date_chunks("2025-04-01", "2025-04-01") == [("2025-04-01", "2025-04-01")]


def test_parse_savant_csv_keeps_needed_columns():
    df = D.parse_savant_csv("﻿" + CSV)
    assert "extra" not in df.columns and len(df) == 7
    assert D.parse_savant_csv("").empty
    assert D.parse_savant_csv("<html>error</html>").empty


def test_called_pitches_filters_and_labels():
    df = D.called_pitches(D.parse_savant_csv(CSV))
    # swinging strike, missing location, pitchout and the broken zone are dropped
    assert df["description"].tolist() == ["called_strike", "ball", "blocked_ball"]
    assert df["is_strike"].tolist() == [1, 0, 0]
    assert df["catcher_id"].tolist() == [30, 30, 30]


def test_parse_schedule_officials():
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 1,
                        "officials": [
                            {
                                "officialType": "First Base",
                                "official": {"id": 9, "fullName": "Wrong Guy"},
                            },
                            {
                                "officialType": "Home Plate",
                                "official": {"id": 5, "fullName": "Plate Ump"},
                            },
                        ],
                    },
                    {"gamePk": 2},
                ]
            }
        ]
    }
    out = D.parse_schedule_officials(payload)
    assert out.to_dict("records") == [{"game_pk": 1, "umpire_id": 5, "umpire": "Plate Ump"}]
    assert D.parse_schedule_officials({}).empty


def test_parse_boxscore_officials():
    box = {"officials": [{"officialType": "Home Plate", "official": {"id": 3, "fullName": "A B"}}]}
    assert D.parse_boxscore_officials(7, box) == {"game_pk": 7, "umpire_id": 3, "umpire": "A B"}
    assert D.parse_boxscore_officials(7, {}) is None


def test_attach_umpires_fills_unknown():
    df = pd.DataFrame({"game_pk": [1, 2]})
    umps = pd.DataFrame({"game_pk": [1], "umpire_id": [5], "umpire": ["Plate Ump"]})
    assert D.attach_umpires(df, umps)["umpire"].tolist() == ["Plate Ump", "Unknown"]


class FakeResponse:
    def __init__(self, text):
        self.text = text


def test_fetch_chunk_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params=None, **kw):
        calls.append(params)
        return FakeResponse(CSV)

    monkeypatch.setattr(D, "_get", fake_get)
    a = D.fetch_statcast_chunk(2025, "2025-05-01", "2025-05-02", tmp_path)
    b = D.fetch_statcast_chunk(2025, "2025-05-01", "2025-05-02", tmp_path)
    assert len(calls) == 1 and len(a) == len(b) == 7
    assert calls[0]["hfSea"] == "2025|" and calls[0]["hfGT"] == "R|"


def test_fetch_chunk_splits_when_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "SAVANT_ROW_CAP", 8)
    seen = []

    def fake_get(url, params=None, **kw):
        seen.append((params["game_date_gt"], params["game_date_lt"]))
        # a two-day window "hits the cap"; single days do not
        body = (
            CSV
            if params["game_date_gt"] != params["game_date_lt"]
            else "\n".join(CSV.splitlines()[:4])
        )
        return FakeResponse(body)

    monkeypatch.setattr(D, "_get", fake_get)
    out = D.fetch_statcast_chunk(2025, "2025-05-01", "2025-05-02", tmp_path)
    assert seen == [
        ("2025-05-01", "2025-05-02"),
        ("2025-05-01", "2025-05-01"),
        ("2025-05-02", "2025-05-02"),
    ]
    assert len(out) == 6


def test_get_retries_then_raises(monkeypatch):
    import requests

    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    n = {"k": 0}

    def boom(*a, **kw):
        n["k"] += 1
        raise requests.ConnectionError("down")

    monkeypatch.setattr(D.requests, "get", boom)
    with pytest.raises(RuntimeError):
        D._get("http://example.invalid", tries=3)
    assert n["k"] == 3
