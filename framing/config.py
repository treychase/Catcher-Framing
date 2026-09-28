"""Constants shared across the pipeline.

Everything that encodes a modelling decision lives here, so a change of mind
about the zone, the grid or the grading weights is a one-line edit.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
SEASONS = (2025, 2026)

# Regular season windows, padded; empty days are skipped by the fetcher.
SEASON_WINDOWS = {
    2025: ("2025-03-18", "2025-09-30"),
    2026: ("2026-03-25", "2026-09-30"),
}

# Statcast `description` values that are a take the umpire had to call.
CALLED_STRIKE = "called_strike"
CALLED_BALL = ("ball", "blocked_ball")
# Pitchouts and intentional balls are not framing opportunities.
EXCLUDED_DESCRIPTIONS = ("pitchout", "intent_ball", "automatic_ball", "automatic_strike")

# Columns kept from the Statcast CSV (it ships ~115).
STATCAST_COLUMNS = [
    "game_pk",
    "game_date",
    "game_year",
    "game_type",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "fielder_2",
    "home_team",
    "away_team",
    "description",
    "pitch_type",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "pfx_x",
    "pfx_z",
    "release_speed",
]

# --------------------------------------------------------------------------
# Strike zone geometry (feet). Statcast plate_x is from the catcher's view.
# --------------------------------------------------------------------------
PLATE_HALF_WIDTH = 17.0 / 2.0 / 12.0  # 0.708 ft
BALL_RADIUS = 1.45 / 12.0  # a pitch is a strike if any part touches
ZONE_HALF_WIDTH = PLATE_HALF_WIDTH + BALL_RADIUS

# Savant's attack zones, as fractions of the (ball-adjusted) zone half size.
HEART_EDGE = 0.67
SHADOW_OUTER = 1.33
CHASE_OUTER = 2.00

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "plate_x",
    "plate_z",  # location
    "sz_top",
    "sz_bot",  # batter-specific zone the location is judged against
    "pfx_x",
    "pfx_z",  # movement: horizontal / induced vertical break (ft)
    "balls",
    "strikes",  # count
]
CATEGORICAL_FEATURES = ["umpire", "stand", "p_throws"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PARAM_GRID = {
    "num_leaves": [15, 63],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [250, 500],
    "min_child_samples": [50, 200],
}
GRID_SAMPLE = 200_000  # rows used for the grid search (the refit uses everything)
CV_FOLDS = 3  # grid search folds, grouped by game
OOF_FOLDS = 5  # out-of-fold folds for the framing residuals
TEST_FRACTION = 0.2  # held-out games for the reported diagnostics
RANDOM_STATE = 2026

# --------------------------------------------------------------------------
# Framing grades
# --------------------------------------------------------------------------
# Credit for a ball called a strike grows linearly from 1 at the zone edge to
# MAX_WEIGHT at the outer edge of the shadow. A strike called a ball costs the
# mirror image: 1 at the edge, MAX_WEIGHT at the inner edge of the shadow.
MAX_WEIGHT = 2.0
RUNS_PER_STRIKE = 0.125  # run value of turning a ball into a strike, league average
MIN_SHADOW_PITCHES = 800  # to qualify for a leaderboard over both seasons
MIN_SHADOW_PITCHES_SEASON = 450

# Heatmap grid over the normalized zone: 18 bins of 1/6 between -1.5 and 1.5
# put bin edges exactly on +-0.667, +-1.0 and +-1.333 - the shadow's borders.
HEAT_EXTENT = 1.5
HEAT_BINS = 18
HEAT_MIN_PITCHES = 6
