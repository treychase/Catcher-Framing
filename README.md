# Catcher Framing

[![CI](https://github.com/treychase/Catcher-Framing/actions/workflows/ci.yml/badge.svg)](https://github.com/treychase/Catcher-Framing/actions/workflows/ci.yml)
[![Pipeline](https://github.com/treychase/Catcher-Framing/actions/workflows/pipeline.yml/badge.svg)](https://github.com/treychase/Catcher-Framing/actions/workflows/pipeline.yml)

A predicted called-strike model and a shadow-zone catcher framing grade, built
on every 2025 and 2026 regular-season take in Statcast.

**Live page:** <https://treychase.github.io/projects/catcher-framing.html>

<!-- RESULTS:START -->
<!-- RESULTS:END -->

## What it does

1. **Data.** Every regular-season pitch from Baseball Savant's CSV search, in
   two-day windows so no request hits Savant's 25,000-row cap, cut down to takes
   the plate umpire had to call (`called_strike`, `ball`, `blocked_ball`;
   pitchouts and intentional balls removed). Statcast's own `umpire` column is
   empty, so the home plate umpire comes from the MLB Stats API schedule
   (`hydrate=officials`, with the boxscore as a fallback) and is joined on
   `game_pk`. Catcher names come from the Stats API people endpoint.
2. **Called-strike model.** LightGBM on plate location (`plate_x`, `plate_z`),
   the batter's zone (`sz_top`, `sz_bot`), movement (`pfx_x`, `pfx_z`), the
   count, handedness, and the **umpire as a native categorical feature**, so
   each umpire gets his own zone shape. The catcher is deliberately not a
   feature: what the model cannot explain about a call is framing.
   Hyperparameters come from `GridSearchCV` over leaves, learning rate, trees
   and minimum child size, scored on log loss with `GroupKFold` by game.
   Diagnostics (AUC, log loss, Brier, calibration, ROC, feature importance, a
   no-umpire ablation) are on a held-out fifth of games. The probabilities used
   for framing are **out-of-fold**, so no pitch is scored by a model that saw
   its game.
3. **Shadow-zone grade.** Locations are converted to zone units: 0 is the
   middle, ±1 is the plate edge widened by a ball's radius, vertically scaled
   to each batter's own zone. The distance from the middle is the larger of the
   two axes, and the **shadow** is 0.67 to 1.33, Savant's attack-zone
   definition. Only shadow takes are graded:

   | Call | Where | Points |
   | --- | --- | --- |
   | Ball called a strike (stolen) | outside the edge, 1.00 to 1.33 | **+1 at the edge rising linearly to +2** at the shadow's outer border |
   | Strike called a ball (lost) | inside the edge, 0.67 to 1.00 | **−1 at the edge falling linearly to −2** at the shadow's inner border |
   | Called correctly | anywhere | 0 |

   Frame Score per 100 is the net per 100 shadow takes. The grade is
   `50 + 10 × z` among qualified catchers, clipped to the 20–80 scouting scale
   (800 shadow takes over both seasons, 450 in one).
4. **Validation against the model.** Pitch level: stolen strikes should carry
   a low model probability, falling further as the weight rises. Catcher level:
   Frame Score per 100 against the model's strikes added per 100 shadow takes
   (Pearson, Spearman, top and bottom ten overlap). Stability: each measure
   from 2025 to 2026. Framing runs are model strikes added × 0.125.
5. **Outputs.** CSVs, `metrics.json` and PNG diagnostics in `results/`, and a
   self-contained page (leaderboards, per-catcher zone heatmaps, validation,
   model diagnostics) in `site/catcher-framing.html`.

## Layout

```
framing/
  config.py      every modelling decision: features, grid, zone geometry, weights
  data.py        Savant + Stats API fetchers, caching, cleaning
  zone.py        normalized zone coordinates and attack zones
  model.py       feature spec, grouped grid search, fit, out-of-fold, diagnostics
  grading.py     shadow-zone pitch scores, catcher table, 20-80 grades, heatmaps
  validate.py    grade vs model, pitch and catcher level, year to year
  plots.py       static PNG diagnostics
  report.py      writes the payload into templates/catcher_framing_template.html
  synthetic.py   Statcast-shaped synthetic data for tests and dry runs
  cli.py         python -m framing {fetch,run}
tests/           pytest suite (runs offline on synthetic data)
results/         the latest build's numbers and figures
site/            the latest build's page
```

## Running it

```bash
pip install -e ".[dev]"
python -m framing fetch            # pull and cache Statcast + umpires under data/
python -m framing run              # model, grade, validate, write results/ and site/
python -m framing run --synthetic --quick   # offline dry run on generated data
pytest                             # unit tests
```

## CI/CD

* **CI** (`.github/workflows/ci.yml`) runs on every push and pull request:
  `ruff` lint and format checks, the pytest suite with coverage on Python 3.10
  and 3.12, and a synthetic end-to-end build whose page is uploaded as an
  artifact.
* **Pipeline** (`.github/workflows/pipeline.yml`) runs on a weekly schedule,
  on demand, and whenever the model code changes. It gates on the tests,
  restores the cached Statcast pulls, fetches only new days, rebuilds the
  model, grades and page, uploads them as an artifact, and commits `results/`
  and `site/` back to the branch. On `main` it also publishes the page to
  `treychase.github.io/projects/catcher-framing.html` when a
  `SITE_DEPLOY_TOKEN` secret (a fine-grained token on the site repository with
  *Contents: read and write*) is present; without it that step is skipped.
