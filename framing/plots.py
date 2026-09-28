"""Static diagnostic figures (PNG) for the repository and the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import config as C  # noqa: E402

INK, MUTED, FIELD, GOLD = "#14201B", "#4B5A52", "#1F5C43", "#DFA22C"
COOL, WARM = "#2166AC", "#B2182B"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#E3E7DB",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titleweight": "bold",
    }
)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def calibration(cal: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.plot(cal["predicted"], cal["observed"], color=FIELD, lw=2, marker="o", ms=5, label="Model")
    ax.set(
        xlabel="Predicted strike probability",
        ylabel="Observed called-strike rate",
        title="Calibration, held-out games",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def roc(pts: pd.DataFrame, auc: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.plot(pts["fpr"], pts["tpr"], color=FIELD, lw=2)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"ROC, held-out games (AUC {auc:.3f})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    _save(fig, path)


def importance(imp: pd.DataFrame, path: Path):
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.barh(imp["feature"], imp["share"] * 100, color=FIELD, height=0.6)
    ax.set(xlabel="Share of total split gain (%)", title="Feature importance")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def grid(table: pd.DataFrame, path: Path):
    t = table.sort_values("cv_log_loss").reset_index(drop=True)
    labels = [
        f"leaves {int(r.num_leaves)}, lr {r.learning_rate:g}, trees {int(r.n_estimators)}, "
        f"min child {int(r.min_child_samples)}"
        for r in t.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(t) + 1.2))
    ax.errorbar(
        t["cv_log_loss"],
        range(len(t)),
        xerr=t["cv_log_loss_sd"],
        fmt="o",
        color=FIELD,
        ecolor=MUTED,
        ms=5,
        capsize=2,
    )
    ax.plot(t["cv_log_loss"].iloc[0], 0, "o", color=GOLD, ms=8, zorder=5)
    ax.set_yticks(range(len(t)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="Cross-validated log loss (lower is better)", title="Grid search")
    _save(fig, path)


def umpires(ump: pd.DataFrame, path: Path, k: int = 12):
    show = pd.concat([ump.head(k), ump.tail(k)])
    fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(show) + 1.2))
    colors = [WARM if v > 0 else COOL for v in show["vs_average"]]
    ax.barh(show["umpire"], show["vs_average"] * 100, color=colors, height=0.65)
    ax.axvline(0, color=MUTED, lw=1)
    ax.invert_yaxis()
    ax.set(
        xlabel="Expected strike rate vs average, same pitches (pct. points)",
        title="Umpire zones: most and least generous",
    )
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def probability_surface(surface: dict, path: Path):
    g = np.array(surface["axis"])
    p = np.array(surface["p"])
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    im = ax.imshow(
        p, origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap="Greens", vmin=0, vmax=1
    )
    cs = ax.contour(g, g, p, levels=[0.5], colors=[GOLD], linewidths=2)
    ax.clabel(cs, fmt={0.5: "50%"}, fontsize=8)
    _zone_box(ax)
    ax.set(
        xlabel="Horizontal (zone units, catcher's view)",
        ylabel="Vertical (zone units)",
        title="Predicted strike probability",
    )
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def distance_profile(prof: pd.DataFrame, path: Path):
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax.plot(
        prof["distance"],
        prof["strike_rate"],
        color=INK,
        lw=2,
        marker="o",
        ms=4,
        label="Observed called-strike rate",
    )
    ax.plot(
        prof["distance"], prof["p_strike"], color=FIELD, lw=2, ls="--", label="Model probability"
    )
    ax.axvline(1, color=MUTED, lw=1)
    ax.set(ylabel="Strike probability", ylim=(0, 1), title="Across the shadow")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    d = prof["distance"]
    ax2.plot(d[d <= 1], prof["weight"][d <= 1], color=COOL, lw=2, label="Cost of a lost strike")
    ax2.plot(d[d > 1], prof["weight"][d > 1], color=WARM, lw=2, label="Credit for a stolen strike")
    ax2.axvline(1, color=MUTED, lw=1)
    ax2.set(
        xlabel="Distance from zone centre (1 = edge)",
        ylabel="Grade weight",
        ylim=(0, C.MAX_WEIGHT * 1.15),
    )
    ax2.legend(frameon=False, loc="lower center", fontsize=8, ncol=2)
    _save(fig, path)


def validation_scatter(table: pd.DataFrame, r: float, path: Path):
    q = table[table["qualified"]]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(q["strikes_added_100"], q["frame_score_100"], s=22, color=FIELD, alpha=0.8)
    b = np.polyfit(q["strikes_added_100"], q["frame_score_100"], 1)
    xs = np.linspace(q["strikes_added_100"].min(), q["strikes_added_100"].max(), 10)
    ax.plot(xs, np.polyval(b, xs), color=GOLD, lw=2)
    for row in pd.concat([q.head(3), q.tail(3)]).itertuples():
        ax.annotate(
            row.catcher,
            (row.strikes_added_100, row.frame_score_100),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
            color=INK,
        )
    ax.set(
        xlabel="Model strikes added per 100 shadow takes",
        ylabel="Frame Score per 100 shadow takes",
        title=f"Grade vs model (Pearson r = {r:.2f})",
    )
    _save(fig, path)


def _zone_box(ax):
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, ec=INK, lw=1.5))
    for s in (C.HEART_EDGE, C.SHADOW_OUTER):
        ax.add_patch(Rectangle((-s, -s), 2 * s, 2 * s, fill=False, ec=MUTED, lw=0.8, ls=":"))


def catcher_heatmaps(heats: list[tuple[str, list]], title: str, path: Path, vmax: float):
    n = len(heats)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.1))
    axes = np.atleast_1d(axes)
    e = C.HEAT_EXTENT
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    for ax, (name, h) in zip(axes, heats, strict=True):
        m = np.array([[np.nan if v is None else v for v in row] for row in h], dtype=float)
        im = ax.imshow(m, extent=[-e, e, -e, e], cmap="RdBu_r", norm=norm)
        _zone_box(ax)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(title, fontweight="bold")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, label="Frame score per shadow take vs league")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
