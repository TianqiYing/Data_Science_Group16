from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Analysis Datasets" / "master_table_v2_uk_inflation_war_baseline_with_ppi.csv"
OUT = ROOT / "Refined Figures"
OUT.mkdir(exist_ok=True)


WAR = "#d62828" 
STABLE = "#2a9d8f" 

CAT_COLORS = {
    "Food":           "#d62828",
    "Housing/Energy": "#f4a261",
    "Transport":      "#457b9d",
    "Core":           "#2a9d8f",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.labelweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.4,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})


def _suptitle(fig, headline: str, subtitle: str):
    fig.suptitle(headline, fontsize=16.5, fontweight="bold", y=0.985, color="#111")
    fig.text(0.5, 0.935, subtitle, fontsize=11.2, color="#555",
             ha="center", style="italic")


def _clean_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#bbb")
    ax.grid(True, color="#ddd", linewidth=0.4)


def _callout_box(ax, text, x=0.03, y=0.96):
    """Axis-fraction overlay text — never occluded by 3D content."""
    ax.text2D(
        x, y, text, transform=ax.transAxes,
        fontsize=10.5, fontweight="bold", color="#111",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#222", lw=1.0, alpha=0.95),
        zorder=20,
    )


def load_master() -> pd.DataFrame:
    return (
        pd.read_csv(DATA, parse_dates=["date"])
        .sort_values("date")
        .set_index("date")
    )

def fig_contributions(df: pd.DataFrame):
    weights = {"Food": 103, "Housing/Energy": 299, "Transport": 135, "Core": 463}
    cols = {
        "Food": "food_yoy",
        "Housing/Energy": "housing_fuels_yoy",
        "Transport": "transport_yoy",
        "Core": "core_cpih_yoy",
    }
    years = list(range(2019, 2026))
    cats = list(cols.keys())

    annual = df[list(cols.values())].resample("YE").mean()
    annual.index = annual.index.year
    contrib = pd.DataFrame(
        {cat: annual[col] * weights[cat] / 1000 for cat, col in cols.items()}
    ).loc[years]

    fig = plt.figure(figsize=(16.5, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28,
                          left=0.04, right=0.98, top=0.86, bottom=0.10)

    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.view_init(elev=22, azim=-60)
    bw, bd = 0.55, 0.55
    for i, cat in enumerate(cats):
        for j, yr in enumerate(years):
            v = float(contrib.loc[yr, cat])
            war_year = yr in (2022, 2023)
            ax3.bar3d(
                j, i, 0, bw, bd, v,
                color=CAT_COLORS[cat],
                alpha=0.95 if war_year else 0.40,
                edgecolor="white", linewidth=0.5, shade=True,
            )
    ax3.set_xticks(np.arange(len(years)) + bw / 2)
    ax3.set_xticklabels(years, fontsize=9)
    ax3.set_yticks(np.arange(len(cats)) + bd / 2)
    ax3.set_yticklabels(cats, fontsize=9)
    ax3.tick_params(axis="y", pad=-2)  
    ax3.tick_params(axis="x", pad=-2)
    ax3.set_xlabel("Year", labelpad=8, fontweight="bold")
    ax3.set_zlabel("Contribution to CPIH (pp)", labelpad=6, fontweight="bold")
    ax3.set_box_aspect((1.0, 1.0, 0.75))
    _clean_3d(ax3)
    for tl in ax3.get_xticklabels():
        if tl.get_text() in ("2022", "2023"):
            tl.set_color(WAR)
            tl.set_fontweight("bold")
    _callout_box(ax3, "3D view\nbar height = contribution", x=0.02, y=0.97)

    ax2 = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(years))
    for cat in cats:
        ax2.bar(
            years, contrib[cat].values, bottom=bottom,
            color=CAT_COLORS[cat], label=cat,
            edgecolor="white", linewidth=0.6,
        )
        bottom += contrib[cat].values

    ymax = float(bottom.max())
    ax2.axvspan(2021.5, 2023.5, color=WAR, alpha=0.10, zorder=0)
    ax2.text(
        2022.5, ymax * 1.12, "WAR SHOCK",
        ha="center", fontsize=12, fontweight="bold", color=WAR,
    )
    for j, yr in enumerate(years):
        ax2.text(yr, bottom[j] + 0.16, f"{bottom[j]:.1f}",
                 ha="center", fontsize=9, color="#222", fontweight="bold")
    ax2.set_xticks(years)
    ax2.set_ylim(0, ymax * 1.32)  
    ax2.set_xlabel("Year", fontweight="bold")
    ax2.set_ylabel("Total CPIH contribution (pp)", fontweight="bold")
    ax2.legend(loc="upper left", frameon=False, fontsize=9.5,
               ncols=1, handlelength=1.2)
    ax2.set_title("Stacked view  ·  total = sum of contributions",
                  fontsize=11, color="#444", loc="left")
    ax2.set_axisbelow(True)
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax2.spines["left"].set_color("#999")
    ax2.spines["bottom"].set_color("#999")

    _suptitle(
        fig,
        "Food and Housing/Energy drove the 2022-23 CPIH surge",
        "Annual contributions of CPIH categories  ·  weighted by COICOP basket share",
    )
    out = OUT / "fig_3d_bar_contributions_refined.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def fig_lag_surface(df: pd.DataFrame):
    ppi = {
        "PPI Input":  "ppi_input_yoy",
        "PPI Output": "ppi_output_yoy",
        "PPI Import": "ppi_import_yoy",
        "PPI Export": "ppi_export_yoy",
    }
    lags = np.arange(0, 19)
    M = np.full((len(ppi), len(lags)), np.nan)
    for i, (_, col) in enumerate(ppi.items()):
        for j, L in enumerate(lags):
            d = pd.DataFrame({"y": df["cpih_yoy"], "x": df[col].shift(L)}).dropna()
            if len(d) > 10:
                M[i, j] = pearsonr(d.y, d.x)[0]

    pi, pj = np.unravel_index(np.nanargmax(M), M.shape)
    peak_var = list(ppi.keys())[pi]
    peak_lag = int(lags[pj])
    peak_r = float(M[pi, pj])

    fig = plt.figure(figsize=(16.5, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.0], wspace=0.30,
                          left=0.04, right=0.96, top=0.86, bottom=0.10)

    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.view_init(elev=18, azim=-45)
    X, Y = np.meshgrid(lags, np.arange(len(ppi)))
    ax3.plot_surface(
        X, Y, M, cmap="RdYlGn", vmin=-0.2, vmax=0.85,
        edgecolor="white", linewidth=0.25, alpha=0.95,
        rstride=1, cstride=1,
    )
    # Drop-line + marker on the peak (3D objects, not text — fine to occlude)
    ax3.plot([peak_lag, peak_lag], [pi, pi], [0, peak_r],
             color="black", linewidth=1.4, linestyle="--", zorder=12)
    ax3.scatter([peak_lag], [pi], [peak_r], color="black", s=85,
                zorder=14, depthshade=False)

    ax3.set_xticks([0, 6, 12, 18])
    ax3.set_xticklabels(["0m", "6m", "12m", "18m"], fontsize=9.5)
    ax3.set_yticks(range(len(ppi)))
    ax3.set_yticklabels(list(ppi.keys()), fontsize=9.5)
    for tl in ax3.get_yticklabels():
        tl.set_rotation(-18)
        tl.set_horizontalalignment("left")
    ax3.tick_params(axis="x", pad=0)
    ax3.tick_params(axis="y", pad=4)
    ax3.set_xlabel("Lag (months)", labelpad=10, fontweight="bold")
    ax3.set_zlabel("Pearson r", labelpad=6, fontweight="bold")
    ax3.set_zlim(-0.2, 0.95)
    ax3.set_box_aspect((1.4, 1.0, 0.75))
    _clean_3d(ax3)
    _callout_box(
        ax3,
        f"PEAK\n{peak_var}  ·  lag = {peak_lag} months\nr = {peak_r:.2f}",
        x=0.02, y=0.97,
    )

    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(M, cmap="RdYlGn", vmin=-0.2, vmax=0.85, aspect="auto")
    ax2.set_xticks(range(len(lags)))
    ax2.set_xticklabels([f"{l}" for l in lags], fontsize=8)
    ax2.set_yticks(range(len(ppi)))
    ax2.set_yticklabels(list(ppi.keys()))
    ax2.set_xlabel("Lag (months)", fontweight="bold")
    ax2.set_title("2D heatmap  ·  same data, no occlusion",
                  fontsize=11, color="#444", loc="left")
    ax2.add_patch(plt.Rectangle(
        (pj - 0.5, pi - 0.5), 1, 1, fill=False,
        edgecolor="black", linewidth=2.6, zorder=10,
    ))
    ax2.annotate(
        f"peak r = {peak_r:.2f}",
        xy=(pj, pi), xytext=(pj + 3.0, pi - 0.85),
        fontsize=10.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            v = M[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6.6,
                     color="white" if abs(v) > 0.55 else "#222")
    cbar = fig.colorbar(im, ax=ax2, shrink=0.85, pad=0.02)
    cbar.set_label("Pearson r", fontweight="bold")
    ax2.grid(False)

    _suptitle(
        fig,
        f"Producer-price shocks lead consumer prices most strongly at ~{peak_lag} months",
        f"Peak correlation: {peak_var} → CPIH @ {peak_lag}-month lag (r = {peak_r:.2f})",
    )
    out = OUT / "fig_3d_lag_surface_refined.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def fig_regime_clusters(df: pd.DataFrame):
    p = df[["food_yoy", "housing_fuels_yoy", "ppi_input_yoy", "war_dummy"]].dropna()
    norm = p[p.war_dummy == 0]
    war = p[p.war_dummy == 1]

    nc = norm[["food_yoy", "housing_fuels_yoy", "ppi_input_yoy"]].mean()
    wc = war[["food_yoy", "housing_fuels_yoy", "ppi_input_yoy"]].mean()
    delta = float(np.linalg.norm(wc.values - nc.values))

    fig = plt.figure(figsize=(16.5, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.25,
                          left=0.03, right=0.98, top=0.86, bottom=0.10)

    # ── 3D scatter ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.view_init(elev=22, azim=-52)
    ax3.scatter(
        norm.food_yoy, norm.housing_fuels_yoy, norm.ppi_input_yoy,
        c=STABLE, s=24, alpha=0.45,
        label=f"Normal  (n = {len(norm)})", edgecolors="none",
    )
    ax3.scatter(
        war.food_yoy, war.housing_fuels_yoy, war.ppi_input_yoy,
        c=WAR, s=52, alpha=0.88,
        label=f"War shock  (n = {len(war)})",
        edgecolors="white", linewidth=0.4,
    )
    ax3.scatter(*nc, c=STABLE, s=380, marker="*",
                edgecolor="black", linewidth=1.6, zorder=15, depthshade=False,
                label="Normal centroid")
    ax3.scatter(*wc, c=WAR, s=380, marker="*",
                edgecolor="black", linewidth=1.6, zorder=15, depthshade=False,
                label="War centroid")
    ax3.plot(
        [nc.food_yoy, wc.food_yoy],
        [nc.housing_fuels_yoy, wc.housing_fuels_yoy],
        [nc.ppi_input_yoy, wc.ppi_input_yoy],
        color="black", linewidth=1.3, linestyle=":", zorder=12,
    )

    ax3.set_xlabel("Food YoY (%)", labelpad=8, fontweight="bold")
    ax3.set_ylabel("Housing/Energy YoY (%)", labelpad=8, fontweight="bold")
    ax3.set_zlabel("PPI Input YoY (%)", labelpad=6, fontweight="bold")
    ax3.tick_params(axis="x", pad=-2)
    ax3.tick_params(axis="y", pad=-1)
    ax3.set_box_aspect((1.15, 1.0, 0.9))
    _clean_3d(ax3)

    leg = ax3.legend(
        loc="upper left",
        frameon=True, framealpha=1.0,
        facecolor="white", edgecolor="#222",
        fontsize=9.8, handlelength=1.2,
        borderpad=0.55, labelspacing=0.55,
    )
    leg.set_zorder(40)
    ax3.text2D(
        0.98, 0.97,
        f"Δ centroid = {delta:.1f}",
        transform=ax3.transAxes,
        fontsize=10.5, fontweight="bold", color="#111",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#222", lw=1.0, alpha=1.0),
        zorder=40,
    )

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(norm.food_yoy, norm.ppi_input_yoy,
                c=STABLE, s=30, alpha=0.45, edgecolors="none", label="Normal")
    ax2.scatter(war.food_yoy, war.ppi_input_yoy,
                c=WAR, s=55, alpha=0.88, edgecolors="white", linewidth=0.4,
                label="War shock")
    ax2.scatter(nc.food_yoy, nc.ppi_input_yoy,
                c=STABLE, s=420, marker="*",
                edgecolor="black", linewidth=1.6, zorder=10)
    ax2.scatter(wc.food_yoy, wc.ppi_input_yoy,
                c=WAR, s=420, marker="*",
                edgecolor="black", linewidth=1.6, zorder=10)

    ax2.annotate(
        "", xy=(wc.food_yoy, wc.ppi_input_yoy),
        xytext=(nc.food_yoy, nc.ppi_input_yoy),
        arrowprops=dict(arrowstyle="->", color="#111", lw=1.8,
                        connectionstyle="arc3,rad=-0.12"),
    )
    midpoint = ((nc + wc) / 2)
    ax2.text(
        midpoint.food_yoy, midpoint.ppi_input_yoy + 1.8,
        f"Δ centroid = {delta:.1f}",
        fontsize=11, fontweight="bold", color="#111", ha="center",
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#666", lw=0.7),
    )

    ax2.set_xlabel("Food YoY (%)", fontweight="bold")
    ax2.set_ylabel("PPI Input YoY (%)", fontweight="bold")
    ax2.set_title("2D projection  ·  Food × PPI Input  (clearest separation)",
                  fontsize=11, color="#444", loc="left")
    ax2.legend(loc="lower right", frameon=False, fontsize=10)
    ax2.set_axisbelow(True)
    ax2.grid(axis="both", linewidth=0.4, alpha=0.5)
    ax2.spines["left"].set_color("#999")
    ax2.spines["bottom"].set_color("#999")

    _suptitle(
        fig,
        "War-shock months separate cleanly from normal months",
        "Three-dimensional inflation regime: Food, Housing/Energy and PPI Input YoY",
    )
    out = OUT / "fig_3d_scatter_clustering_refined.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def main():
    df = load_master()
    print(f"Loaded {len(df)} rows from {DATA.name}")
    print(f"Writing refined figures to: {OUT}/")
    fig_contributions(df)
    fig_lag_surface(df)
    fig_regime_clusters(df)
    print("Done.")


if __name__ == "__main__":
    main()
