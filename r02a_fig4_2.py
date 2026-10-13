import sys; sys.path.insert(0, "<PROJECT_ROOT>/revised")
import pickle, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches, matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings("ignore")
import config as C
_d = pickle.load(open(C.R + "/data/module1.pkl", "rb")); ts_data = _d["ts_data"]; mk_results = _d["mk_results"]
YEAR_START, YEAR_END = 1981, 2025; ALPHA = 0.05; MA_YEARS = 5
TREND_COLOR = "#1A252F"; MA_COLOR = "#FF6B35"; MA_LW = 2.2; TREND_LW = 1.8; TREND_LS = "--"
REGIONS = {
    C.LABELS["Whole"]:  {"color": "#6A4C93", "bar_alpha": 0.45},
    C.LABELS["Upper"]:  {"color": "#2E86AB", "bar_alpha": 0.45},
    C.LABELS["Middle"]: {"color": "#E84855", "bar_alpha": 0.45},
    C.LABELS["Lower"]:  {"color": "#3BB273", "bar_alpha": 0.45},
}
OUTPUT_FILE = C.R + "/plots/fig4_2_region_subregion_timeseries.png"; DPI = 300
# =============================================================================
# FIGURE — 2 × 2
# =============================================================================

print("\nBuilding figure ...")

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("white")

gs = GridSpec(
    2, 2,
    figure=fig,
    wspace=0.15, hspace=0.45,
    left=0.06, right=0.98,
    top=0.93,  bottom=0.10,
)

axes = [
    fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
]
# Whole-catchment panel first, then the three sub-regions
names     = list(REGIONS.keys())
panel_lbl = ["(a)", "(b)", "(c)", "(d)"]

for idx, name in enumerate(names):
    ax      = axes[idx]
    cfg     = REGIONS[name]
    years   = ts_data[name]["years"]
    vals    = ts_data[name]["values"]
    mkr     = mk_results[name]
    color   = cfg["color"]

    # ── Bar chart — annual total values ───────────────────────────────────────
    ax.bar(years, vals,
           color=color, alpha=cfg["bar_alpha"],
           width=0.85, label="Annual total",
           zorder=2)

    # ── 5-year moving average ─────────────────────────────────────────────────
    ma_vals = np.convolve(
        vals,
        np.ones(MA_YEARS) / MA_YEARS,
        mode="valid"
    )
    ma_years = years[MA_YEARS - 1:]
    ax.plot(ma_years, ma_vals,
            color=MA_COLOR, linewidth=MA_LW,
            label=f"{MA_YEARS}-year moving average",
            zorder=4)

    # ── Sen's slope trend line ────────────────────────────────────────────────
    # pymannkendall's slope/intercept are expressed against an evenly spaced
    # index (0, 1, 2, ...), not the raw calendar year values, so the fitted
    # line must be built on that index and then plotted against the actual
    # years for correct alignment on the x-axis.
    x = np.arange(len(years))
    trend_y = mkr["intercept"] + mkr["slope"] * x
    ax.plot(years, trend_y,
            color=TREND_COLOR, linewidth=TREND_LW,
            linestyle=TREND_LS,
            label="Sen's slope trend",
            zorder=5)

    # ── Annotation — Sen's slope and significance ─────────────────────────────
    slope_decade = mkr["slope"] * 10   # convert per year → per decade
    sign_sym     = "+" if slope_decade >= 0 else ""
    sig_label    = f"p < {ALPHA}" if mkr["significant"] else f"p = {mkr['p']:.3f}"
    trend_label  = mkr["trend"].capitalize()

    annotation = (
        f"Trend: {trend_label}\n"
        f"Sen's slope: {sign_sym}{slope_decade:.3f} mm/decade\n"
        f"{sig_label}"
    )

    # Annotation box in top-left or top-right depending on trend direction
    ha_pos = 0.03 if slope_decade >= 0 else 0.97
    ha_txt = "left" if slope_decade >= 0 else "right"

    ax.text(ha_pos, 0.97, annotation,
            transform=ax.transAxes,
            fontsize=10, va="top", ha=ha_txt,
            color=TREND_COLOR,
            bbox=dict(facecolor="white", alpha=0.60,
                      edgecolor="#CCCCCC", pad=4,
                      boxstyle="round,pad=0.4"))

    # ── Reference line at long-term mean ──────────────────────────────────────
    long_mean = np.nanmean(vals)
    ax.axhline(long_mean,
               color="red", linewidth=1.0,
               linestyle=":", alpha=0.7, zorder=3,
               label=f"Long-term mean ({long_mean:.0f} mm/yr)")

    # ── Panel letter ──────────────────────────────────────────────────────────
    ax.text(0.03, 1, panel_lbl[idx],
            transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            va="bottom", ha="left", color="black")

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.set_title(name, fontsize=13, fontweight="bold",
                 color="#1A252F", pad=7)
    ax.set_xlabel("Year", fontsize=12, color="black")
    ax.set_ylabel("Precipitation (mm / yr)", fontsize=12, color="black")
    ax.tick_params(axis="both", labelsize=10, colors="black")
    ax.set_xlim(years[0] - 0.8, years[-1] + 0.8)

    # Y-axis lower bound at zero
    ax.set_ylim(bottom=0)

    # X-axis ticks every 5 years
    ax.set_xticks(np.arange(
        (years[0] // 5) * 5,
        years[-1] + 5, 5
    ))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.tick_params(axis="x", rotation=45)

    # Grid
    ax.yaxis.grid(True, linewidth=0.4, color="#CCCCCC",
                  linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAAAAA")

# =============================================================================
# SHARED LEGEND (below all panels)
# =============================================================================

legend_elements = [
    #mpatches.Patch(facecolor="#888888", alpha=0.5, label="Annual total (CHIRPS)"),
    plt.Line2D([0], [0], color=MA_COLOR,    linewidth=2.2,
               label=f"{MA_YEARS}-year moving average"),
    plt.Line2D([0], [0], color=TREND_COLOR, linewidth=1.8,
               linestyle=TREND_LS,
               label="Sen's slope trend line"),
    plt.Line2D([0], [0], color="red",   linewidth=1.0,
               linestyle=":", label="Long-term mean"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=4,
    fontsize=10, framealpha=0.95,
    bbox_to_anchor=(0.51, -0.01),
    edgecolor="#CCCCCC"
)

# =============================================================================
# MAIN TITLE
# =============================================================================
'''
fig.suptitle(
    f"Annual Total Precipitation Time Series by Sub-Region  |  "
    f"Niger Basin  |  CHIRPS v2.0  |  {YEAR_START}–{YEAR_END}",
    fontsize=13, fontweight="bold",
    color="#1A252F", y=1.01
)'''

# =============================================================================
# SAVE
# =============================================================================

plt.savefig(OUTPUT_FILE, dpi=DPI,
            bbox_inches="tight",
            facecolor="white")
print(f"\nFigure saved to: {OUTPUT_FILE}")
plt.show()
