import sys; sys.path.insert(0, "<PROJECT_ROOT>/revised")
import pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as mpatches, matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings("ignore")
import config as C
region_data = pickle.load(open(C.R + "/data/module1.pkl", "rb"))["region_data"]
YEAR_START, YEAR_END = 1981, 2025; SPI_SCALE = 12
THRESHOLD_MODERATE = -1.0; THRESHOLD_SEVERE = -1.5; THRESHOLD_EXTREME = -2.0; THRESHOLD_WET = 1.0
COLOR_DROUGHT = "#E74C3C"; COLOR_WET = "#2980B9"; COLOR_NEAR_NORM = "#ECF0F1"; COLOR_SPI_LINE = "#2C3E50"; COLOR_ZERO = "#95A5A6"
COLOR_MODERATE = "#E67E22"; COLOR_SEVERE = "#E74C3C"; COLOR_EXTREME = "#8E44AD"
REGIONS = {C.LABELS["Whole"]: {"badge": "#6A4C93"}, C.LABELS["Upper"]: {"badge": "#2E86AB"}, C.LABELS["Middle"]: {"badge": "#E84855"}, C.LABELS["Lower"]: {"badge": "#3BB273"}}
OUTPUT_FILE = C.R + "/plots/fig4_3_spi12_timeseries_regions.png"; DPI = 300
# =============================================================================
# PANEL DRAWING FUNCTION
# =============================================================================

def draw_spi_panel(ax, decimal_time, spi_vals, events, label, badge_color,
                    show_ylabel=True, show_xlabel=True):
    """Draw one SPI-12 panel (compact version for a 2x2 multi-panel figure)."""

    valid = ~np.isnan(spi_vals)

    # Zero reference line
    ax.axhline(0, color=COLOR_ZERO, linewidth=1.0, linestyle="-", zorder=1)

    # Near-normal band
    ax.axhspan(THRESHOLD_MODERATE, THRESHOLD_WET,
               facecolor=COLOR_NEAR_NORM, alpha=0.35, zorder=1)

    # Drought shading
    ax.fill_between(
        decimal_time, spi_vals, THRESHOLD_MODERATE,
        where=(spi_vals < THRESHOLD_MODERATE), interpolate=True,
        facecolor=COLOR_DROUGHT, alpha=0.55, zorder=2
    )

    # Wet shading
    ax.fill_between(
        decimal_time, spi_vals, THRESHOLD_WET,
        where=(spi_vals > THRESHOLD_WET), interpolate=True,
        facecolor=COLOR_WET, alpha=0.55, zorder=2
    )

    # SPI line
    ax.plot(decimal_time, spi_vals,
            color=COLOR_SPI_LINE, linewidth=1.1, zorder=5)

    # Threshold lines (no inline text labels here — explained in shared legend)
    for val, col in [(THRESHOLD_MODERATE, COLOR_MODERATE),
                      (THRESHOLD_SEVERE,   COLOR_SEVERE),
                      (THRESHOLD_EXTREME,  COLOR_EXTREME),
                      (THRESHOLD_WET,      COLOR_WET)]:
        ax.axhline(val, color=col, linewidth=1.1, linestyle="--",
                   alpha=0.9, zorder=3)

    # Annotate major drought events (min SPI < -1.5)
    for ev in events:
        if ev["min_spi"] < THRESHOLD_SEVERE:
            mid_time = (ev["start_yr"] + ev["end_yr"]) / 2
            ax.annotate(
                f"{int(ev['start_yr'])}–{int(ev['end_yr'])}",
                xy=(mid_time, ev["min_spi"]),
                xytext=(mid_time, ev["min_spi"] - 0.6),
                fontsize=7, color="#8E44AD",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=0.7)
            )

    # Axis formatting
    ax.set_xlim(decimal_time[valid][0] - 0.5, decimal_time[-1] + 0.5)
    ax.set_ylim(min(-3.2, np.nanmin(spi_vals) - 0.4),
                max( 3.2, np.nanmax(spi_vals) + 0.4))

    if show_xlabel:
        ax.set_xlabel("Year", fontsize=11, color="#333333", labelpad=5)
    if show_ylabel:
        ax.set_ylabel(f"SPI-{SPI_SCALE}", fontsize=11, color="#333333", labelpad=5)

    ax.set_xticks(np.arange(YEAR_START, YEAR_END + 1, 10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.tick_params(axis="both", labelsize=9, colors="#444444")

    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD", linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAAAAA")

    # Region label badge above panel
    ax.set_title(label, fontsize=12, fontweight="bold",
                 color="black")

# =============================================================================
# BUILD FIGURE — 2 × 2
# =============================================================================

print("\nBuilding figure ...")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("white")

gs = GridSpec(
    2, 2,
    figure=fig,
    wspace=0.12, hspace=0.35,
    left=0.05, right=0.98,
    top=0.92,  bottom=0.14,
)

axes = [
    fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
]
names = list(REGIONS.keys())   # Whole Catchment, Upper, Middle, Lower

for idx, name in enumerate(names):
    row, col = divmod(idx, 2)
    d = region_data[name]
    draw_spi_panel(
        axes[idx],
        d["decimal_time"], d["spi_vals"], d["events"],
        name, REGIONS[name]["badge"],
        show_ylabel=(col == 0),
        show_xlabel=(row == 1),
    )

# =============================================================================
# SHARED LEGEND
# =============================================================================

legend_elements = [
    plt.Line2D([0], [0], color=COLOR_SPI_LINE, linewidth=1.5,
               label=f"SPI-{SPI_SCALE}"),
    mpatches.Patch(facecolor=COLOR_DROUGHT, alpha=0.6,
                   label=f"Drought (SPI < {THRESHOLD_MODERATE})"),
    mpatches.Patch(facecolor=COLOR_WET, alpha=0.6,
                   label=f"Wet (SPI > {THRESHOLD_WET})"),
    mpatches.Patch(facecolor=COLOR_NEAR_NORM, alpha=0.6,
                   label="Near-normal"),
    plt.Line2D([0], [0], color=COLOR_MODERATE, linewidth=1.3,
               linestyle="--", label=f"Moderate drought ({THRESHOLD_MODERATE})"),
    plt.Line2D([0], [0], color=COLOR_SEVERE,   linewidth=1.3,
               linestyle="--", label=f"Severe drought ({THRESHOLD_SEVERE})"),
    plt.Line2D([0], [0], color=COLOR_EXTREME,  linewidth=1.3,
               linestyle="--", label=f"Extreme drought ({THRESHOLD_EXTREME})"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=4,
    fontsize=9.5, framealpha=0.95,
    bbox_to_anchor=(0.51, 0.0),
    edgecolor="#CCCCCC"
)

# =============================================================================
# MAIN TITLE
# =============================================================================
'''
fig.suptitle(
    f"SPI-{SPI_SCALE} Time Series by Region  |  Niger Basin  |  "
    f"CHIRPS v2.0  |  {YEAR_START}–{YEAR_END}",
    fontsize=14, fontweight="bold", color="#1A252F", y=0.98
)'''

# =============================================================================
# SAVE
# =============================================================================

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"\nFigure saved to: {OUTPUT_FILE}")
plt.show()

# =============================================================================
# PRINT SUMMARY STATISTICS PER REGION
# =============================================================================

print("\n── SPI-12 Summary by Region ──────────────────────────────────────")
for name, d in region_data.items():
    spi_vals = d["spi_vals"]
    total_valid = np.sum(~np.isnan(spi_vals))
    drought_months = np.sum(spi_vals < THRESHOLD_MODERATE)
    wet_months     = np.sum(spi_vals >  THRESHOLD_WET)
    print(f"  {name.replace(chr(10), ' ')}")
    print(f"    Total months    : {total_valid}")
    print(f"    Drought months  : {drought_months} "
          f"({100*drought_months/total_valid:.1f}%)")
    print(f"    Wet months      : {wet_months} "
          f"({100*wet_months/total_valid:.1f}%)")
    print(f"    Severe drought  : {np.sum(spi_vals < THRESHOLD_SEVERE)} months")
    print(f"    Extreme drought : {np.sum(spi_vals < THRESHOLD_EXTREME)} months")
    print(f"    Min / Max SPI   : {np.nanmin(spi_vals):.2f} / {np.nanmax(spi_vals):.2f}")
print("───────────────────────────────────────────────────────────────────")
