import sys; sys.path.insert(0, "<PROJECT_ROOT>/revised")
import pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import warnings; warnings.filterwarnings("ignore")
import config as C
records = pickle.load(open(C.R + "/data/taylor_records.pkl", "rb"))
STD_MAX = 1.8; DPI = 300; YEAR_START, YEAR_END = 1981, 2014
OUTPUT_TAYLOR = C.R + "/plots/fig4_6_taylor_diagram.png"
STYLE = {
    "ACCESS-CM2": {"color": "#E74C3C", "marker": "o", "ms": 10}, "ACCESS-ESM1-5": {"color": "#E67E22", "marker": "s", "ms": 10},
    "CanESM5": {"color": "#8E44AD", "marker": "d", "ms": 10}, "EC-Earth3": {"color": "#F1C40F", "marker": "^", "ms": 10},
    "INM-CM4-8": {"color": "#1ABC9C", "marker": "v", "ms": 10}, "INM-CM5-0": {"color": "#3498DB", "marker": "p", "ms": 10},
    "IPSL-CM6A-LR": {"color": "#9B59B6", "marker": "h", "ms": 10}, "KIOST-ESM": {"color": "#E91E63", "marker": "8", "ms": 10},
    "NorESM2-MM": {"color": "#00BCD4", "marker": "<", "ms": 10}, "MPI-ESM1-2-LR": {"color": "#FF5722", "marker": "P", "ms": 10},
    "Ensemble Mean": {"color": "#000000", "marker": "X", "ms": 14},
}
# =============================================================================
# TAYLOR DIAGRAM
# =============================================================================
# The diagram uses a polar-like coordinate system:
#   - Angular axis  : Pearson correlation coefficient
#   - Radial axis   : Normalised standard deviation (model σ / obs σ)
#   - Dashed arcs   : Centred RMSE contours
#   - Reference pt  : (r=1, NSTD=1) — the observation point
# =============================================================================

print("\nPlotting Taylor diagram ...")

fig = plt.figure(figsize=(10, 8.5))
fig.patch.set_facecolor("white")

# Use only the left half of a polar plot (0° to 90°)
ax = fig.add_subplot(111, projection="polar")
ax.set_facecolor("#F9FBFD")
ax.set_theta_direction(-1)           # clockwise
ax.set_theta_offset(np.pi / 2)      # start from top (90°)

# Only show the right half (0 to π/2 in polar = r=1 to r=-1 in correlation)
ax.set_thetamin(0)
ax.set_thetamax(90)

# ── Radial (std dev) axis ─────────────────────────────────────────────────────
ax.set_rlabel_position(90)           # put std labels at top
ax.set_rmax(STD_MAX)
ax.set_rticks(np.arange(0.25, STD_MAX + 0.01, 0.25))
ax.set_yticklabels(
    [f"{v:.2f}" for v in np.arange(0.25, STD_MAX + 0.01, 0.25)],
    fontsize=8.5, color="#555555"
)
ax.set_ylabel("Normalised Standard Deviation",
              labelpad=40, fontsize=10, color="#333333")

# ── Correlation arcs on angular axis ─────────────────────────────────────────
corr_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
theta_ticks = np.arccos(corr_ticks)
ax.set_thetagrids(
    np.degrees(theta_ticks),
    labels=[str(c) for c in corr_ticks],
    fontsize=8.5, color="#555555"
)
ax.set_xlabel("Correlation Coefficient", labelpad=20,
              fontsize=10, color="#333333")

# ── Standard deviation reference circle ───────────────────────────────────────
theta_range = np.linspace(0, np.pi / 2, 300)
ax.plot(theta_range, np.ones_like(theta_range),
        color="#C0392B", linewidth=1.5, linestyle="--",
        zorder=3, label="Reference (CHIRPS σ)")

# ── CRMSE contours (dashed arcs centred on reference point r=1, nstd=1) ──────
crmse_levels = np.arange(0.25, STD_MAX, 0.25)
ref_x = 1.0   # reference point in Cartesian (nstd=1, r=1 → x=1, y=0)
ref_y = 0.0

for crmse_val in crmse_levels:
    # Points on the CRMSE arc in Cartesian
    arc_theta = np.linspace(0, np.pi / 2, 300)
    arc_r     = np.linspace(0, STD_MAX, 300)
    # Convert reference to Cartesian: x = nstd*cos(arccos(r)), y = nstd*sin(arccos(r))
    # CRMSE² = σ_r² + σ_m² - 2·σ_r·σ_m·r  (all normalised, σ_r=1)
    # 1 + σ_m² - 2·σ_m·r = CRMSE²
    r_range = np.linspace(0.001, STD_MAX, 1000)
    points_theta = []
    points_r     = []
    for rm in r_range:
        # correlation: r = (1 + rm² - CRMSE²) / (2·rm)
        r_val = (1 + rm**2 - crmse_val**2) / (2 * rm)
        if -1 <= r_val <= 1:
            th = np.arccos(r_val)
            if 0 <= th <= np.pi / 2:
                points_theta.append(th)
                points_r.append(rm)

    if points_theta:
        ax.plot(points_theta, points_r,
                color="#95A5A6", linewidth=0.7,
                linestyle=":", zorder=2)
        # Label the arc
        if len(points_theta) > 10:
            mid = len(points_theta) // 2
            ax.text(points_theta[mid], points_r[mid],
                    f"{crmse_val:.2f}",
                    fontsize=6.5, color="#7F8C8D",
                    ha="center", va="bottom")

# ── CRMSE arc label ───────────────────────────────────────────────────────────
ax.text(np.deg2rad(70), STD_MAX * 0.78,
        "CRMSE",
        fontsize=8, color="#7F8C8D", rotation=-30)

# ── Reference point (CHIRPS) ──────────────────────────────────────────────────
ax.plot(0, 1.0,
        marker="*", color="#C0392B", markersize=16,
        zorder=6, label="CHIRPS (Reference)",
        markeredgecolor="white", markeredgewidth=0.8)

# ── Plot each model and ensemble ──────────────────────────────────────────────
for rec in records:
    name  = rec["Model"]
    r_val = rec["Pearson r"]
    nstd  = rec["Normalised Std Dev"]

    # Only plot if correlation is valid (positive) and within range
    if r_val <= 0 or nstd > STD_MAX:
        print(f"  Warning: {name} outside plot range "
              f"(r={r_val:.3f}, NSTD={nstd:.3f}) — skipped")
        continue

    theta = np.arccos(r_val)
    sty   = STYLE.get(name, {"color": "grey", "marker": "o", "ms": 9})

    ax.plot(theta, nstd,
            marker=sty["marker"],
            color=sty["color"],
            markersize=sty["ms"],
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=name)

# ── Legend ────────────────────────────────────────────────────────────────────
legend = ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.08, 1.05),
    fontsize=9,
    framealpha=0.95,
    edgecolor="#CCCCCC",
    title="Dataset",
    title_fontsize=10,
)

# ── Title ─────────────────────────────────────────────────────────────────────
'''fig.suptitle(
    f"Taylor Diagram — CMIP6 Models vs CHIRPS v2.0\n"
    f"Niger Basin  |  {YEAR_START}–{YEAR_END} Climatology",
    fontsize=13, fontweight="bold",
    color="#1A252F", y=1.01
)'''

# ── Annotation box ────────────────────────────────────────────────────────────
ax.text(
    np.deg2rad(45), STD_MAX * 1.05,
    "Closer to ★ = better performance",
    fontsize=8.5, color="#555555",
    ha="center", style="italic",
    transform=ax.transData
)

plt.tight_layout()
plt.savefig(OUTPUT_TAYLOR, dpi=DPI, bbox_inches="tight",
            facecolor="white")
print(f"Taylor diagram saved to: {OUTPUT_TAYLOR}")
plt.show()
