"""REVISED copy of plot_precip_change_by_model.py: reads the Kainji-upstream-mask tables in revised/plots (module2_precip_pct_change_*.csv)."""
"""
=============================================================================
Projected Precipitation Change by CMIP6 Model — Dry-Season and Annual-Mean
=============================================================================
Companion figure to Module 3's "Projected Discharge by Model" plot
(fig_projected_discharge.png), but showing the ΔP driver itself (%) rather
than the discharge it produces. Three panels: (a) dry-season (Nov-Mar) ΔP,
the primary elasticity driver used in Figures 1-4; (b) wet-season (Apr-Oct)
ΔP, the comparison driver used in Figure 6; (c) annual-mean ΔP, the
comparison driver used in Figure 5.

Data source: analysis/module2_precip_change.py output CSVs
  module2_precip_pct_change_dry_season.csv
  module2_precip_pct_change_wet_season.csv
  module2_precip_pct_change_annual.csv

REQUIREMENTS:
    pip install numpy pandas matplotlib
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "./output/"
DPI = 300

# =============================================================================
# LOAD DATA
# =============================================================================

def load_delta_p(csv_path):
    """Load per-model ΔP (%), keeping the 'Ensemble Mean' row separate."""
    df = pd.read_csv(csv_path, index_col="model")
    ensemble = df.loc["Ensemble Mean"]
    models = df.drop(index="Ensemble Mean", errors="ignore")
    return models, ensemble

dry_models, dry_ensemble       = load_delta_p(f"{OUTPUT_DIR}module2_precip_pct_change_dry_season.csv")
wet_models, wet_ensemble       = load_delta_p(f"{OUTPUT_DIR}module2_precip_pct_change_wet_season.csv")
annual_models, annual_ensemble = load_delta_p(f"{OUTPUT_DIR}module2_precip_pct_change_annual.csv")

model_names = list(dry_models.index)
x_labels    = model_names + ["Ensemble\nMean"]
x           = np.arange(len(x_labels))
bar_w       = 0.18
offsets     = [-1.5, -0.5, 0.5, 1.5]   # near245, far245, near585, far585

combo_labels = [
    ("ssp245_near", "SSP2-4.5 Mid-century",   "#AED6F1"),
    ("ssp245_far",  "SSP2-4.5 Late-century",    "#2980B9"),
    ("ssp585_near", "SSP5-8.5 Mid-century",   "#F1948A"),
    ("ssp585_far",  "SSP5-8.5 Late-century",    "#C0392B"),
]

# =============================================================================
# FIGURE — DRY-SEASON vs WET-SEASON vs ANNUAL-MEAN ΔP BY MODEL AND ENSEMBLE
# =============================================================================

print("Building figure — Projected Precipitation Change by Model ...")

fig, axes = plt.subplots(1, 3, figsize=(26, 7), sharey=True)
fig.patch.set_facecolor("white")

panels = [
    (axes[0], dry_models,    dry_ensemble,    "Dry-Season (Nov-Mar) ΔP"),
    (axes[1], wet_models,    wet_ensemble,    "Wet-Season (Apr-Oct) ΔP"),
    (axes[2], annual_models, annual_ensemble, "Annual-Mean ΔP"),
]

for ax_idx, (ax, models_df, ensemble_row, panel_label) in enumerate(panels):
    for ci, (key, lbl, col) in enumerate(combo_labels):
        model_vals = models_df[key].values
        ens_val    = ensemble_row[key]

        # Individual-model bars
        ax.bar(x[:-1] + offsets[ci] * bar_w, model_vals,
               width=bar_w, color=col, alpha=0.85,
               label=lbl if ax_idx == 0 else "_nolegend_",
               edgecolor="white", linewidth=0.4, zorder=3)
        # Ensemble-mean bar (bold outline, same colour, set apart by the divider below)
        ax.bar(x[-1] + offsets[ci] * bar_w, ens_val,
               width=bar_w, color=col, alpha=0.85,
               label="_nolegend_",
               edgecolor="black", linewidth=1.3, zorder=3)

    # Divider between individual models and the ensemble mean
    ax.axvline(len(model_names) - 0.5, color="#888888", linewidth=1.0, linestyle="-", zorder=2)
    # Zero-change reference line
    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Precipitation Change, ΔP (%)", fontsize=11)
    ax.set_title(panel_label, fontsize=12, fontweight="bold", color="#1A252F")
    ax.tick_params(labelsize=10)
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.01, 0.98, f"({'abc'[ax_idx]})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")

fig.legend(loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.06), framealpha=0.95)
fig.suptitle(
    "Projected Precipitation Change over the Kainji Contributing Catchment (49 Draining HydroBASINS Polygons) by CMIP6 Model and Scenario\n"
    "Baseline period: 1984-2014  |  Black-outlined bars = 10-model ensemble mean",
    fontsize=13, fontweight="bold", color="#1A252F", y=1.03
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}figA2_precip_change_by_model.png",
            dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"  Saved: figA2_precip_change_by_model.png")

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("PRECIPITATION CHANGE BY MODEL — SUMMARY")
print("=" * 70)
for label, ensemble_row in [("Dry-season", dry_ensemble), ("Wet-season", wet_ensemble), ("Annual-mean", annual_ensemble)]:
    print(f"\n{label} ensemble-mean ΔP (%):")
    for key, lbl, _ in combo_labels:
        print(f"  {lbl:16s}: {ensemble_row[key]:+.2f}%")
