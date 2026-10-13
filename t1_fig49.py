"""Figure 4.9 (Tier 1): identical to the V2.0 figure except that the HMF-WA series is labelled 'simulated flow', not 'natural flow'
(the population-driven water-use variant of HMF-WA includes abstractions, so the flows are not natural flows)."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "<PROJECT_ROOT>/revised"; BASE = (1984, 2014)
OUT = "<PROJECT_ROOT>/tier1/figs/"
cell = pd.read_csv(R + "/data/hmfwa_cells_monthly.csv"); cell = cell[cell.cell == "Kainji"]
OBS = [784.16, 813.77, 747.86, 793.90, 720.41, 666.18, 632.00, 709.35, 746.30, 752.98, 740.50, 766.69]; OBS_MEAN = 739.51
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
hist = cell[(cell.scenario == "HIST") & cell.year.between(*BASE)]
clim = hist.groupby(["gcm", "month"]).flow_m3s.mean().unstack()
fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor("white")
ax.fill_between(range(12), clim.min().values, clim.max().values, color="#2980B9", alpha=.18, lw=0, label="HMF-WA 5-GCM range")
for g in clim.index: ax.plot(range(12), clim.loc[g].values, color="#2980B9", lw=.7, alpha=.5)
ax.plot(range(12), clim.mean().values, color="#2980B9", lw=2.4, marker="o", ms=4, label=f"HMF-WA ensemble mean, simulated flow (annual mean {clim.mean().mean():.0f} m³/s)")
ax.plot(range(12), OBS, color="#C0392B", lw=2.2, marker="s", ms=4, label=f"Observed turbine discharge, 1970-2010 (annual mean {OBS_MEAN:.1f} m³/s)")
ax.axvspan(-0.5, 2.5, color="0.5", alpha=.07); ax.axvspan(9.5, 11.5, color="0.5", alpha=.07)
ax.text(1, 40, "Nov–Mar window", ha="center", fontsize=8.5, color="0.35"); ax.text(10.5, 40, "Nov–Mar\nwindow", ha="center", fontsize=8.5, color="0.35")
ax.set_xticks(range(12)); ax.set_xticklabels(MON); ax.set_ylabel("Mean monthly flow at Kainji (m³/s)"); ax.set_xlabel("Month")
ax.set_ylim(0, 2900); ax.yaxis.grid(True, lw=.4, color="#DDD", ls="--"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig4_09_simulated_flow.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("annual mean", clim.mean().mean(), "range", clim.mean(axis=1).min(), clim.mean(axis=1).max())
