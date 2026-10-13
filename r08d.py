"""HMF-WA figures on the Kainji upstream mask: Fig 4.9 seasonal cycle vs observed, Fig 4.10 historical mean-flow map, Fig 4.11 Kainji-cell annual flow 1980-2014,
Figs 4.12/4.13 % change maps (SSP2-4.5 / SSP5-8.5), Fig A.3 annual flow 1950-2100."""
import sys, os, pickle, warnings, numpy as np, pandas as pd, xarray as xr, geopandas as gpd, rioxarray
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mc
import cartopy.crs as ccrs, cartopy.feature as cfeature
from config import *
NOTITLE = "--notitle" in sys.argv
OUTD = "./output/"
os.makedirs(OUTD, exist_ok=True)
def title(ax, *a, **k):
    if not NOTITLE: ax.set_title(*a, **k)
def suptitle(fig, *a, **k):
    if not NOTITLE: fig.suptitle(*a, **k)
warnings.filterwarnings("ignore")
ds = xr.open_dataset(R + "/data/hmfwa_period_means.nc").load()
mask = gpd.read_file(MASK_SHP).to_crs("EPSG:4326")
niger = gpd.read_file(A + "/shp/niger_basin.shp").to_crs("EPSG:4326")
GCMS = list(ds.gcm.values)
cell = pd.read_csv(R + "/data/hmfwa_cells_monthly.csv"); cell = cell[cell.cell == "Kainji"]
OBS = [784.16, 813.77, 747.86, 793.90, 720.41, 666.18, 632.00, 709.35, 746.30, 752.98, 740.50, 766.69]
OBS_MEAN = 739.51
def clip(x): return x.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs("EPSG:4326").rio.clip(mask.geometry, mask.crs, drop=True, all_touched=True)
def setup(ax, ext):
    ax.set_extent(ext, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#F0EDE4"); ax.add_feature(cfeature.OCEAN, facecolor="#D6EAF8")
    ax.add_feature(cfeature.BORDERS, linewidth=.4, linestyle="--", edgecolor="#888"); ax.add_feature(cfeature.COASTLINE, linewidth=.5)
    niger.boundary.plot(ax=ax, color="#999999", lw=0.6, ls=":", transform=ccrs.PlateCarree())
    mask.boundary.plot(ax=ax, color="#C0392B", lw=1.3, transform=ccrs.PlateCarree())
    ax.plot(4.65, 9.85, "^", color="#E74C3C", ms=10, mec="k", transform=ccrs.PlateCarree(), zorder=6)
    gl = ax.gridlines(draw_labels=True, lw=.3, color="grey", alpha=.5, ls="--"); gl.top_labels = gl.right_labels = False
def rivers(ax, da, **kw):
    """Draw 0.1-degree river cells as enlarged squares (a 0.1-degree cell is <1 pixel wide at this map scale)."""
    v = da.values; yy, xx = np.where(np.isfinite(v))
    return ax.scatter(da.longitude.values[xx], da.latitude.values[yy], c=v[yy, xx], marker="s", s=kw.pop("s", 7), linewidths=0, transform=ccrs.PlateCarree(), zorder=4, **kw)
def ens(scn, per, sea): return ds.flow.sel(scn=scn, period=per, season=sea).mean("gcm")
bounds = mask.total_bounds; ext = [bounds[0] - 1, bounds[2] + 1, bounds[1] - 0.4, bounds[3] + 1]

# ---- Fig 4.10 historical mean-flow map (1980-2014), Kainji mask
h = clip(ens("HIST", "hist80", "annual")).where(lambda x: x > 0.5)
fig = plt.figure(figsize=(8.5, 7)); ax = fig.add_subplot(111, projection=ccrs.PlateCarree()); setup(ax, ext)
im = rivers(ax, h, cmap="YlGnBu", norm=mc.Normalize(vmin=0, vmax=float(h.max())))
fig.colorbar(im, ax=ax, orientation="horizontal", pad=.07, shrink=.85).set_label("Mean river flow, 1980-2014 (m$^3$/s)")
title(ax, "HMF-WA historical annual mean flow, ensemble mean of 5 GCMs (1980-2014)\nKainji contributing catchment", fontweight="bold", fontsize=11.5)
ax.plot([], [], "^", color="#E74C3C", mec="k", ms=9, label="Kainji Dam cell (9.85°N, 4.65°E)"); ax.plot([], [], color="#C0392B", lw=1.3, label="Kainji Catchment Boundary"); ax.legend(loc="lower right", fontsize=8, framealpha=.9)
fig.savefig(OUTD + "fig4_10_hmfwa_hist_annual_mean_flow_map.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- Fig 4.12 / 4.13 % change maps (annual mean), Kainji mask, stippling = >=4 of 5 GCMs agree on sign
base = ds.flow.sel(scn="HIST", period="base", season="annual")
bm = clip(base.mean("gcm"))
stats_out = {}
for s, lab, fn in (("SSP245", "SSP2-4.5", "fig4_12"), ("SSP585", "SSP5-8.5", "fig4_13")):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.4), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (t, k) in zip(axs, (("Mid-century 2030-2060", "near"), ("Late-century 2070-2100", "far"))):
        fut = ds.flow.sel(scn=s, period=k, season="annual")
        pc_g = 100 * (fut - base) / base                       # per GCM
        pc_e = pc_g.mean("gcm")                                 # ensemble mean of the per-GCM % changes (as in Table 4.5)
        agree = (np.sign(pc_g) == np.sign(pc_e)).sum("gcm")
        pc = clip(pc_e).where(bm > 5); ag = clip(agree).where(bm > 5)
        setup(ax, ext)
        im = rivers(ax, pc, cmap="RdBu", norm=mc.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=100))
        yy, xx = np.where((ag.values < 4) & np.isfinite(pc.values))
        ax.scatter(ag.longitude.values[xx], ag.latitude.values[yy], s=14, c="k", marker="x", linewidths=.7, transform=ccrs.PlateCarree(), zorder=5)
        kc = float(pc_e.sel(latitude=9.85, longitude=4.65, method="nearest"))
        ax.set_title(f"{t}: change at Kainji cell {kc:+.0f}%", fontweight="bold", fontsize=11)
        stats_out[f"{s}_{k}"] = dict(kainji=kc, river_cells=int(np.isfinite(pc.values).sum()), median=float(np.nanmedian(pc.values)), p10=float(np.nanpercentile(pc.values, 10)), p90=float(np.nanpercentile(pc.values, 90)),
                                      share_agree4=float(np.mean(ag.values[np.isfinite(pc.values)] >= 4)))
    fig.colorbar(im, ax=axs, orientation="horizontal", pad=.08, shrink=.5, extend="both").set_label(f"Ensemble-mean % change in annual mean flow vs 1984-2014 ({lab}); × = fewer than 4 of 5 GCMs agree on the sign of change")
    suptitle(fig, f"HMF-WA {lab}: change in annual mean river flow within the Kainji contributing catchment, ensemble of 5 GCMs", fontweight="bold", y=.98)
    fig.savefig(OUTD + f"{fn}_hmfwa_pct_change_{s.lower()}.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
import json; json.dump(stats_out, open(R + "/data/hmfwa_map_stats.json", "w"), indent=1); print(json.dumps(stats_out, indent=1))

# ---- Fig 4.9 seasonal cycle HMF-WA vs observed
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
hist = cell[(cell.scenario == "HIST") & cell.year.between(*BASE)]
clim = hist.groupby(["gcm", "month"]).flow_m3s.mean().unstack()
fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor("white")
ax.fill_between(range(12), clim.min().values, clim.max().values, color="#2980B9", alpha=.18, lw=0, label="HMF-WA 5-GCM range")
for g in clim.index: ax.plot(range(12), clim.loc[g].values, color="#2980B9", lw=.7, alpha=.5)
ax.plot(range(12), clim.mean().values, color="#2980B9", lw=2.4, marker="o", ms=4, label=f"HMF-WA ensemble mean, natural flow (annual mean {clim.mean().mean():.0f} m³/s)")
ax.plot(range(12), OBS, color="#C0392B", lw=2.2, marker="s", ms=4, label=f"Observed turbine discharge, 1970-2010 (annual mean {OBS_MEAN:.1f} m³/s)")
ax.axvspan(-0.5, 2.5, color="0.5", alpha=.07); ax.axvspan(9.5, 11.5, color="0.5", alpha=.07)
ax.text(1, 40, "Nov–Mar window", ha="center", fontsize=8.5, color="0.35"); ax.text(10.5, 40, "Nov–Mar\nwindow", ha="center", fontsize=8.5, color="0.35")
ax.set_xticks(range(12)); ax.set_xticklabels(MON); ax.set_ylabel("Mean monthly flow at Kainji (m³/s)"); ax.set_xlabel("Month")
ax.set_ylim(0, 2900); ax.yaxis.grid(True, lw=.4, color="#DDD", ls="--"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9, loc="upper left")
title(ax, "Mean Monthly Cycle at Kainji Dam: HMF-WA Simulated Natural Flow (1984-2014) vs Observed Turbine Discharge", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(OUTD + "fig4_09_hmfwa_seasonal_cycle_vs_observed.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- Fig 4.11 Kainji-cell annual mean flow 1980-2014 by GCM
ann = cell.groupby(["gcm", "scenario", "year"]).flow_m3s.mean().reset_index()
T = ann[(ann.scenario == "HIST") & ann.year.between(1980, 2014)].pivot(index="year", columns="gcm", values="flow_m3s")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
cols = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD", "#8C564B"]
for c, col in zip(T.columns, cols): ax.plot(T.index, T[c], lw=.9, alpha=.7, color=col, label=c)
ax.plot(T.index, T.mean(axis=1), "k", lw=2.4, label="Ensemble mean"); ax.axhline(OBS_MEAN, color="#C0392B", ls="--", lw=1.2, label="Observed turbine discharge 739.5 m³/s")
ax.set_xlabel("Year"); ax.set_ylabel("Annual mean flow (m³/s)"); title(ax, "HMF-WA Historical Annual Mean Flow at the Kainji Dam Cell (9.85°N, 4.65°E), 1980-2014", fontweight="bold", fontsize=11)
ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.yaxis.grid(True, lw=.4, color="#DDD", ls="--"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUTD + "fig4_11_hmfwa_kainji_annual_flow_1980_2014.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- Fig A.3 annual mean flow 1950-2100
fig, ax = plt.subplots(figsize=(12, 6)); fig.patch.set_facecolor("white")
H = ann[ann.scenario == "HIST"].pivot(index="year", columns="gcm", values="flow_m3s")
ax.fill_between(H.index, H.min(axis=1), H.max(axis=1), color="0.5", alpha=.25, lw=0); ax.plot(H.index, H.mean(axis=1), color="0.2", lw=1.6, label="Historical ensemble mean (1950-2014)")
for s, col, lab in (("SSP245", "#0057E7", "SSP2-4.5"), ("SSP585", "#E60000", "SSP5-8.5")):
    S = ann[ann.scenario == s].pivot(index="year", columns="gcm", values="flow_m3s")
    ax.fill_between(S.index, S.min(axis=1), S.max(axis=1), color=col, alpha=.15, lw=0); ax.plot(S.index, S.mean(axis=1), color=col, lw=1.6, label=f"{lab} ensemble mean (2015-2100)")
    ax.plot(S.index, S.mean(axis=1).rolling(11, center=True).mean(), color=col, lw=2.6, ls="-", alpha=.9)
ax.plot(H.index, H.mean(axis=1).rolling(11, center=True).mean(), color="k", lw=2.6)
ax.axhline(OBS_MEAN, color="#C0392B", ls="--", lw=1.1, label="Observed turbine discharge 739.5 m³/s"); ax.axvline(2014.5, color="0.5", ls=":", lw=1)
ax.set_xlabel("Year"); ax.set_ylabel("Annual mean flow (m³/s)"); ax.set_xlim(1950, 2100); ax.set_ylim(bottom=0)
title(ax, "HMF-WA Historical and Projected Annual Mean Flow at Kainji Dam (Kainji Cell)\nThin line: annual ensemble mean; thick line: 11-year running mean; shading: 5-GCM range", fontweight="bold", fontsize=11, loc="left")
ax.yaxis.grid(True, lw=.4, color="#DDD", ls="--"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(OUTD + "figA3_hmfwa_annual_flow_1950_2100.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("done")
