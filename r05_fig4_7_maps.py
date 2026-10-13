"""
=============================================================================
Percentage Change in Mean Precipitation — Historical vs Future
Model Historical Baseline vs Future Projection  |  Layout: 1 x 3
=============================================================================
Panel (a): Historical mean precipitation   (model, baseline period)
Panel (b): Future mean precipitation       (model, future period)
Panel (c): Percentage change ((future - historical) / historical x 100)
           with stippling where change is statistically significant (p < 0.05)

Shared colourbar below panels (a) and (b) - absolute precipitation
Diverging colourbar to the right of panel (c) - percentage change

REQUIREMENTS:
    pip install xarray rioxarray geopandas matplotlib cartopy numpy scipy
=============================================================================
"""

import sys
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# USER SETTINGS -- adjust all values in this block
# =============================================================================

# ── FILE PATHS ────────────────────────────────────────────────────────────────
HIST_FILE  = "<PROJECT_ROOT>/data/models/hist/Ensmean.nc"

SCENARIO_KEY = sys.argv[1] if len(sys.argv) > 1 else "ssp585"   # ssp245 or ssp585
PERIOD_KEY   = sys.argv[2] if len(sys.argv) > 2 else "far"      # near or far

FUTURE_FILE = f"<PROJECT_ROOT>/data/models/proj/{SCENARIO_KEY}/Ensmean_mon.nc"

SHAPEFILE   = "<PROJECT_ROOT>/shp/kainji_upstream_49.shp"

FUTURE_RANGE = {"near": (2030, 2060), "far": (2070, 2100)}[PERIOD_KEY]
SCENARIO_LABEL = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}[SCENARIO_KEY]

OUTPUT_FILE = f"<PROJECT_ROOT>/revised/plots/fig4_7_precip_pct_change_{SCENARIO_KEY}_{FUTURE_RANGE[0]}-{FUTURE_RANGE[1]}.png"

# ── VARIABLE AND COORDINATE NAMES ─────────────────────────────────────────────
MODEL_VAR = "pr"
LON_VAR   = "longitude"
LAT_VAR   = "latitude"

# ── HISTORICAL BASELINE PERIOD ────────────────────────────────────────────────
HIST_START = 1984
HIST_END   = 2014

# ── FUTURE PERIOD ─────────────────────────────────────────────────────────────
FUTURE_START, FUTURE_END = FUTURE_RANGE

SCENARIO = SCENARIO_LABEL

# ── SIGNIFICANCE ──────────────────────────────────────────────────────────────
ALPHA         = 0.05
STIPPLE_EVERY = 2
STIPPLE_SIZE  = 9.0
STIPPLE_COLOR = "black"
STIPPLE_ALPHA = 0.85

# ── COLOUR SCALES ─────────────────────────────────────────────────────────────
PREC_VMIN = 0
PREC_VMAX = 215
PREC_CMAP = "YlGnBu"

PCT_VMIN  = -5
PCT_VMAX  =  250
PCT_CMAP  = "RdBu_r"

# ── DAM MARKER ────────────────────────────────────────────────────────────────
KAINJI_LON = 4.60
KAINJI_LAT = 9.87

DPI = 300

# =============================================================================
# LOAD DATA
# =============================================================================

def load_period(filepath, var_name, year_start, year_end):
    ds = xr.open_mfdataset(filepath, combine="by_coords")
    rn = {}
    if LON_VAR in ds.coords and LON_VAR != "lon":
        rn[LON_VAR] = "lon"
    if LAT_VAR in ds.coords and LAT_VAR != "lat":
        rn[LAT_VAR] = "lat"
    if rn:
        ds = ds.rename(rn)
    da = ds[var_name]
    da = da.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=(da["lon"] + 180) % 360 - 180)
        da = da.sortby("lon")
    return da

print(f"Loading historical model ({HIST_START}-{HIST_END}) ...")
hist_da = load_period(HIST_FILE, MODEL_VAR, HIST_START, HIST_END)

print(f"Loading future model ({FUTURE_START}-{FUTURE_END}, {SCENARIO}) ...")
future_da = load_period(FUTURE_FILE, MODEL_VAR, FUTURE_START, FUTURE_END)

print(f"  Historical: {hist_da.sizes['time']} months")
print(f"  Future    : {future_da.sizes['time']} months")

print("Loading shapefile ...")
basin = gpd.read_file(SHAPEFILE)
if basin.crs is None:
    basin = basin.set_crs("EPSG:4326")
else:
    basin = basin.to_crs("EPSG:4326")

def clip_to_basin(da):
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip(basin.geometry, basin.crs, drop=True, all_touched=True)

try:
    future_regrid = future_da.interp(
        lon=hist_da["lon"], lat=hist_da["lat"], method="linear"
    )
    print("  Regridded future data to historical grid.")
except Exception:
    future_regrid = future_da
    print("  Grids already match -- no regridding needed.")

print("Computing period means ...")
hist_mean   = hist_da.mean(dim="time")
future_mean = future_regrid.mean(dim="time")

hist_mean_c   = clip_to_basin(hist_mean)
future_mean_c = clip_to_basin(future_mean)

print(f"  Historical mean : {float(hist_mean_c.mean()):.2f} mm/month  "
      f"max = {float(hist_mean_c.max()):.2f}")
print(f"  Future mean     : {float(future_mean_c.mean()):.2f} mm/month  "
      f"max = {float(future_mean_c.max()):.2f}")

pct_change_raw = ((future_mean - hist_mean) / hist_mean) * 100
pct_change_c   = clip_to_basin(pct_change_raw)

print(f"  % Change: mean = {float(pct_change_c.mean()):.2f}%  "
      f"range = [{float(pct_change_c.min()):.2f}, {float(pct_change_c.max()):.2f}]%")

print("Computing significance (Welch's t-test) ...")

hist_vals   = hist_da.values
future_vals = future_regrid.values

if hist_vals.ndim == 3 and future_vals.ndim == 3:
    _, pvals = stats.ttest_ind(
        future_vals, hist_vals,
        axis=0,
        equal_var=False,
        nan_policy="omit"
    )
else:
    pvals = np.ones(hist_vals.shape[1:])

pval_da = xr.DataArray(
    pvals,
    dims=["lat", "lon"],
    coords={"lat": hist_da["lat"].values, "lon": hist_da["lon"].values}
)
pval_da = pval_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
pval_da = pval_da.rio.write_crs("EPSG:4326")
pval_clip = pval_da.rio.clip(basin.geometry, basin.crs, drop=True, all_touched=True)

sig_mask = (pval_clip < ALPHA).values

sig_pct = 100 * sig_mask.sum() / np.isfinite(pval_clip.values).sum()   # revised: basin cells in numerator AND denominator
print(f"  Significant grid cells: {sig_mask.sum()} ({sig_pct:.1f}%)")

lons    = hist_mean_c["lon"].values
lats    = hist_mean_c["lat"].values
lon_min = float(lons.min()) - 0.8
lon_max = float(lons.max()) + 0.8
lat_min = float(lats.min()) - 0.8
lat_max = float(lats.max()) + 0.8

print("Building figure ...")

proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(20, 7.5))
fig.patch.set_facecolor("#F7F9FC")

gs = GridSpec(
    2, 4,
    figure=fig,
    wspace=0.10, hspace=0.12,
    left=0.04, right=0.95,
    top=0.88,  bottom=0.14,
    width_ratios=[1, 1, 1, 0.045],
    height_ratios=[1, 0.045]
)

ax_hist   = fig.add_subplot(gs[0, 0], projection=proj)
ax_future = fig.add_subplot(gs[0, 1], projection=proj)
ax_change = fig.add_subplot(gs[0, 2], projection=proj)

cax_prec  = fig.add_subplot(gs[1, 0:2])
cax_change = fig.add_subplot(gs[0:2, 3])

def draw_panel(ax, data, cmap, vmin, vmax, title, panel_letter,
               show_left=True, show_bottom=True, stipple=None):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    ax.add_feature(cfeature.LAND,      facecolor="#F0EDE4", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#D6EAF8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,  edgecolor="#444444", zorder=4)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4,  edgecolor="#999999",
                   linestyle="--", zorder=4)
    ax.add_feature(cfeature.RIVERS,    linewidth=0.35, edgecolor="#5DADE2",
                   alpha=0.5, zorder=3)
    ax.add_feature(cfeature.LAKES,     facecolor="#D6EAF8", edgecolor="#5DADE2",
                   linewidth=0.3, zorder=3)

    img = data.drop_vars("spatial_ref", errors="ignore").plot(
        ax=ax, transform=proj,
        cmap=cmap, vmin=vmin, vmax=vmax,
        add_colorbar=False, zorder=2
    )

    basin.boundary.plot(ax=ax, color="#C0392B", linewidth=1.3,
                        transform=proj, zorder=5)

    ax.plot(KAINJI_LON, KAINJI_LAT, marker="^",
            color="#E74C3C", markersize=7,
            transform=proj, zorder=6)

    if stipple is not None:
        lon2d, lat2d = np.meshgrid(
            data["lon"].values, data["lat"].values
        )
        subsample = np.zeros_like(stipple, dtype=bool)
        subsample[::STIPPLE_EVERY, ::STIPPLE_EVERY] = True
        sig = stipple & subsample

        if sig.shape == lon2d.shape:
            ax.scatter(
                lon2d[sig], lat2d[sig],
                s=STIPPLE_SIZE, c=STIPPLE_COLOR,
                alpha=STIPPLE_ALPHA,
                transform=proj, zorder=7,
                linewidths=0
            )

    gl = ax.gridlines(draw_labels=True,
                      linewidth=0.3, color="grey",
                      alpha=0.4, linestyle="--")
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = show_left
    gl.bottom_labels = show_bottom
    gl.xlabel_style  = {"size": 14, "color": "black"}
    gl.ylabel_style  = {"size": 14, "color": "black"}
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER

    ax.set_title(title, fontsize=13, fontweight="bold",
                 color="#1A252F", pad=7)

    ax.text(0.03, 0.96, panel_letter,
            transform=ax.transAxes,
            fontsize=14, va="top", ha="left", color="black")

    return img

img_hist = draw_panel(
    ax_hist, hist_mean_c,
    PREC_CMAP, PREC_VMIN, PREC_VMAX,
    f"Historical Mean\n({HIST_START}-{HIST_END})",
    "(a)", show_left=True, show_bottom=True
)

img_future = draw_panel(
    ax_future, future_mean_c,
    PREC_CMAP, PREC_VMIN, PREC_VMAX,
    f"Future Mean\n({FUTURE_START}-{FUTURE_END}, {SCENARIO})",
    "(b)", show_left=False, show_bottom=True
)

img_change = draw_panel(
    ax_change, pct_change_c,
    PCT_CMAP, PCT_VMIN, PCT_VMAX,
    "Percentage Change\nFuture Mean - Historical Mean",
    "(c)", show_left=False, show_bottom=True,
    stipple=sig_mask
)

cb_prec = fig.colorbar(img_hist, cax=cax_prec, orientation="horizontal")
cb_prec.set_label("Mean Monthly Precipitation (mm / month)",
                  fontsize=11, color="#333333", labelpad=5)
cb_prec.ax.tick_params(labelsize=10, colors="#333333")
ticks_prec = np.linspace(PREC_VMIN, PREC_VMAX, 9)
cb_prec.set_ticks(ticks_prec)
cb_prec.set_ticklabels([f"{int(t)}" for t in ticks_prec])

cb_change = fig.colorbar(img_change, cax=cax_change, orientation="vertical")
cb_change.set_label("Precipitation Change (%)",
                    fontsize=11, color="#333333", labelpad=8)
cb_change.ax.tick_params(labelsize=10, colors="#333333")
ticks_pct = np.linspace(PCT_VMIN, PCT_VMAX, 11)
cb_change.set_ticks(ticks_pct)
cb_change.set_ticklabels([f"{int(t)}" for t in ticks_pct])

cb_change.ax.axhline(0, color="black", linewidth=1.0, linestyle="-")

legend_elements = [
    mpatches.Patch(facecolor="none", edgecolor="#C0392B",
                   linewidth=1.3, label="Kainji Catchment Boundary"),
    plt.Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#E74C3C", markersize=9,
               label="Kainji Dam"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=STIPPLE_COLOR,
               markersize=5, alpha=STIPPLE_ALPHA,
               label=f"Significant change (p < {ALPHA})"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=9.5, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01),
           edgecolor="#CCCCCC")

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nFigure saved to: {OUTPUT_FILE}")

print("\n-- Summary --------------------------------------------------------")
print(f"  Historical mean : {float(hist_mean_c.mean()):.2f} mm/month")
print(f"  Future mean     : {float(future_mean_c.mean()):.2f} mm/month")
print(f"  Mean % change   : {float(pct_change_c.mean()):.2f}%")
print(f"  Max % increase  : {float(pct_change_c.max()):.2f}%")
print(f"  Max % decrease  : {float(pct_change_c.min()):.2f}%")
print(f"  Significant cells: {sig_pct:.1f}% of basin")
print("---------------------------------------------------------------------")

# ---- revised: corrected statistics (area-weighted ratio of means; consistent significance denominator) -> JSON
import json
_w = np.cos(np.deg2rad(hist_mean_c["lat"]))
_hm = float(hist_mean_c.weighted(_w).mean(("lat", "lon"))); _fm = float(future_mean_c.weighted(_w).mean(("lat", "lon")))
_stats = dict(scenario=SCENARIO_KEY, period=PERIOD_KEY, hist_mean_mm_month=_hm, future_mean_mm_month=_fm,
              pct_change_of_mean=100 * (_fm - _hm) / _hm, mean_of_pixel_pct=float(pct_change_c.mean()),
              median_pixel_pct=float(np.nanmedian(pct_change_c.values)), max_pixel_pct=float(pct_change_c.max()), min_pixel_pct=float(pct_change_c.min()),
              n_cells=int(np.isfinite(pct_change_c.values).sum()), sig_share_pct=float(sig_pct),
              share_cells_baseline_lt10=float(100 * (hist_mean_c < 10).sum() / np.isfinite(hist_mean_c.values).sum()))
json.dump(_stats, open(f"<PROJECT_ROOT>/revised/data/map_stats_{SCENARIO_KEY}_{PERIOD_KEY}.json", "w"), indent=1)
print(_stats)
