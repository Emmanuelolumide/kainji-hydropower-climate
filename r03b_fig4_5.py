"""
=============================================================================
Mean Monthly Precipitation Climatology — 1981–2014
CHIRPS vs 10 CMIP6 Models  |  All Regions  |  Layout: 2 × 2
=============================================================================
Four panels:
  (a) Kainji contributing catchment
  (b) Upper Niger (Guinea Highlands)
  (c) Middle Niger (Sahel Zone)
  (d) Lower Niger (Near Kainji)

Each panel shows:
  • CHIRPS v2.0 — thick black line (reference)
  • 10 CMIP6 models — individual coloured lines
  • Ensemble mean — thick coloured dashed line
  • Shaded uncertainty band — min to max across all models

REQUIREMENTS:
    pip install xarray rioxarray geopandas matplotlib numpy shapely
=============================================================================
"""

import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# USER SETTINGS
# =============================================================================

# ── CHIRPS FILE ───────────────────────────────────────────────────────────────
CHIRPS_FILE = "<PROJECT_ROOT>/data/chirps/chirps_wa_mon.nc"
CHIRPS_VAR  = "pr"

# ── INDIVIDUAL MODEL FILES ────────────────────────────────────────────────────
MODEL_FILES = {
   "ACCESS-CM2"      : "<PROJECT_ROOT>/data/models/hist/Access-cm2.nc",
    "ACCESS-ESM1-5"   : "<PROJECT_ROOT>/data/models/hist/Access-esm1.nc",
    "CanESM5"         : "<PROJECT_ROOT>/data/models/hist/Canesm5.nc",
    "EC-Earth3"       : "<PROJECT_ROOT>/data/models/hist/Ec-earth3.nc",
    "INM-CM4-8"       : "<PROJECT_ROOT>/data/models/hist/Inm-cm4.nc",
    "INM-CM5-0"       : "<PROJECT_ROOT>/data/models/hist/Inm-cm5.nc",
    "IPSL-CM6A-LR"    : "<PROJECT_ROOT>/data/models/hist/Ipsl-cm6a-lr.nc",
    "KIOST-ESM"       : "<PROJECT_ROOT>/data/models/hist/Kiost.nc",
    "NorESM2-MM"          : "<PROJECT_ROOT>/data/models/hist/Noresm2.nc",
    "MPI-ESM1-2-LR"   : "<PROJECT_ROOT>/data/models/hist/Mpi-esm1.nc",
}
MODEL_VAR = "pr"

# ── COORDINATE NAMES ──────────────────────────────────────────────────────────
LON_VAR = "longitude"
LAT_VAR = "latitude"

# ── SHAPEFILE ─────────────────────────────────────────────────────────────────
SHAPEFILE = "<PROJECT_ROOT>/shp/kainji_upstream_49.shp"

# ── CLIMATOLOGY PERIOD ────────────────────────────────────────────────────────
YEAR_START = 1981    # <── adjust start year
YEAR_END   = 2014    # <── adjust end year

# ── REGION DEFINITIONS ────────────────────────────────────────────────────────
# bbox = (lon_min, lon_max, lat_min, lat_max)
# Use None for the whole catchment (uses the full basin shapefile)
REGIONS = {
    "Kainji Contributing\nCatchment"      : {"bbox": None},
    "Upper Niger\n(Guinea Highlands)"   : {"bbox": (-12.0, -2.0, 8.0, 13.0)},
    "Middle Niger\n(Sahel Zone)"        : {"bbox": (-4.0,   4.0, 13.0, 18.0)},
    "Lower Niger\n(Near Kainji)"        : {"bbox": (2.0,    8.0,  9.87, 13.0)},
}

# ── MODEL COLOURS ─────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "ACCESS-CM2"      : "#E74C3C",
    "ACCESS-ESM1-5"   : "#E67E22",
    "EC-Earth3"       : "#F1C40F",
    "INM-CM4-8"       : "#1ABC9C",
    "INM-CM5-0"       : "#3498DB",
    "IPSL-CM6A-LR"    : "#9B59B6",
    "KIOST-ESM"       : "#E91E63",
    "NorESM2-MM"          : "#00BCD4",
    "CanESM5"         : "#8E44AD",
    "MPI-ESM1-2-LR"   : "#FF5722",
}

# ── APPEARANCE ────────────────────────────────────────────────────────────────
CHIRPS_LW     = 2.8    # CHIRPS line width
MODEL_LW      = 1.2    # individual model line width
ENS_LW        = 2.2    # ensemble mean line width
BAND_ALPHA    = 0.15   # shaded uncertainty band transparency
MODEL_ALPHA   = 0.75   # individual model line transparency

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "<PROJECT_ROOT>/revised/plots/fig4_5_monthly_climatology_2x2.png"
DPI = 300

# =============================================================================
# LOAD DATA
# =============================================================================

def load_da(filepath, var_name):
    ds = xr.open_mfdataset(filepath, combine="by_coords")
    rn = {}
    if LON_VAR in ds.coords and LON_VAR != "lon":
        rn[LON_VAR] = "lon"
    if LAT_VAR in ds.coords and LAT_VAR != "lat":
        rn[LAT_VAR] = "lat"
    if rn:
        ds = ds.rename(rn)
    da = ds[var_name]
    da = da.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=(da["lon"] + 180) % 360 - 180)
        da = da.sortby("lon")
    return da

print("Loading CHIRPS ...")
chirps_da = load_da(CHIRPS_FILE, CHIRPS_VAR)

model_das = {}
for name, path in MODEL_FILES.items():
    print(f"  Loading {name} ...")
    model_das[name] = load_da(path, MODEL_VAR)

# =============================================================================
# LOAD SHAPEFILE
# =============================================================================

print("Loading shapefile ...")
basin = gpd.read_file(SHAPEFILE)
if basin.crs is None:
    basin = basin.set_crs("EPSG:4326")
else:
    basin = basin.to_crs("EPSG:4326")

# =============================================================================
# CLIPPING HELPERS
# =============================================================================

def clip_full_basin(da):
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip(basin.geometry, basin.crs,
                       drop=True, all_touched=True)

def clip_subregion(da, bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
    sub_gdf   = basin.copy()
    sub_gdf   = sub_gdf[sub_gdf.geometry.intersects(bbox_poly)]
    sub_gdf   = gpd.GeoDataFrame(
        geometry=[sub_gdf.geometry.intersection(bbox_poly).union_all()],
        crs="EPSG:4326"
    )
    da_sub = da.sel(
        lon=slice(lon_min - 0.1, lon_max + 0.1),
        lat=slice(lat_min - 0.1, lat_max + 0.1)
    )
    da_sub = da_sub.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da_sub = da_sub.rio.write_crs("EPSG:4326")
    return da_sub.rio.clip(sub_gdf.geometry, sub_gdf.crs,
                           drop=True, all_touched=True)

def area_mean(da):
    w = np.cos(np.deg2rad(da["lat"]))      # cos-latitude area weights (Section 3.5.2)
    return da.weighted(w).mean(dim=["lon", "lat"])

# =============================================================================
# COMPUTE MONTHLY CLIMATOLOGY PER REGION
# =============================================================================

def monthly_climatology(ts_da):
    """Return 12-element array: mean for each calendar month."""
    return np.array([
        float(ts_da.sel(time=ts_da["time.month"] == m).mean())
        for m in range(1, 13)
    ])

print("\nComputing monthly climatologies ...")
MONTHS   = np.arange(1, 13)
MON_LBLS = ["J","F","M","A","M","J","J","A","S","O","N","D"]

all_region_data = {}

for reg_name, cfg in REGIONS.items():
    print(f"\n  Region: {reg_name.replace(chr(10),' ')}")
    bbox = cfg["bbox"]

    # ── CHIRPS climatology ────────────────────────────────────────────────────
    if bbox is None:
        ch_clipped = clip_full_basin(chirps_da)
    else:
        ch_clipped = clip_subregion(chirps_da, bbox)

    chirps_clim = monthly_climatology(area_mean(ch_clipped))
    print(f"    CHIRPS annual mean: {chirps_clim.mean():.2f} mm/month")

    # ── Model climatologies ───────────────────────────────────────────────────
    model_clims = {}
    for m_name, m_da in model_das.items():
        if bbox is None:
            m_clipped = clip_full_basin(m_da)
        else:
            m_clipped = clip_subregion(m_da, bbox)
        model_clims[m_name] = monthly_climatology(area_mean(m_clipped))

    # ── Ensemble statistics ───────────────────────────────────────────────────
    stacked   = np.array(list(model_clims.values()))   # (n_models, 12)
    ens_mean  = stacked.mean(axis=0)
    ens_min   = stacked.min(axis=0)
    ens_max   = stacked.max(axis=0)

    all_region_data[reg_name] = {
        "chirps"      : chirps_clim,
        "models"      : model_clims,
        "ens_mean"    : ens_mean,
        "ens_min"     : ens_min,
        "ens_max"     : ens_max,
    }

# =============================================================================
# FIGURE — 2 × 2
# =============================================================================

print("\nBuilding figure ...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("white")

gs = GridSpec(
    2, 2,
    figure=fig,
    wspace=0.30,
    hspace=0.40,
    left=0.08, right=0.97,
    top=0.91,  bottom=0.14,
)

reg_names  = list(REGIONS.keys())
panel_lbls = ["(a)", "(b)", "(c)", "(d)"]
axes       = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
]

for idx, reg_name in enumerate(reg_names):
    ax   = axes[idx]
    data = all_region_data[reg_name]

    # ── Uncertainty band (min–max across models) ──────────────────────────────
    ax.fill_between(
        MONTHS,
        data["ens_min"],
        data["ens_max"],
        color="#888888",
        alpha=BAND_ALPHA,
        label="Model spread (min–max)",
        zorder=2
    )

    # ── Individual model lines ────────────────────────────────────────────────
    for m_name, m_clim in data["models"].items():
        ax.plot(
            MONTHS, m_clim,
            color=MODEL_COLORS.get(m_name, "#888888"),
            linewidth=MODEL_LW,
            alpha=MODEL_ALPHA,
            label=m_name,
            zorder=3
        )

    # ── Ensemble mean ─────────────────────────────────────────────────────────
    ax.plot(
        MONTHS, data["ens_mean"],
        color="red",
        linewidth=ENS_LW,
        linestyle="--",
        label="Ensemble mean",
        zorder=4
    )

    # ── CHIRPS — thick black line ─────────────────────────────────────────────
    ax.plot(
        MONTHS, data["chirps"],
        color="black",
        linewidth=CHIRPS_LW,
        label="CHIRPS v2.0",
        zorder=5
    )


    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.set_xticks(MONTHS)
    ax.set_xticklabels(MON_LBLS, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Month", fontsize=12, color="black", labelpad=5)
    ax.set_ylabel("Precipitation (mm / month)",
                  fontsize=12, color="black", labelpad=5)

    ax.yaxis.grid(True, linewidth=0.45,
                  color="#DDDDDD", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAAAAA")

    # ── Panel letter ──────────────────────────────────────────────────────────
    ax.text(0.03, 0.97, panel_lbls[idx],
            transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            va="top", ha="left", color="black")

    # ── Panel title ───────────────────────────────────────────────────────────
    ax.set_title(reg_name,
                 fontsize=12, fontweight="bold",
                 color="#1A252F", pad=7)

    # ── Annotate annual means ─────────────────────────────────────────────────
    ch_ann  = data["chirps"].mean()
    ens_ann = data["ens_mean"].mean()
    ax.text(0.02, 0.87,
            f"CHIRPS mean: {ch_ann:.1f} mm/mon\n"
            f"Ensemble mean: {ens_ann:.1f} mm/mon",
            transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            color="#333333",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="#CCCCCC", pad=3,
                      boxstyle="round,pad=0.4"))

# =============================================================================
# SHARED LEGEND (below all panels)
# =============================================================================

# Build legend entries
legend_handles = [
    plt.Line2D([0], [0], color="black",   linewidth=CHIRPS_LW,
               label="CHIRPS v2.0"),
    plt.Line2D([0], [0], color="red", linewidth=ENS_LW,
               linestyle="--", label="Ensemble mean"),
    #mpatches.Patch(facecolor="#888888",   alpha=0.35,
     #              label="Model spread (min–max)"),
]
# Individual models
for m_name, col in MODEL_COLORS.items():
    legend_handles.append(
        plt.Line2D([0], [0], color=col, linewidth=1.5,
                   alpha=0.9, label=m_name)
    )

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=7,
    fontsize=8.5,
    framealpha=0.95,
    bbox_to_anchor=(0.5, 0.04),
    edgecolor="#CCCCCC",
    columnspacing=0.8,
    handlelength=1.8,
)

# =============================================================================
# MAIN TITLE
# =============================================================================
'''
fig.suptitle(
    f"Mean Monthly Precipitation Climatology  |  Niger Basin Sub-Regions  |  "
    f"CHIRPS v2.0 and 10 CMIP6 Models  |  {YEAR_START}–{YEAR_END}",
    fontsize=13, fontweight="bold",
    color="#1A252F", y=0.97
)
'''
# =============================================================================
# SAVE
# =============================================================================

plt.savefig(OUTPUT_FILE, dpi=DPI,
            bbox_inches="tight",
            facecolor="white")
print(f"\nFigure saved to: {OUTPUT_FILE}")
plt.show()
