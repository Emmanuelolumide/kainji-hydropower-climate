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
import warnings
warnings.filterwarnings("ignore")
# =============================================================================
# USER SETTINGS — adjust everything in this block
# =============================================================================

# ── CHIRPS FILE ───────────────────────────────────────────────────────────────
CHIRPS_FILE = "<PROJECT_ROOT>/data/chirps/chirps_wa_mon.nc"
CHIRPS_VAR  = "pr"

# ── ENSEMBLE MEAN FILE ────────────────────────────────────────────────────────
ENSEMBLE_FILE = "<PROJECT_ROOT>/data/models/hist/Ensmean.nc"
ENSEMBLE_VAR  = "pr"

# ── INDIVIDUAL MODEL FILES ────────────────────────────────────────────────────
# Add the full path to each model's monthly NetCDF file (already in mm/month)
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
MODEL_VAR = "pr"   # variable name in every model file

# ── COORDINATE NAMES ──────────────────────────────────────────────────────────
LON_VAR = "longitude"
LAT_VAR = "latitude"

# ── SHAPEFILE ─────────────────────────────────────────────────────────────────
SHAPEFILE = "<PROJECT_ROOT>/shp/kainji_upstream_49.shp"

# ── CLIMATOLOGY PERIOD ────────────────────────────────────────────────────────
YEAR_START = 1981    # <── adjust start year here
YEAR_END   = 2014    # <── adjust end year here

# ── SHARED COLOUR SCALE ───────────────────────────────────────────────────────
VMIN = 0      # mm/month — keep at 0
VMAX = 250    # mm/month — annual mean is lower than seasonal peak; adjust as needed
CMAP = "YlGnBu"

# ── DAM MARKER ────────────────────────────────────────────────────────────────
KAINJI_LON = 4.60
KAINJI_LAT = 9.87

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "<PROJECT_ROOT>/revised/plots/fig4_4_annual_mean_CHIRPS_ensemble_models.png"
DPI = 300

# =============================================================================
# PANEL ORDER — CHIRPS and Ensemble first, then 10 models in order
# =============================================================================
PANEL_ORDER = (
    [("CHIRPS",        "CHIRPS"),
     ("Ensemble Mean", "ENSEMBLE")]
    + [(name, name) for name in MODEL_FILES.keys()]
)
N_ROWS, N_COLS = 3, 4

PANEL_LETTERS = [
    "(a)", "(b)", "(c)", "(d)",
    "(e)", "(f)", "(g)", "(h)",
    "(i)", "(j)", "(k)", "(l)",
]
# =============================================================================
# LOAD DATA
# =============================================================================

def load_da(filepath, var_name):
    """Load a DataArray, rename coords to lon/lat, subset to period."""
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

    # Fix 0–360 longitudes if present
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=(da["lon"] + 180) % 360 - 180)
        da = da.sortby("lon")
    return da

print("Loading data ...")
DATA = {}
DATA["CHIRPS"]   = load_da(CHIRPS_FILE,   CHIRPS_VAR)
DATA["ENSEMBLE"] = load_da(ENSEMBLE_FILE, ENSEMBLE_VAR)
for name, path in MODEL_FILES.items():
    print(f"  {name}")
    DATA[name] = load_da(path, MODEL_VAR)

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
# CLIP TO BASIN
# =============================================================================

def clip_to_basin(da):
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip(basin.geometry, basin.crs, drop=True, all_touched=True)

# =============================================================================
# COMPUTE ANNUAL MEAN AND CLIP
# =============================================================================

print("\nComputing annual means ...")
MEANS = {}
for key, da in DATA.items():
    MEANS[key] = clip_to_basin(da.mean(dim="time"))
    print(f"  {key}: mean = {float(MEANS[key].mean()):.1f} mm/month  "
          f"max = {float(MEANS[key].max()):.1f} mm/month")
# =============================================================================
# MAP EXTENT
# =============================================================================

ref  = MEANS["CHIRPS"]
lons = ref["lon"].values
lats = ref["lat"].values
LON_MIN = float(lons.min()) - 0.8
LON_MAX = float(lons.max()) + 0.8
LAT_MIN = float(lats.min()) - 0.8
LAT_MAX = float(lats.max()) + 0.8

# =============================================================================
# PANEL DRAWING FUNCTION
# =============================================================================

proj = ccrs.PlateCarree()

def draw_panel(ax, data, label, panel_letter,
               show_left=True, show_bottom=True):
    """Draw one map panel. Returns the QuadMesh for the shared colourbar."""

    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)

    # Background
    ax.add_feature(cfeature.LAND,      facecolor="#F0EDE4", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#D6EAF8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,  edgecolor="#444444", zorder=4)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4,  edgecolor="#999999", linestyle="--", zorder=4)
    ax.add_feature(cfeature.RIVERS,    linewidth=0.35, edgecolor="#5DADE2", alpha=0.5, zorder=3)
    ax.add_feature(cfeature.LAKES,     facecolor="#D6EAF8", edgecolor="#5DADE2", linewidth=0.3, zorder=3)

    # Precipitation raster
    img = data.drop_vars("spatial_ref", errors="ignore").plot(
        ax=ax, transform=proj,
        cmap=CMAP, vmin=VMIN, vmax=VMAX,
        add_colorbar=False, zorder=2
    )

    # Basin boundary
    basin.boundary.plot(ax=ax, color="#C0392B", linewidth=1.1,
                        transform=proj, zorder=5)

    # Kainji Dam marker
    ax.plot(KAINJI_LON, KAINJI_LAT, marker="^",
            color="#E74C3C", markersize=6,
            transform=proj, zorder=6)

    # Gridlines
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

    # Label — colour-coded by dataset type, placed ABOVE the panel (as a title)
    if label == "CHIRPS":
        bg = "#1F3864"
    elif label == "Ensemble Mean":
        bg = "#006400"
    else:
        bg = "#2E4057"

    ax.set_title(label,
                 fontsize=14, fontweight="bold",
                 color="black", pad=6
                 )

    # Panel letter
    ax.text(0.03, 0.96, panel_letter,
            transform=ax.transAxes,
            fontsize=14, va="top", ha="left", color="black")

    return img

# =============================================================================
# BUILD FIGURE
# =============================================================================

print("\nBuilding figure ...")

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#F7F9FC")

gs = GridSpec(
    N_ROWS, N_COLS + 1,
    figure=fig,
    wspace=0.06, hspace=0.18,
    left=0.04, right=0.90,
    top=0.90,  bottom=0.06,
    width_ratios=[1, 1, 1, 1, 0.04]
)

# Data axes
axes = np.empty((N_ROWS, N_COLS), dtype=object)
for r in range(N_ROWS):
    for c in range(N_COLS):
        axes[r, c] = fig.add_subplot(gs[r, c], projection=proj)

# Adjust these four values: [left, bottom, width, height]
cax = fig.add_axes([0.89, 0.20, 0.01, 0.60])

# =============================================================================
# DRAW PANELS
# =============================================================================

last_img = None

for idx, (label, key) in enumerate(PANEL_ORDER):
    row = idx // N_COLS
    col = idx % N_COLS

    img = draw_panel(
        axes[row, col],
        MEANS[key],
        label,
        PANEL_LETTERS[idx],
        show_left   = (col == 0),
        show_bottom = (row == N_ROWS - 1),
    )
    last_img = img

# =============================================================================
# SHARED COLOURBAR
# =============================================================================

cb = fig.colorbar(last_img, cax=cax, orientation="vertical")
cb.set_label("Mean Monthly Precipitation (mm / month)",
             fontsize=14, color="black", labelpad=8)
cb.ax.tick_params(labelsize=12, colors="#333333")
ticks = np.linspace(VMIN, VMAX, 9)
cb.set_ticks(ticks)
cb.set_ticklabels([f"{int(t)}" for t in ticks])

# =============================================================================
# LEGEND
# =============================================================================

legend_elements = [
    mpatches.Patch(facecolor="none", edgecolor="#C0392B",
                   linewidth=1.2, label="Kainji Catchment Boundary"),
    plt.Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#E74C3C", markersize=8,
               label="Kainji Dam"),
    #mpatches.Patch(facecolor="#1F3864", label="CHIRPS (Observation)"),
    #mpatches.Patch(facecolor="#006400", label="Ensemble Mean"),
    #mpatches.Patch(facecolor="#2E4057", label="Individual CMIP6 Model"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.46, 0.02),
           edgecolor="#CCCCCC")

# =============================================================================
# MAIN TITLE
# =============================================================================
'''
fig.suptitle(
    f"Annual Mean Monthly Precipitation over the Niger Basin  |  "
    f"CHIRPS v2.0, Ensemble Mean and 10 CMIP6 Models  |  "
    f"{YEAR_START}–{YEAR_END} Climatology",
    fontsize=13, fontweight="bold", color="#1A252F", y=0.98
) '''

# =============================================================================
# SAVE
# =============================================================================

plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nFigure saved to: {OUTPUT_FILE}")
plt.show()
