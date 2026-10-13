import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings
warnings.filterwarnings("ignore")
# =============================================================================
# USER SETTINGS — adjust all values in this block
# =============================================================================

# ── FILE PATHS ───────────────────────────────────────────────────────────────
# Path to your CHIRPS monthly NetCDF file
# For multiple files use a wildcard e.g. "chirps/chirps_monthly_*.nc"
CHIRPS_FILE = "<PROJECT_ROOT>/data/chirps/chirps_wa_mon.nc"

# Path to Niger Basin shapefile
SHAPEFILE = "<PROJECT_ROOT>/shp/kainji_upstream_49.shp"

# ── CLIMATOLOGY PERIOD ───────────────────────────────────────────────────────
YEAR_START = 1981    # <── adjust start year here
YEAR_END   = 2025    # revised: matches the 1981-2025 period stated in the text

# ── VARIABLE AND COORDINATE NAMES IN NetCDF ──────────────────────────────────
# CHIRPS standard names; change only if your file uses different names
PRECIP_VAR = "pr"
LON_VAR    = "longitude"
LAT_VAR    = "latitude"

# ── COLOUR SCALE ─────────────────────────────────────────────────────────────
# Shared colour scale applied to all four panels.
# Adjust VMAX upward if your region receives very high rainfall.
VMIN = 0      # mm/month (minimum — keep at 0)
VMAX = 400    # mm/month (maximum — adjust to suit your data range)
CMAP = "YlGnBu"   # alternatives: "Blues", "PuBu", "BuGn", "RdYlBu"

# ── KAINJI DAM COORDINATES ───────────────────────────────────────────────────
KAINJI_LON = 4.60   # degrees East
KAINJI_LAT = 9.87   # degrees North

# ── OUTPUT ───────────────────────────────────────────────────────────────────
OUTPUT_FILE = "<PROJECT_ROOT>/revised/plots/fig4_1_DJF_MAM_JJA_SON_climatology.png"
DPI         = 300
# =============================================================================
# SEASON DEFINITIONS
# =============================================================================
# Each entry: (label, long title, list of months)
SEASONS = [
    ("DJF", "December–January–February", [12, 1, 2]),
    ("MAM", "March–April–May",            [3, 4, 5]),
    ("JJA", "June–July–August",            [6, 7, 8]),
    ("SON", "September–October–November",  [9, 10, 11]),
]

# =============================================================================
# LOAD AND SUBSET DATA
# =============================================================================

print(f"Loading CHIRPS data: {CHIRPS_FILE}")
ds = xr.open_mfdataset(CHIRPS_FILE, combine="by_coords")

# Standardise coordinate names to lon / lat
rename_dict = {}
if LON_VAR in ds.coords and LON_VAR != "lon":
    rename_dict[LON_VAR] = "lon"
if LAT_VAR in ds.coords and LAT_VAR != "lat":
    rename_dict[LAT_VAR] = "lat"
if rename_dict:
    ds = ds.rename(rename_dict)

precip = ds[PRECIP_VAR]

print(f"Subsetting to {YEAR_START}–{YEAR_END} ...")
precip = precip.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))

# =============================================================================
# LOAD SHAPEFILE
# =============================================================================

print(f"Loading shapefile: {SHAPEFILE}")
niger_basin = gpd.read_file(SHAPEFILE)

if niger_basin.crs is None:
    niger_basin = niger_basin.set_crs("EPSG:4326")
else:
    niger_basin = niger_basin.to_crs("EPSG:4326")

# =============================================================================
# COMPUTE SEASONAL MEANS AND CLIP
# =============================================================================

season_data = {}

for abbr, title, months in SEASONS:
    print(f"  Computing {abbr} ({months}) ...")

    seas_mean = (
        precip.sel(time=precip["time.month"].isin(months))
        .mean(dim="time")
    )

    # Set spatial info and clip
    seas_mean = seas_mean.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    seas_mean = seas_mean.rio.write_crs("EPSG:4326")
    seas_clipped = seas_mean.rio.clip(
        niger_basin.geometry, niger_basin.crs, drop=True
    )

    season_data[abbr] = {"data": seas_clipped, "title": title}

print("All seasonal means computed and clipped.")

# =============================================================================
# MAP EXTENT FROM CLIPPED DATA
# =============================================================================

sample = list(season_data.values())[0]["data"]
lons = sample["lon"].values
lats = sample["lat"].values
lon_min = float(lons.min()) - 0.8
lon_max = float(lons.max()) + 0.8
lat_min = float(lats.min()) - 0.8
lat_max = float(lats.max()) + 0.8
# =============================================================================
# FIGURE — 1 x 4 LAYOUT
# =============================================================================

print("Building figure ...")

proj      = ccrs.PlateCarree()
panel_labels = ["(a)", "(b)", "(c)", "(d)"]

fig, axes = plt.subplots(
    1, 4,
    figsize=(22, 7),
    subplot_kw={"projection": proj},
    gridspec_kw={"wspace": 0.06}
)

fig.patch.set_facecolor("#F7F9FC")

# =============================================================================
# PANEL PLOTTING FUNCTION
# =============================================================================

def plot_panel(ax, data, season_abbr, season_title, panel_label,
               show_ylabel=True):
    """Draw one seasonal panel."""

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Base layers
    ax.add_feature(cfeature.LAND,
                   facecolor="#F0EDE4", zorder=0)
    ax.add_feature(cfeature.OCEAN,
                   facecolor="#D6EAF8", zorder=0)
    ax.add_feature(cfeature.COASTLINE,
                   linewidth=0.5, edgecolor="#444444", zorder=4)
    ax.add_feature(cfeature.BORDERS,
                   linewidth=0.45, edgecolor="#888888",
                   linestyle="--", zorder=4)
    ax.add_feature(cfeature.RIVERS,
                   linewidth=0.4, edgecolor="#5DADE2",
                   alpha=0.55, zorder=3)
    ax.add_feature(cfeature.LAKES,
                   facecolor="#D6EAF8", edgecolor="#5DADE2",
                   linewidth=0.35, zorder=3)

    # Precipitation raster
    img = data.plot(
        ax=ax,
        transform=proj,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        add_colorbar=False,
        zorder=2
    )

    # Niger Basin boundary
    niger_basin.boundary.plot(
        ax=ax, color="#C0392B", linewidth=1.4,
        transform=proj, zorder=5
    )

    # Kainji Dam marker
    ax.plot(KAINJI_LON, KAINJI_LAT,
            marker="^", color="#E74C3C", markersize=8,
            transform=proj, zorder=6)
    ax.text(KAINJI_LON + 0.5, KAINJI_LAT - 0.5,
            "", transform=proj,              #write name on kainji latlon
            fontsize=7, color="#C0392B",
            fontweight="bold", zorder=6)

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.35, color="grey",
        alpha=0.5, linestyle="--"
    )
    gl.top_labels   = False
    gl.right_labels = False

    # Only left-most panel gets y-axis labels
    if not show_ylabel:
        gl.left_labels = False

    gl.xlabel_style = {"size": 8, "color": "#444444"}
    gl.ylabel_style = {"size": 8, "color": "#444444"}
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER

    # Season abbreviation badge
    ax.text(0.03, 0.97, season_abbr,
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            va="top", ha="left", color="#1A252F",
            bbox=dict(facecolor="white", alpha=0.75,
                      edgecolor="none", pad=3))

    # Panel letter
    ax.text(0.03, 0.88, panel_label,
            transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            color="#555555")

    # Title beneath season abbreviation
    ax.set_title(" ", fontsize=9.5,
                 color="#1A252F", pad=6)

    return img

# =============================================================================
# DRAW ALL PANELS
# =============================================================================

last_img = None
for idx, (abbr, _, months) in enumerate(SEASONS):
    info = season_data[abbr]
    show_y = (idx == 0)  # only leftmost panel gets lat labels
    img = plot_panel(
        axes[idx],
        info["data"],
        abbr,
        info["title"],
        panel_labels[idx],
        show_ylabel=show_y
    )
    last_img = img

# =============================================================================
# SHARED COLOURBAR (bottom, spanning all 4 panels)
# =============================================================================

cbar_ax = fig.add_axes([0.266, 0.2, 0.5, 0.025]) #Adjust colorbar position
cb = fig.colorbar(last_img, cax=cbar_ax, orientation="horizontal")
cb.set_label(
    "Mean Monthly Precipitation (mm/month)",
    fontsize=11, color="#333333", labelpad=6
)
cb.ax.tick_params(labelsize=9, colors="#333333")

# Custom tick positions across the colour scale
tick_positions = np.linspace(VMIN, VMAX, 8)
cb.set_ticks(tick_positions)
cb.set_ticklabels([f"{int(t)}" for t in tick_positions])

# =============================================================================
# SHARED LEGEND
# =============================================================================

legend_elements = [
    mpatches.Patch(
        facecolor="none", edgecolor="#C0392B",
        linewidth=1.4, label="Kainji Catchment Boundary"
    ),
    plt.Line2D(
        [0], [0], marker="^", color="w",
        markerfacecolor="#E74C3C", markersize=9,
        label="Kainji Dam"
    ),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=2,
    fontsize=10,
    framealpha=0.9,
    bbox_to_anchor=(0.5155, 0.08), #Adjust legend position
    edgecolor="#CCCCCC"
)

# =============================================================================
# MAIN TITLE
# =============================================================================
'''
fig.suptitle(
    f"Seasonal Precipitation Climatology over the Niger Basin\n"
    f"CHIRPS v2.0  |  {YEAR_START}–{YEAR_END}",
    fontsize=15, fontweight="bold",
    color="#1A252F", y=1.04
)
'''
# =============================================================================
# SAVE
# =============================================================================
plt.savefig(
    OUTPUT_FILE,
    dpi=DPI,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)
print(f"\nFigure saved to: {OUTPUT_FILE}")
plt.show()
