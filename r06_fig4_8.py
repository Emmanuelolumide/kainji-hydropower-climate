"""Annual model precipitation percentage-change time series for four regions.

Historical, SSP2-4.5 and SSP5-8.5 anomalies all use the model historical
1984-2014 mean.  Where multiple models/members are available, the line is the
ensemble mean and the shaded envelope is the 10th-90th percentile spread.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # registers the .rio accessor
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# USER SETTINGS -- replace the three model paths and variable name if needed.
# A glob (for example "/path/to/historical/*.nc") is also accepted.
# Files may contain an ensemble/model/member dimension; it is averaged for the
# line and shown as the 10th-90th percentile envelope.
# -----------------------------------------------------------------------------
HISTORICAL_DIR = "<PROJECT_ROOT>/data/models/hist"
SSP245_DIR = "<PROJECT_ROOT>/data/models/proj/ssp245"
SSP585_DIR = "<PROJECT_ROOT>/data/models/proj/ssp585"

MODEL_VAR = "pr"
SHAPEFILE = "<PROJECT_ROOT>/shp/kainji_upstream_49.shp"
OUTPUT_FILE = "./output/fig4_8.png"

BASELINE_START, BASELINE_END = 1984, 2014
HIST_START, HIST_END = 1981, 2014
FUTURE_START, FUTURE_END = 2026, 2100
DPI = 300

SUB_REGIONS = {
    "Upper Niger\n(Guinea Highlands)": {"bbox": (-12.0, -2.0, 8.0, 13.0)},
    "Middle Niger\n(Sahel Zone)": {"bbox": (-4.0, 4.0, 13.0, 18.0)},
    "Lower Niger\n(Near Kainji)": {"bbox": (2.0, 8.0, 9.87, 13.0)},
}
WHOLE_BASIN_LABEL = "Kainji Contributing\nCatchment"

COLORS = {"historical": "#111111", "ssp245": "#0057E7", "ssp585": "#E60000"}
LABELS = {"historical": "Historical", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}

# The full min--max range is deliberately used so the inter-model envelope is
# easy to see.  Set to "p10_p90" if you prefer to suppress ensemble outliers.
SPREAD_STYLE = "minmax"


def timedelta_days(delta):
    """Return a day count for either NumPy or CF-time timestamp differences."""
    if hasattr(delta, "days"):
        return delta.days + delta.seconds / 86400
    return float(delta / np.timedelta64(1, "s")) / 86400


def standardise_coordinates(da):
    """Standardise longitude and latitude coordinate names and ordering."""
    rename = {}
    for old, new in (("longitude", "lon"), ("latitude", "lat")):
        if old in da.coords and new not in da.coords:
            rename[old] = new
    if rename:
        da = da.rename(rename)
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    return da.sortby("lat").sortby("lon")


def open_model_ensemble(directory, variable):
    """Open each individual-model file, deliberately excluding Ensmean files.

    The model files use slightly different time calendars.  They are therefore
    kept separate until after annual totals are calculated, when all members
    can safely be aligned by integer year.
    """
    files = sorted(path for path in Path(directory).glob("*.nc")
                   if "ensmean" not in path.name.lower())
    if not files:
        raise FileNotFoundError(f"No individual-model NetCDF files in {directory}")
    members = []
    for path in files:
        # Daily projected files are large.  One-year chunks keep regional
        # aggregation memory-safe while still allowing Dask to stream results.
        da = standardise_coordinates(xr.open_dataset(path, chunks={"time": 366})[variable])
        members.append((path.stem, da))
    print(f"  {Path(directory).name}: using {len(members)} individual models")
    return members


def precipitation_amount_mm(da):
    """Return precipitation amount per input time step in mm.

    Daily CMIP fluxes (kg m-2 s-1) are converted by their time step.  The
    supplied historical files carry ``cell_methods: time: sum`` and already
    contain monthly accumulated mm despite their inherited flux unit label;
    those values must *not* be multiplied by seconds again.
    Change this function only if a source uses another precipitation convention.
    """
    units = str(da.attrs.get("units", "")).lower().replace("**", "")
    is_rate = ("s-1" in units or "/s" in units or "sec-1" in units)
    is_time_sum = "time: sum" in str(da.attrs.get("cell_methods", "")).lower()
    if is_rate and not is_time_sum:
        step = da.time.values[1] - da.time.values[0]
        step_days = timedelta_days(step)
        seconds = 86400 if step_days <= 1.5 else da.time.dt.days_in_month * 86400
        return da * seconds
    return da


def annual_member_series(da, basin_gdf, bbox_, label):
    """Clip one model, spatially average it, then calculate annual totals."""
    lon_min, lon_max, lat_min, lat_max = bbox_
    region_geom = basin_gdf.geometry.union_all().intersection(
        box(lon_min, lat_min, lon_max, lat_max)
    )
    if region_geom.is_empty:
        raise ValueError(f"{label}: bounding box does not intersect the basin.")

    subset = da.sel(lon=slice(lon_min - .1, lon_max + .1),
                    lat=slice(lat_min - .1, lat_max + .1))
    subset = (subset.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                    .rio.write_crs("EPSG:4326")
                    .rio.clip([region_geom], "EPSG:4326", drop=True, all_touched=True))
    spatial = precipitation_amount_mm(subset).mean(("lat", "lon"), skipna=True)
    step = da.time.values[1] - da.time.values[0]
    step_days = timedelta_days(step)
    sample_count = spatial.groupby("time.year").count("time")
    annual = spatial.groupby("time.year").sum("time", skipna=True)
    # Do not allow an incomplete calendar year to influence an anomaly or trend.
    # Compute the indexer *before* using it: Xarray cannot drop with a lazy
    # Dask boolean array because its resulting shape is unknown.
    # Historical data are monthly (12 values/year); the projections supplied
    # here are daily (365/366 values/year).  A 360-day calendar is also valid.
    required_count = 12 if step_days > 1.5 else 360
    complete_year = (sample_count >= required_count).compute()
    annual = annual.sel(year=annual.year.values[complete_year.values]).compute()
    return annual


def annual_region_series(members, basin_gdf, bbox_, label):
    """Return annual ensemble mean and spread after aligning members by year."""
    model_names, annual_members = zip(*[
        (model_name, annual_member_series(da, basin_gdf, bbox_, label))
        for model_name, da in members
    ])
    annual = xr.concat(annual_members,
                       dim=xr.IndexVariable("model", list(model_names)),
                       join="outer")
    ensemble_dims = [d for d in annual.dims if d != "year"]
    mean = annual.mean(ensemble_dims, skipna=True) if ensemble_dims else annual
    if ensemble_dims:
        if SPREAD_STYLE == "minmax":
            low = annual.min(ensemble_dims, skipna=True)
            high = annual.max(ensemble_dims, skipna=True)
        elif SPREAD_STYLE == "p10_p90":
            low = annual.quantile(.10, dim=ensemble_dims, skipna=True)
            high = annual.quantile(.90, dim=ensemble_dims, skipna=True)
        else:
            raise ValueError("SPREAD_STYLE must be 'minmax' or 'p10_p90'.")
    else:
        low = high = mean
    return {"years": annual.year.values.astype(int), "mean": mean.values,
            "low": low.values, "high": high.values}


def select_period(series, start, end):
    keep = (series["years"] >= start) & (series["years"] <= end)
    return {key: np.asarray(value)[keep] for key, value in series.items()}


def anomaly(series, baseline):
    """Percentage change relative to the historical-model baseline."""
    return {key: (100 * (value - baseline) / baseline if key != "years" else value)
            for key, value in series.items()}


def trend(years, values):
    """Least-squares trend, expressed in percentage points per decade."""
    valid = np.isfinite(values)
    slope, intercept = np.polyfit(years[valid], values[valid], 1)
    return slope * 10, intercept + slope * years


print("Loading precipitation datasets …")
import pandas as pd
_NEX = pd.read_csv("<PROJECT_ROOT>/revised/data/nex_regional_monthly.csv")
datasets = {"historical": "HIST", "ssp245": "SSP245", "ssp585": "SSP585"}
def csv_series(sc, code):
    x = _NEX[(_NEX.scenario == sc) & (_NEX.region == code)]
    ann = x.groupby(["model", "year"]).pr_mm.agg(["sum", "count"]).reset_index(); ann = ann[ann["count"] == 12]
    P = ann.pivot(index="year", columns="model", values="sum")
    return {"years": P.index.values.astype(int), "mean": P.mean(axis=1).values, "low": P.min(axis=1).values, "high": P.max(axis=1).values}

basin = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326")
minx, miny, maxx, maxy = basin.total_bounds
regions = {WHOLE_BASIN_LABEL: {"bbox": (minx - .5, maxx + .5, miny - .5, maxy + .5)}}
regions.update(SUB_REGIONS)

fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharey=False)
axes = axes.ravel()
panel_letters = ("(a)", "(b)", "(c)", "(d)")

for panel, (region_name, cfg) in enumerate(regions.items()):
    print(f"Extracting {region_name.replace(chr(10), ' ')} …")
    _code = dict(zip(regions.keys(), ["Whole", "Upper", "Middle", "Lower"]))[region_name]
    raw = {key: csv_series(sc, _code) for key, sc in datasets.items()}

    # Every model line is referenced to the historical-model 1984-2014 mean.
    model_base = np.nanmean(select_period(raw["historical"], BASELINE_START, BASELINE_END)["mean"])
    data = {
        "historical": anomaly(select_period(raw["historical"], HIST_START, HIST_END), model_base),
        "ssp245": anomaly(select_period(raw["ssp245"], FUTURE_START, FUTURE_END), model_base),
        "ssp585": anomaly(select_period(raw["ssp585"], FUTURE_START, FUTURE_END), model_base),
    }

    ax = axes[panel]
    # Background shading separates the near future (NF) from far future (FF).
    ax.axvspan(2030, 2060, color="0.70", alpha=.20, zorder=0)
    ax.axvspan(2070, 2100, color="0.45", alpha=.16, zorder=0)
    ax.text(2047.5, .98, "Mid-century\n2030–2060", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.25")
    ax.text(2083, .98, "Late-century\n2070–2100", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.25")

    handles = []
    for key in ("historical", "ssp245", "ssp585"):
        d = data[key]
        if len(d["years"]) < 2:
            raise ValueError(f"{region_name}: fewer than two valid years for {LABELS[key]}.")
        slope_decade, fitted = trend(d["years"], d["mean"])
        # This envelope visualises the model/member spread (min--max by default).
        if not np.allclose(d["low"], d["high"], equal_nan=True):
            ax.fill_between(d["years"], d["low"], d["high"], color=COLORS[key], alpha=.20, lw=0)
        ax.plot(d["years"], d["mean"], color=COLORS[key], lw=1.5, zorder=3)
        ax.plot(d["years"], fitted, color=COLORS[key], lw=1.8, ls="--", zorder=4)
        handles.append(Line2D([], [], color=COLORS[key], lw=1.8, ls="--",
                              label=f"{LABELS[key]}: {slope_decade:+.2f} %/decade"))

    ax.axhline(0, color="0.35", lw=.8, zorder=1)
    ax.axvspan(1984, 2014, color="0.7", alpha=.10, zorder=0)
    ax.set_title(f"{panel_letters[panel]} {region_name}", loc="center", fontweight="bold", fontsize=13)
    ax.set_xlim(1980.5, 2100.5)
    ax.set_xticks(np.arange(1981, 2101, 10))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="0.85", lw=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitation change (%)")
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=.88)
    print("  " + "; ".join(h.get_label() for h in handles))

fig.suptitle("Annual model precipitation percentage change relative to the 1984–2014 historical baseline", fontweight="bold", y=.995)
fig.text(.5, .955, "Solid line: ensemble mean; dashed line: linear trend; coloured envelope: model spread", ha="center", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_FILE}")
