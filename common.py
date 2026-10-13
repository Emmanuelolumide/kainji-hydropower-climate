import numpy as np, pandas as pd, xarray as xr, geopandas as gpd, rioxarray, calendar, warnings
from shapely.geometry import box
from config import *
warnings.filterwarnings("ignore")
mask_gdf = gpd.read_file(MASK_SHP).to_crs("EPSG:4326")

def region_geom(name):
    g = mask_gdf.geometry.union_all()
    if REGIONS[name] is None: return g
    x0, x1, y0, y1 = REGIONS[name]
    return g.intersection(box(x0, y0, x1, y1))

def std_lon(ds):
    lon = [c for c in ds.coords if c.lower().startswith("lon")][0]; lat = [c for c in ds.coords if c.lower().startswith("lat")][0]
    ds = ds.rename({lon: "lon", lat: "lat"})
    if float(ds.lon.max()) > 180: ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    return ds.sortby("lat")

def std_time(tv):
    if isinstance(tv[0], np.datetime64): return pd.DatetimeIndex(tv)
    out = []
    for t in tv:
        d = min(t.day, calendar.monthrange(t.year, t.month)[1]); out.append(pd.Timestamp(year=t.year, month=t.month, day=d))
    return pd.DatetimeIndex(out)

def region_weights(da2d, name):
    """cos(lat)-weighted 0/1 mask of region `name` on the (lat, lon) grid of da2d (all_touched, as in the original scripts)."""
    one = xr.ones_like(da2d.isel(time=0, drop=True)) if "time" in da2d.dims else xr.ones_like(da2d)
    one = one.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    g = region_geom(name)
    m = one.rio.clip([g], "EPSG:4326", drop=False, all_touched=True).fillna(0)
    w = m * np.cos(np.deg2rad(m.lat))
    return w

def wmean(da, w):
    """Weighted spatial mean over (lat, lon); da may have NaN (excluded)."""
    num = (da.fillna(0) * w).sum(("lat", "lon")); den = (da.notnull() * w).sum(("lat", "lon"))
    return num / den
