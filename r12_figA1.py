"""Appendix A.1 (revised): seasonal precipitation, CHIRPS (row 1), CMIP6 ensemble mean (row 2) and bias (row 3; stippling = significant difference, Welch t-test p<0.05) on the Kainji mask, 1984-2014."""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mc
import cartopy.crs as ccrs, cartopy.feature as cfeature
from scipy import stats
from common import *
NOTITLE = "--notitle" in sys.argv
OUT = R + "/plots/" + ("doc/" if NOTITLE else "") + "figA1_seasonal_climatology_bias.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Y0, Y1 = 1984, 2014
ch = std_lon(xr.open_dataset(CHIRPS))["pr"].sel(time=slice(f"{Y0-1}-12-01", f"{Y1}-12-31")).load()
em = std_lon(xr.open_dataset(HIST_DIR + "/Ensmean.nc"))["pr"].sel(time=slice(f"{Y0-1}-12-01", f"{Y1}-12-31")).load()
em["time"] = pd.DatetimeIndex(std_time(em.time.values)); ch["time"] = pd.DatetimeIndex(std_time(ch.time.values))
SEAS = [("DJF", 12), ("MAM", 3), ("JJA", 6), ("SON", 9)]
def seasonal_series(da, start_month):
    q = da.resample(time="QS-DEC").mean()            # quarters starting Dec, Mar, Jun, Sep
    q = q.sel(time=q.time.dt.month == start_month)
    yrs = q.time.dt.year + (1 if start_month == 12 else 0)   # DJF labelled by the year of its January
    q = q.assign_coords(year=("time", yrs.values)).swap_dims({"time": "year"})
    return q.sel(year=slice(Y0, Y1))
g = mask_gdf.geometry.union_all()
b = mask_gdf.total_bounds; ext = [b[0] - .5, b[2] + .5, b[1] - .5, b[3] + .5]
def clip(x):
    return x.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.clip([g], "EPSG:4326", drop=True, all_touched=True)
fig, axs = plt.subplots(3, 4, figsize=(22, 11.5), subplot_kw={"projection": ccrs.PlateCarree()}); fig.patch.set_facecolor("#F8F9FF")
vmax = 400; norm = mc.Normalize(0, vmax); bnorm = mc.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
letters = "abcdefghijkl"
for j, (sn, sm) in enumerate(SEAS):
    sc = seasonal_series(ch, sm); sm_ = seasonal_series(em, sm)
    mo = sc.mean("year"); me = sm_.mean("year"); bias = me - mo
    t, p = stats.ttest_ind(sm_.values, sc.values, axis=0, equal_var=False)
    pv = xr.DataArray(p, coords=[sc.lat, sc.lon], dims=["lat", "lon"])
    for i, (da, nm) in enumerate(((mo, "CHIRPS"), (me, "Model"), (bias, "Bias"))):
        ax = axs[i, j]; ax.set_extent(ext, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#F0EDE4"); ax.add_feature(cfeature.OCEAN, facecolor="#D6EAF8"); ax.add_feature(cfeature.BORDERS, lw=.3, ls="--", edgecolor="#999"); ax.add_feature(cfeature.COASTLINE, lw=.4)
        c = clip(da)
        im = c.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="YlGnBu" if i < 2 else "RdBu", norm=norm if i < 2 else bnorm, add_colorbar=False) if i < 2 else c.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu_r", norm=bnorm, add_colorbar=False)
        if i == 2:
            sig = clip(pv) < 0.05; yy, xx = np.where(sig.values & np.isfinite(c.values))
            ax.scatter(c.lon.values[xx], c.lat.values[yy], s=16, c="k", marker=".", linewidths=0.6, transform=ccrs.PlateCarree(), zorder=6)
        mask_gdf.boundary.plot(ax=ax, color="#C0392B", lw=1.1, transform=ccrs.PlateCarree(), zorder=5); ax.plot(4.55, 9.867, "^", color="#E74C3C", ms=7, mec="k", transform=ccrs.PlateCarree(), zorder=7)
        ax.text(.04, .93, f"({letters[i*4+j]})", transform=ax.transAxes, fontsize=13, va="top")
        ax.set_title(sn if i == 0 else "", fontsize=14, fontweight="bold")
        gl = ax.gridlines(draw_labels=True, lw=.25, color="grey", alpha=.5, ls="--"); gl.top_labels = gl.right_labels = False
        gl.left_labels = (j == 0); gl.bottom_labels = (i == 2); gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}
        if j == 0: ax.text(-.22, .5, ["CHIRPS", "Model", "Bias\n(Model − CHIRPS)"][i], transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    if j == 3: imm = (axs[0, 3].collections[0], axs[2, 3].collections[0])
fig.subplots_adjust(left=.06, right=.90, hspace=.13, wspace=.05, top=.96, bottom=.09)
p0 = axs[0, 0].get_position(); p1 = axs[1, 0].get_position(); p2 = axs[2, 0].get_position()
cax1 = fig.add_axes([.915, p1.y0, .012, p0.y1 - p1.y0]); cax2 = fig.add_axes([.915, p2.y0, .012, p2.y1 - p2.y0])
cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="YlGnBu"), cax=cax1); cb1.set_label("Mean Monthly Precipitation (mm / month)", fontsize=12)
cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=bnorm, cmap="RdBu_r"), cax=cax2); cb2.set_label("Bias (mm / month)", fontsize=12)
fig.text(.5, .035, "▲ Kainji Dam      ▭ Kainji Catchment Boundary      • Significant difference (Welch's t-test, p < 0.05)", ha="center", fontsize=11)
if not NOTITLE: fig.suptitle("Seasonal precipitation, 1984-2014: CHIRPS, CMIP6 ensemble mean and bias (Kainji contributing catchment)", fontsize=15, fontweight="bold", y=.93)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()); print("saved", OUT)
