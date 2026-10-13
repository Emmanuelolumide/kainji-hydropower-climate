"""PREVIEW: mean seasonal cycle (Julian day) of HMF-WA river flow at Kainji (top) and NEX-GDDP-CMIP6 catchment precipitation (bottom):
historical (1984-2014) versus mid-century (2030-2060) and late-century (2070-2100), SSP2-4.5 and SSP5-8.5; ensemble mean of the five HMF-WA GCMs."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.interpolate import PchipInterpolator
R = "<PROJECT_ROOT>/revised"; OUT = "./output/"
SET = sys.argv[1] if len(sys.argv) > 1 else "five"; MONTHS = "--months" in sys.argv; DOC = "--doc" in sys.argv or MONTHS; KF = 1.0            # "five" = same GCMs as the flow; "ten" = full NEX-GDDP ensemble
GF = ["CanESM5", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "MPI-ESM1-2-HR"]
GP = ["CanESM5", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "MPI-ESM1-2-LR"] if SET == "five" else None
fl = pd.read_csv(R + "/data/hmfwa_cells_monthly.csv"); fl = fl[fl.cell == "Kainji"]
pr = pd.read_csv(R + "/data/nex_regional_monthly.csv"); pr = pr[pr.region == "Whole"]
if GP is None: GP = sorted(pr.model.unique())
PER = {"hist": (1984, 2014), "near": (2030, 2060), "far": (2070, 2100)}
def clim(df, models, mcol, scen, per, col):
    y0, y1 = PER[per]; d = df[(df.scenario == scen) & df.year.between(y0, y1) & df[mcol].isin(models)]
    return d.groupby([mcol, "month"])[col].mean().unstack("month").reindex(models).values       # models x 12
Fh, Ph = clim(fl, GF, "gcm", "HIST", "hist", "flow_m3s"), clim(pr, GP, "model", "HIST", "hist", "pr_mm")
MID = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]); DAYS = np.linspace(0, 365, 366)
def smooth(v):
    x = np.concatenate([MID - 365, MID, MID + 365]); y = np.tile(v, 3); return np.clip(PchipInterpolator(x, y)(DAYS), 0, None)
PAN = [("a", "SSP245", "near", "SSP2-4.5, mid-century (2030-2060)"), ("b", "SSP585", "near", "SSP5-8.5, mid-century (2030-2060)"),
       ("c", "SSP245", "far", "SSP2-4.5, late-century (2070-2100)"), ("d", "SSP585", "far", "SSP5-8.5, late-century (2070-2100)")]
data = {}
for lab, s, p, _ in PAN: data[lab] = (clim(fl, GF, "gcm", s, p, "flow_m3s"), clim(pr, GP, "model", s, p, "pr_mm"))
fmax = max(1.08 * smooth(F.mean(0)).max() for F, _ in data.values()); pmax = 1.1 * max(smooth(P.mean(0)).max() for _, P in data.values())
FS = 1.0; B, K = "#1F4FFF", "black"
fig = plt.figure(figsize=(10.8, 10.0) if DOC else (13.5, 12.5)); outer = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.28)
for n, (lab, s, p, ttl) in enumerate(PAN):
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[n // 2, n % 2], height_ratios=[1.35, 1], hspace=0.0)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax1); F, Pm = data[lab]
    # ---- river flow (above)
    ax1.plot(DAYS, smooth(Fh.mean(0)), color=K, lw=1.6, label="Flow, historical (1984-2014)")
    ax1.plot(DAYS, smooth(F.mean(0)), color=B, lw=1.8, label=f"Flow, {'mid-century' if p == 'near' else 'late-century'} (ensemble mean)")
    ax1.set_ylim(0, fmax); ax1.set_ylabel("River flow (m$^3$/s)", fontsize=11 * KF); ax1.tick_params(labelbottom=False)
    ax1.set_title(f"{lab}   {ttl}", fontsize=12 * KF, fontweight="bold", loc="left")
    ax1.legend(loc="upper left", fontsize=8.2 * KF, framealpha=0.95)
    # ---- precipitation (below)
    ax2.fill_between(DAYS, 0, smooth(Pm.mean(0)), color=B, alpha=0.85, lw=0, label=f"Precipitation, {'mid-century' if p == 'near' else 'late-century'}")
    ax2.plot(MID, Ph.mean(0), color=K, ls=":", lw=1.2, marker="^", mfc="white", mec=K, ms=6, label="Precipitation, historical")
    ax2.set_ylim(0, pmax); ax2.set_ylabel("Precipitation (mm/month)", fontsize=11 * KF); ax2.set_xlim(0, 365)
    if MONTHS:
        ax2.set_xticks(MID); ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8.6 * KF)
        ax2.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], minor=True); ax2.tick_params(axis="x", which="minor", length=4); ax2.tick_params(axis="x", which="major", length=0)
        ax2.set_xlabel("Month", fontsize=11 * KF)
    else: ax2.set_xlabel("Julian day", fontsize=11 * KF)
    ax2.legend(loc="upper left", fontsize=8.2 * KF, framealpha=0.95); ax2.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
    for a in (ax1, ax2): a.tick_params(labelsize=9.5 * KF)
fig.savefig(OUT + ("fig4_10_pchip.png" if MONTHS else "fig4_10_seasonal_flow_precip.png" if DOC else f"seasonal_flow_precip_{SET}.png"), dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
# ---- summary numbers
mon = lambda v: int(np.argmax(v)) + 1
print("historical flow peak month", mon(Fh.mean(0)), "| annual mean flow %.0f" % Fh.mean(), "| precip annual %.0f mm" % (12 * Ph.mean()))
for lab, s, p, ttl in PAN:
    F, Pm = data[lab]; print(lab, ttl, "| flow peak month", mon(F.mean(0)), "peak %.0f vs hist %.0f m3/s" % (F.mean(0).max(), Fh.mean(0).max()), "| annual flow %+.0f%%" % (100 * (F.mean() / Fh.mean() - 1)),
          "| precip annual %+.1f%%, JAS %+.1f%%, DJF %+.1f%%" % (100 * (Pm.sum(1).mean() / Ph.sum(1).mean() - 1), 100 * (Pm[:, 6:9].sum() / Ph[:, 6:9].sum() - 1), 100 * (Pm[:, [11, 0, 1]].sum() / Ph[:, [11, 0, 1]].sum() - 1)))
