"""PREVIEW ONLY: Figs 4.17/4.18 with annual precipitation change (%) as the primary axis (empirical HMF-WA link dQ = a + b dP) and flow change as the secondary axis."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
R = "<PROJECT_ROOT>/revised"; OUT = R + "/plots/doc/"
ETA, RHO, G, QT, HN, HNM, P_INS, P_FIRM = 0.87, 1000., 9.81, 739.51, (33., 42.), 37.5, 760., 292.44
OBS = 2519.459 * 1000 / 8760; FS = 1.15
P = lambda Q, H: ETA * RHO * G * Q * H / 1e6
Pp = pd.read_csv(R + "/plots/module2_precip_pct_change_annual.csv").set_index("model"); Qq = pd.read_csv(R + "/plots/module3_table_flow_annual.csv").set_index("GCM")
pm = {"CanESM5": "CanESM5", "INM-CM4-8": "INM-CM4-8", "INM-CM5-0": "INM-CM5-0", "IPSL-CM6A-LR": "IPSL-CM6A-LR", "MPI-ESM1-2-HR": "MPI-ESM1-2-LR"}
KS = [("ssp245_near", "SSP2-4.5 Near", "#AED6F1"), ("ssp245_far", "SSP2-4.5 Far", "#2980B9"), ("ssp585_near", "SSP5-8.5 Near", "#F1948A"), ("ssp585_far", "SSP5-8.5 Far", "#C0392B")]
dP = {k: np.array([Pp.loc[pm[g], k] for g in pm]) for k, _, _ in KS}; dQ = {k: np.array([Qq.loc[g, k.replace("ssp245", "SSP245").replace("ssp585", "SSP585") + "_pct"] for g in pm]) for k, _, _ in KS}
x = np.concatenate([dP[k] for k, _, _ in KS]); y = np.concatenate([dQ[k] for k, _, _ in KS]); n = len(x)
r = stats.linregress(x, y); a, b = r.intercept, r.slope; res = y - (a + b * x); s = np.sqrt((res ** 2).sum() / (n - 2)); tq = stats.t.ppf(.975, n - 2)
def pi(xx): return tq * s * np.sqrt(1 + 1 / n + (xx - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
fit = lambda p: a + b * p; inv = lambda q: (q - a) / b
import json; json.dump(dict(a=a,b=b,r2=r.rvalue**2,r=r.rvalue,s=s,n=int(n),p=r.pvalue,xbar=float(x.mean()),sxx=float(((x-x.mean())**2).sum()),tq=float(tq),xmin=float(x.min()),xmax=float(x.max())),open(R+"/data/dQ_dP_link.json","w")); print(f"fit dQ = {a:.2f} + {b:.2f} dP ; r={r.rvalue:.2f} R2={r.rvalue**2:.2f} s={s:.1f} n={n}")
def grid(ax): ax.yaxis.grid(True, lw=.4, color="#DDD", ls="--"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
# ------------------------------------------------ A: scatter
fig, ax = plt.subplots(figsize=(8, 5.6)); xx = np.linspace(0, 70, 100)
ax.fill_between(xx, fit(xx) - pi(xx), fit(xx) + pi(xx), color="#BBB", alpha=.4, label="95% prediction band"); ax.plot(xx, fit(xx), "k-", lw=2, label=f"Fit: ΔQ = {a:.1f} + {b:.2f} ΔP  (R² = {r.rvalue**2:.2f})")
for k, lab, col in KS: ax.scatter(dP[k], dQ[k], s=60, color=col, edgecolor="k", zorder=5, label=lab)
ax.set_xlabel("Annual precipitation change, ΔP (%)  [NEX-GDDP-CMIP6, catchment mean]", fontsize=11 * FS); ax.set_ylabel("Annual flow change at Kainji, ΔQ (%)  [HMF-WA]", fontsize=11 * FS); ax.tick_params(labelsize=10 * FS); ax.legend(fontsize=9 * FS, loc="upper left"); grid(ax); ax.xaxis.grid(True, lw=.4, color="#DDD", ls="--")
plt.tight_layout(); fig.savefig(OUT + "figA8_dQ_vs_dP_fit.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
# ------------------------------------------------ 4.17 matrix
fig, ax = plt.subplots(figsize=(10, 7.2)); ps = np.linspace(-12, 100, 500); Hr = np.linspace(*HN, 300); PS, HH = np.meshgrid(ps, Hr); PG = P(QT * (1 + fit(PS) / 100), HH); vmax = float(np.ceil(PG.max() / 100) * 100)
im = ax.contourf(ps, Hr, PG, levels=np.linspace(0, vmax, 25), cmap="RdYlGn", vmin=0, vmax=vmax)
iso = [200, P_FIRM, 400, 500, 600, 760]; cs = ax.contour(ps, Hr, PG, levels=iso, colors="black", linewidths=[1.8 if v == P_FIRM else 1.4 if v == 760 else .8 for v in iso], linestyles=["-" if v in (P_FIRM, 760) else "--" for v in iso])
ax.clabel(cs, fmt=lambda v: f"{v:g} MW", fontsize=8 * FS, inline=True); ax.axvline(0, color="grey", lw=.7, ls=":")
ax.set_xlabel("Change in Annual Mean Precipitation, ΔP (%)", fontsize=11 * FS); ax.set_ylabel("Net Head, H (m)", fontsize=11 * FS); ax.set_xlim(ps[0], ps[-1]); ax.set_ylim(*HN); ax.tick_params(labelsize=10 * FS)
sec = ax.secondary_xaxis("top", functions=(fit, inv)); sec.set_xlabel("Equivalent flow change, ΔQ (%)  [fitted HMF-WA link, approximate]", fontsize=10 * FS); sec.tick_params(labelsize=9 * FS)
cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label(f"Projected Hydropower Potential (MW)  [baseline Qt = {QT:.2f} m³/s]", fontsize=11 * FS); cb.set_ticks(np.arange(0, vmax + 1, 100))
fig.savefig(OUT + "fig4_17_scenario_matrix.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
# ------------------------------------------------ 4.18 critical
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True); pr = np.linspace(-12, 125, 700)
for ai, Hv in enumerate(HN):
    ax = axes[ai]; line = P(QT * (1 + fit(pr) / 100), Hv); lo = P(QT * (1 + (fit(pr) - pi(pr)) / 100), Hv); hi = P(QT * (1 + (fit(pr) + pi(pr)) / 100), Hv)
    ax.fill_between(pr, lo, hi, color="#2980B9", alpha=.18, label="95% prediction band of the ΔP–ΔQ link"); ax.plot(pr, line, color="#2980B9", lw=2.2, label=f"Annual mean baseline Qt = {QT:.1f} m³/s")
    for lvl, col in ((P_FIRM, "#E74C3C"), (P_INS, "#2C3E50")):
        qd = 100 * (lvl / P(QT, Hv) - 1); pc = inv(qd); ax.plot([pc], [lvl], "o", color="#2980B9", ms=8, mec="k", zorder=5)
        gg = np.linspace(-80, 300, 8000); pl = gg[np.argmin(abs(P(QT * (1 + (fit(gg) + pi(gg)) / 100), Hv) - lvl))]; ph = gg[np.argmin(abs(P(QT * (1 + (fit(gg) - pi(gg)) / 100), Hv) - lvl))]
        xt, yt = ((pc + 14, 70) if lvl == P_FIRM else (82 if ai == 0 else 90, 905))
        ax.annotate(f"ΔP ≈ {pc:+.0f}%  ({pl:+.0f} to {ph:+.0f}%)\nΔQ = {qd:+.1f}%", xy=(pc, lvl), xytext=(xt, yt), fontsize=8.5 * FS, color="#2980B9", fontweight="bold", ha="left", bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5), arrowprops=dict(arrowstyle="-", color="#2980B9", lw=.8))
    ax.axhline(P_FIRM, color="#E74C3C", lw=1.6, ls="--", label=f"Firm-power threshold, 2014 minimum ({P_FIRM:.2f} MW)"); ax.axhline(P_INS, color="#2C3E50", lw=1, ls=":", label=f"Installed capacity ({P_INS:.0f} MW)")
    ax.axvline(0, color="grey", lw=.7, ls=":")
    for (k, lab, col), (y0, y1) in zip(KS, [(.95, 1.0), (.9, .95), (.85, .9), (.8, .85)]): ax.axvspan(dP[k].min(), dP[k].max(), ymin=y0, ymax=y1, facecolor=col, alpha=.55, label=lab + " GCM range" if ai == 0 else "_")
    ax.set_xlim(-12, 125); ax.set_ylim(0, 1050); ax.set_xlabel("Change in Annual Mean Precipitation, ΔP (%)", fontsize=11 * FS); ax.tick_params(labelsize=10 * FS); grid(ax)
    if ai == 0: ax.set_ylabel("Hydropower Potential (MW)", fontsize=11 * FS)
    ax.set_title(f"({'ab'[ai]})  H = {Hv:g} m ({'minimum' if ai == 0 else 'maximum'} net head)", fontsize=12 * FS, fontweight="bold", color="#1A252F")
    sx = ax.secondary_xaxis("top", functions=(fit, inv)); sx.set_xlabel("Equivalent ΔQ (%)", fontsize=9.5 * FS); sx.tick_params(labelsize=9 * FS)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5 * FS, bbox_to_anchor=(.5, -.2), framealpha=.95); plt.tight_layout()
fig.savefig(OUT + "fig4_18_critical_discharge_threshold.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig); print("done")
