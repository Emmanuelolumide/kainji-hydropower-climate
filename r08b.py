"""Module 3 (revised) figures, original styling of plot_module3_sensitivity.py / module_discharge_timeseries.py but driven by HMF-WA dQ.
usage: python r08b_module3_figs.py [user|net]   ->  plots/fig4_14..fig4_18 (user head)  or  plots/alt_netHead/*"""
import sys, pickle, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from config import *

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
NOTITLE = "--notitle" in sys.argv
MODE = ARGS[0] if ARGS else "user"
D = pickle.load(open(R + "/data/module3.pkl", "rb"))
F, E, ann, QT, HEADS, GCMS = D["F"], D["E"], D["ann"], D["QT"], D["HEADS"], D["GCMS"]
H_MIN, H_MAX = HEADS[MODE]; H_MEAN = 0.5 * (H_MIN + H_MAX)
ETA, RHO, G, P_INS, P_FIRM = 0.87, 1000.0, 9.81, 760.0, 380.0
QT_MIN, QT_MAX = 334.70, 1271.63
OUT = "./output/"
os.makedirs(OUT, exist_ok=True)
SUF = "" if MODE == "user" else "_netHead"
DPI = 300
def suptitle(fig, *a, **k):
    if not NOTITLE: fig.suptitle(*a, **k)
def P(Q, H): return ETA * RHO * G * Q * H / 1e6
HEADLAB = f"H = {H_MIN:g}-{H_MAX:g} m" + ("" if MODE == "user" else " (net head, Mohammed et al. 2019)")
COMBOS = [("SSP245", "near", "SSP2-4.5 Mid-century", "#AED6F1"), ("SSP245", "far", "SSP2-4.5 Late-century", "#2980B9"),
          ("SSP585", "near", "SSP5-8.5 Mid-century", "#F1948A"), ("SSP585", "far", "SSP5-8.5 Late-century", "#C0392B")]
SPC = [("SSP245", "near", "SSP2-4.5\nMid-century"), ("SSP245", "far", "SSP2-4.5\nLate-century"), ("SSP585", "near", "SSP5-8.5\nMid-century"), ("SSP585", "far", "SSP5-8.5\nLate-century")]
DNAME = {"annual": "Annual mean", "dry": "Dry season (Nov–Mar)", "wet": "Wet season (Apr–Oct)"}
def pct(w, g, s, p):
    x = F[(F.driver == w) & (F.gcm == g) & (F.scenario == s) & (F.period == p)]
    return float(x.pct.iloc[0])
def pct_ens(w, s, p): return float(F[(F.driver == w) & (F.scenario == s) & (F.period == p)].pct.mean())
def Qproj(w, g, s, p):
    pc = pct_ens(w, s, p) if g == "Ensemble" else pct(w, g, s, p)
    return max(0.0, QT[w] * (1 + pc / 100))
def grid(ax):
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD", linestyle="--", zorder=0); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

# ===================== Fig 4.15  projected turbine discharge by GCM and scenario (3 drivers) =====================
fig, axes = plt.subplots(1, 3, figsize=(23, 7), sharey=True); fig.patch.set_facecolor("white")
labels = GCMS + ["Ensemble\nMean"]; x = np.arange(len(labels)); bw = 0.18; offs = [-1.5, -0.5, 0.5, 1.5]
for ai, w in enumerate(["annual", "dry", "wet"]):
    ax = axes[ai]
    for ci, (s, p, lab, col) in enumerate(COMBOS):
        v = [Qproj(w, g, s, p) for g in GCMS] + [Qproj(w, "Ensemble", s, p)]
        ax.bar(x[:-1] + offs[ci] * bw, v[:-1], width=bw, color=col, alpha=0.85, label=lab if ai == 0 else "_nolegend_", edgecolor="white", linewidth=0.4, zorder=3)
        ax.bar(x[-1] + offs[ci] * bw, v[-1], width=bw, color=col, alpha=0.85, label="_nolegend_", edgecolor="black", linewidth=1.3, zorder=3)
    ax.axvline(len(GCMS) - 0.5, color="#888888", linewidth=1.0, zorder=2)
    ax.axhline(QT[w], color="black", linewidth=2.0, label=f"Baseline mean, {DNAME[w].lower()} ({QT[w]:.0f} m³/s)" if ai == 0 else "_")
    if ai == 0:
        ax.axhline(QT_MIN, color="#7F8C8D", linewidth=1.2, linestyle="--", label=f"Baseline min ({QT_MIN:.0f} m³/s)")
        ax.axhline(QT_MAX, color="#BDC3C7", linewidth=1.2, linestyle="--", label=f"Baseline max ({QT_MAX:.0f} m³/s)")
    else:
        ax.axhline(QT[w], color="black", linewidth=2.0, label="_")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    if ai == 0: ax.set_ylabel("Projected Turbine Discharge Q (m³/s)", fontsize=11)
    ax.set_title(f"{DNAME[w]} ΔQ driver  (baseline Qt = {QT[w]:.1f} m³/s)", fontsize=12, fontweight="bold", color="#1A252F")
    ax.tick_params(labelsize=10); grid(ax)
    ax.text(0.01, 0.98, f"({'abc'[ai]})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, -0.04), framealpha=0.95)
suptitle(fig, "Projected Turbine Discharge at Kainji Dam by GCM and Scenario (HMF-WA River-Flow Change Factors)\n"
             "Qproj = Qt × (1 + ΔQ)  |  ΔQ from HMF-WA monthly mean flow at the Kainji cell, 5 GCMs  |  Black-outlined bars = 5-GCM ensemble mean",
             fontsize=13, fontweight="bold", color="#1A252F", y=1.03)
plt.tight_layout(); plt.savefig(OUT + f"fig4_14_projected_discharge_by_gcm{SUF}.png", dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close()

# ===================== Fig 4.14  discharge trajectory 2015-2100 (annual-mean driver) =====================
fig, ax = plt.subplots(figsize=(12, 7)); fig.patch.set_facecolor("white")
COL = {"SSP245": "#0057E7", "SSP585": "#E60000"}; LAB = {"SSP245": "SSP2-4.5", "SSP585": "SSP5-8.5"}
handles = []
tr = pd.read_csv(R + "/plots/module3_trends_2015_2100.csv")
for s in ("SSP245", "SSP585"):
    xx = ann[ann.scenario == s].pivot(index="year", columns="gcm", values="dq")
    em = xx.mean(axis=1); lo, hi = xx.min(axis=1), xx.max(axis=1)
    Qc, Ql, Qh = QT["annual"] * (1 + em / 100), QT["annual"] * (1 + lo / 100), QT["annual"] * (1 + hi / 100)
    yrs = xx.index.values
    ax.fill_between(yrs, Ql, Qh, color=COL[s], alpha=.18, lw=0, zorder=2)
    ax.plot(yrs, Qc, color=COL[s], lw=1.8, zorder=3)
    sen, icpt = stats.theilslopes(Qc.values, yrs)[:2]
    ax.plot(yrs, icpt + sen * yrs, color=COL[s], lw=0.9, ls="--", zorder=3)
    p_mk = stats.kendalltau(yrs, Qc.values)[1]
    handles.append(Line2D([], [], color=COL[s], lw=1.8, label=f"{LAB[s]} ensemble mean ({sen*10:+.1f} m³/s per decade, Mann-Kendall p {'<0.001' if p_mk < 0.001 else '= %.2f' % p_mk})"))
handles.append(Patch(facecolor="0.5", alpha=.25, label="5-GCM range (min–max) of annual-mean ΔQ applied to Qt"))
ax.axhline(QT["annual"], color="black", lw=1.2, ls=":", zorder=1)
ax.text(2016, QT["annual"] - 105, f"Baseline mean Qt ({QT['annual']:.0f} m³/s)", fontsize=8.5, color="0.25", ha="left")
ax.axvspan(2030, 2060, color="0.45", alpha=.07, zorder=0); ax.axvspan(2070, 2100, color="0.45", alpha=.07, zorder=0)
ax.text(2045, .04, "Mid-century\n2030-2060", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="0.25")
ax.text(2085, .04, "Late-century\n2070-2100", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="0.25")
if not NOTITLE: ax.set_title("Projected Turbine Discharge at Kainji Dam, 2015-2100\nAnnual-Mean HMF-WA River-Flow Change (Kainji cell), Qproj = Qt × (1 + ΔQ), Ensemble Mean of 5 GCMs",
             loc="left", fontweight="bold", fontsize=12.5)
ax.set_xlim(2014.5, 2100.5); ax.set_xticks(np.arange(2020, 2101, 10)); ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", color="0.85", lw=.7); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Year"); ax.set_ylabel("Projected turbine discharge (m³/s)")
sec = ax.secondary_yaxis("right", functions=(lambda q: 100 * (q / QT["annual"] - 1), lambda p: QT["annual"] * (1 + p / 100))); sec.set_ylabel("Change relative to baseline (%)")
ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=.9)
fig.tight_layout(); fig.savefig(OUT + f"fig4_15_projected_discharge_trajectory{SUF}.png", dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close()

# ===================== Fig 4.16  hydropower potential and surplus/shortfall, three drivers =====================
fig, (axa, axb) = plt.subplots(1, 2, figsize=(16, 6.5)); fig.patch.set_facecolor("white")
DC = {"annual": "#2980B9", "dry": "#C0392B", "wet": "#27AE60"}
x3 = np.arange(4); bw3 = 0.26
for di, w in enumerate(["annual", "dry", "wet"]):
    off = (di - 1) * bw3
    means, lows, highs, meds = [], [], [], []
    for s, p, _ in SPC:
        pv = np.array([P(Qproj(w, g, s, p), H_MEAN) for g in GCMS]); ens = P(Qproj(w, "Ensemble", s, p), H_MEAN)
        means.append(ens); lows.append(pv.min()); highs.append(pv.max()); meds.append(np.median(pv))
    means, lows, highs = map(np.array, (means, lows, highs))
    for ax, sh in ((axa, 0.0), (axb, -P_INS)):
        ax.bar(x3 + off, means + sh, width=bw3, color=DC[w], alpha=0.82, label=DNAME[w], edgecolor="white", linewidth=0.4, zorder=3)
        ax.errorbar(x3 + off, means + sh, yerr=[means - lows, highs - means], fmt="none", color="black", capsize=4, linewidth=1.1, zorder=4)
        ax.scatter(x3 + off, np.array(meds) + sh, marker="D", s=22, color="white", edgecolor="black", linewidth=0.8, zorder=5, label="GCM median" if di == 0 and ax is axa else "_")
for w, ls in (("annual", "-"),):
    axa.axhline(P(QT[w], H_MEAN), color="black", linewidth=2.0, label=f"Baseline power, annual ({P(QT[w], H_MEAN):.0f} MW)")
axa.axhline(P_FIRM, color="#E74C3C", linewidth=1.8, linestyle="--", label=f"Firm power threshold ({P_FIRM:.0f} MW)")
axa.axhline(P_INS, color="#2C3E50", linewidth=1.0, linestyle=":", label=f"Installed capacity ({P_INS:.0f} MW)")
axb.axhline(0, color="#2C3E50", linewidth=1.3, label=f"Installed capacity ({P_INS:.0f} MW)")
for ax in (axa, axb):
    ax.set_xticks(x3); ax.set_xticklabels([l for _, _, l in SPC], fontsize=11); ax.tick_params(labelsize=10); grid(ax)
axa.set_ylim(bottom=0); axa.set_ylabel("Projected Hydropower Potential (MW)", fontsize=11)
axb.set_ylabel("Hydropower Potential minus Installed Capacity (MW)", fontsize=11)
axa.set_title(f"(a)  Hydropower Potential by ΔQ Driver ({HEADLAB.split(' (')[0]}, mean head {H_MEAN + 1e-9:.1f} m)", fontsize=11.5, fontweight="bold")
axb.set_title("(b)  Surplus / Shortfall vs Installed Capacity", fontsize=11.5, fontweight="bold")
axa.legend(fontsize=9, loc="upper left", framealpha=0.9); axb.legend(fontsize=9, loc="upper left", framealpha=0.9)
suptitle(fig, "Projected Hydropower Potential at Kainji Dam: Annual-Mean vs Dry-Season vs Wet-Season HMF-WA ΔQ Driver\n"
             "Bars = 5-GCM ensemble mean; whiskers = GCM range (min–max); diamonds = GCM median", fontsize=13, fontweight="bold", color="#1A252F", y=1.02)
plt.tight_layout(); plt.savefig(OUT + f"fig4_16_hydropower_potential_surplus{SUF}.png", dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close()

# ===================== Fig 4.17  scenario matrix =====================
XR = (-60, 125) if MODE == "user" else (-40, 125)
dq = np.linspace(*XR, 400); Hr = np.linspace(H_MIN, H_MAX, 300)
DQ, HH = np.meshgrid(dq, Hr); PG = P(np.maximum(QT["annual"] * (1 + DQ / 100), 0), HH)
fig, ax = plt.subplots(figsize=(11, 7.5)); fig.patch.set_facecolor("white")
vmax = float(np.ceil(PG.max() / 100) * 100)
im = ax.contourf(dq, Hr, PG, levels=np.linspace(0, vmax, 25), cmap="RdYlGn", vmin=0, vmax=vmax)
iso = [200, 380, 500, 600, 760, 900, 1000, 1200, 1400, 1600, 1800, 2000] if MODE == "user" else [200, 300, 380, 500, 600, 760]
cs = ax.contour(dq, Hr, PG, levels=[v for v in iso if v < vmax], colors="black",
                linewidths=[1.8 if v == 380 else 1.4 if v == 760 else 0.8 for v in iso if v < vmax],
                linestyles=["-" if v in (380, 760) else "--" for v in iso if v < vmax])
ax.clabel(cs, fmt="%d MW", fontsize=8, inline=True)
ax.axvline(0, color="grey", linewidth=0.7, linestyle=":")
# projected annual-mean dQ markers (ensemble mean = filled, GCMs = ticks)
ymk = H_MIN + 0.03 * (H_MAX - H_MIN)
for (s, p, lab, col), dy in zip(COMBOS, [0.0, 0.035, 0.07, 0.105]):
    vals = [pct("annual", g, s, p) for g in GCMS]
    yy = H_MIN + (0.03 + dy) * (H_MAX - H_MIN)
    ax.scatter(vals, [yy] * len(vals), marker="|", s=110, color=col, linewidths=1.6, zorder=6)
    ax.scatter([pct_ens("annual", s, p)], [yy], marker="o", s=55, color=col, edgecolor="black", linewidth=0.9, zorder=7)
ax.set_xlabel("Change in Annual Mean River Flow, ΔQ (%)", fontsize=11); ax.set_ylabel(f"{'Net Head, H (m)' if MODE=='net' else 'Net Head, H (m)  [dissertation definition]'}", fontsize=11)
ax.set_xlim(*XR); ax.set_ylim(H_MIN, H_MAX); ax.tick_params(labelsize=10)
cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02); cb.set_label(f"Projected Hydropower Potential (MW)  [baseline Qt = {QT['annual']:.2f} m³/s]", fontsize=11); cb.ax.tick_params(labelsize=10)
leg = [Line2D([], [], marker="o", color=c, markeredgecolor="black", ls="", markersize=7, label=f"{l}: ensemble mean (|: individual GCMs)") for _, _, l, c in COMBOS]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.5, framealpha=0.92)
suptitle(fig, "Hydropower Generation Sensitivity Matrix — Kainji Dam\nHydropower Potential (MW) as a Function of Annual-Mean River-Flow Change (HMF-WA ΔQ) and Head", fontsize=13, fontweight="bold", color="#1A252F", y=1.01)
plt.savefig(OUT + f"fig4_17_scenario_matrix{SUF}.png", dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close()

# ===================== Fig 4.18  critical discharge threshold =====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True); fig.patch.set_facecolor("white")
XR2 = (-65, 235) if MODE == "user" else (-45, 235)
dqr = np.linspace(*XR2, 600)
for ai, Hv in enumerate([H_MIN, H_MAX]):
    ax = axes[ai]
    for w, col in (("annual", "#2980B9"), ("dry", "#C0392B")):
        Pc = P(np.maximum(QT[w] * (1 + dqr / 100), 0), Hv)
        ax.plot(dqr, Pc, color=col, lw=2.2, label=f"{DNAME[w]} baseline Qt = {QT[w]:.1f} m³/s")
        dstar = 100 * (P_FIRM * 1e6 / (ETA * RHO * G * Hv) / QT[w] - 1)
        ax.plot([dstar], [P_FIRM], marker="o", color=col, markersize=8, markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        ax.annotate(f"ΔQ* = {dstar:+.1f}%", xy=(dstar, P_FIRM), xytext=(dstar + 9, P_FIRM + (95 if w == "annual" else -125)), fontsize=9, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.axhline(P_FIRM, color="#E74C3C", lw=1.8, ls="--", label=f"Firm power ({P_FIRM:.0f} MW)")
    ax.axhline(P_INS, color="#2C3E50", lw=1.0, ls=":", label=f"Installed capacity ({P_INS:.0f} MW)")
    ax.axhline(P(QT["annual"], Hv), color="black", lw=1.5, label=f"Baseline power, annual ({P(QT['annual'], Hv):.0f} MW)")
    ax.axvline(0, color="grey", lw=0.7, ls=":")
    ax.axhspan(0, P_FIRM, facecolor="#FADBD8", alpha=0.30, zorder=0); ax.text(XR2[0] + 1.5, P_FIRM * 0.45, "Below\nFirm Power", fontsize=9, color="#C0392B", style="italic")
    bands = [("SSP245", "near", "SSP2-4.5 Mid-century GCM range", "#AED6F1", 0.95, 1.00), ("SSP245", "far", "SSP2-4.5 Late-century GCM range", "#2980B9", 0.90, 0.95),
             ("SSP585", "near", "SSP5-8.5 Mid-century GCM range", "#F1948A", 0.85, 0.90), ("SSP585", "far", "SSP5-8.5 Late-century GCM range", "#C0392B", 0.80, 0.85)]
    for s, p, bl, c, y0, y1 in bands:
        v = [pct("annual", g, s, p) for g in GCMS]
        ax.axvspan(min(v), max(v), ymin=y0, ymax=y1, facecolor=c, alpha=0.55, label=bl if ai == 0 else "_", zorder=1)
    ax.set_xlim(*XR2); ax.set_ylim(0, P_INS * (2.4 if MODE == "user" else 1.4)); ax.set_xlabel("Change in River Flow, ΔQ (%)", fontsize=11)
    if ai == 0: ax.set_ylabel("Hydropower Potential (MW)", fontsize=11)
    ax.set_title(f"({'ab'[ai]})  H = {Hv:g} m ({'minimum' if ai == 0 else 'maximum'} head)", fontsize=12, fontweight="bold", color="#1A252F")
    ax.tick_params(labelsize=10); grid(ax)
axes[0].legend(fontsize=8.0, loc="lower right", framealpha=0.9); axes[1].legend(fontsize=8.0, loc="lower right", framealpha=0.9)
suptitle(fig, "Critical River-Flow Threshold for Firm Power Generation at Kainji Dam\nDots mark the exact ΔQ* at which projected power equals the firm-power threshold (380 MW); GCM bands show the HMF-WA annual-mean ΔQ range",
             fontsize=13, fontweight="bold", color="#1A252F", y=1.02)
plt.tight_layout(); plt.savefig(OUT + f"fig4_18_critical_discharge_threshold{SUF}.png", dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close()
print("figures written to", OUT)
