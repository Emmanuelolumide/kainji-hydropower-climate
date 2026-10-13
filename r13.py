"""Module 3 figures for the final document: NET HEAD (33-42 m, Mohammed et al. 2019) and ANNUAL-MEAN driver in the main text;
dry/wet-season drivers and the head-definition comparison go to the appendix.  Output: plots/doc/ (no titles)."""
import os, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import *
D = pickle.load(open(R + "/data/module3.pkl", "rb"))
F, QT, GCMS, PW, CR = D["F"], D["QT"], D["GCMS"], D["PW"], D["crit"]
OUT = "./output/"; os.makedirs(OUT, exist_ok=True)
HN, HU = (33.0, 42.0), (129.0, 141.9); HNM = 37.5; HUM = 0.5 * (HU[0] + HU[1])
ETA, RHO, G, P_INS, P_FIRM = 0.87, 1000.0, 9.81, 760.0, 292.44
OBS_MW = 2519.459 * 1000 / 8760
FS = 1.15      # font scale: the figures are printed 6-7 inches wide in the document
QMIN_M, QMAX_M = 334.70, 1271.63
def P(Q, H): return ETA * RHO * G * Q * H / 1e6
def dq(w, g, s, p):
    x = F[(F.driver == w) & (F.scenario == s) & (F.period == p)]
    return float(x.pct.mean()) if g == "Ensemble" else float(x[x.gcm == g].pct.iloc[0])
def qproj(w, g, s, p): return max(0.0, QT[w] * (1 + dq(w, g, s, p) / 100))
COMBOS = [("SSP245", "near", "SSP2-4.5 Mid-century", "#AED6F1"), ("SSP245", "far", "SSP2-4.5 Late-century", "#2980B9"), ("SSP585", "near", "SSP5-8.5 Mid-century", "#F1948A"), ("SSP585", "far", "SSP5-8.5 Late-century", "#C0392B")]
SPC = [("SSP245", "near", "SSP2-4.5\nMid-century"), ("SSP245", "far", "SSP2-4.5\nLate-century"), ("SSP585", "near", "SSP5-8.5\nMid-century"), ("SSP585", "far", "SSP5-8.5\nLate-century")]
DN = {"annual": "Annual mean", "dry": "Dry season (Nov–Mar)", "wet": "Wet season (Apr–Oct)"}
DC = {"annual": "#2980B9", "dry": "#C0392B", "wet": "#27AE60"}
def grid(ax):
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD", linestyle="--", zorder=0); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
def crit(mode, w, hh): return CR[(CR.head_mode == mode) & (CR.driver == w) & (CR["head"] == hh)].iloc[0]
def save(fig, name): fig.savefig(OUT + name, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig); print("saved", name)

# ---------------------------------------------------------------- discharge by GCM (Fig 4.14 / A.4)
def fig_discharge(drivers, name):
    n = len(drivers); fig, axes = plt.subplots(1 if n == 1 else n, n if n == 1 else 1, figsize=(12.5, 5.4) if n == 1 else (9.0, 9.4), sharey=True, sharex=True); axes = np.atleast_1d(axes)
    labels = GCMS + ["Ensemble\nMean"]; x = np.arange(len(labels)); bw = 0.18; offs = [-1.5, -0.5, 0.5, 1.5]
    for ai, w in enumerate(drivers):
        ax = axes[ai]
        for ci, (s, p, lab, col) in enumerate(COMBOS):
            v = [qproj(w, g, s, p) for g in GCMS] + [qproj(w, "Ensemble", s, p)]
            ax.bar(x[:-1] + offs[ci] * bw, v[:-1], width=bw, color=col, alpha=0.85, label=lab if ai == 0 else "_nolegend_", edgecolor="white", linewidth=0.4, zorder=3)
            ax.bar(x[-1] + offs[ci] * bw, v[-1], width=bw, color=col, alpha=0.85, label="_nolegend_", edgecolor="black", linewidth=1.3, zorder=3)
        ax.axvline(len(GCMS) - 0.5, color="#888888", linewidth=1.0, zorder=2)
        ax.axhline(QT[w], color="black", linewidth=1.6, linestyle="--", label=f"Baseline mean discharge ({QT[w]:.0f} m³/s)" if ai == 0 else "_")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9*FS)
        if n == 1 and ai == 0: ax.set_ylabel("Projected Turbine Discharge Q (m³/s)", fontsize=11*FS)
        if n > 1:
            ax.set_title(f"{DN[w]} ΔQ driver  (baseline Qt = {QT[w]:.1f} m³/s)", fontsize=12*FS, fontweight="bold", color="#1A252F")
            ax.text(0.01, 0.98, f"({'abc'[ai]})", transform=ax.transAxes, fontsize=13*FS, fontweight="bold", va="top", ha="left")
        ax.tick_params(labelsize=10*FS); grid(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=(9 if n == 1 else 7.4)*FS, bbox_to_anchor=(0.5, -0.1 if n == 1 else -0.045), framealpha=0.95)
    plt.tight_layout(rect=(0.035, 0, 1, 1) if n > 1 else None)
    if n > 1: fig.supylabel("Projected Turbine Discharge Q (m³/s)", fontsize=11*FS, x=0.012)
    save(fig, name)

# ---------------------------------------------------------------- potential and surplus (Fig 4.16 / A.5)
def fig_potential(drivers, name):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5.2)); x3 = np.arange(4); n = len(drivers); bw3 = 0.5 if n == 1 else 0.26
    for di, w in enumerate(drivers):
        off = (di - (n - 1) / 2) * bw3; means, lows, highs, meds = [], [], [], []
        for s, p, _ in SPC:
            pv = np.array([P(qproj(w, g, s, p), HNM) for g in GCMS]); means.append(P(qproj(w, "Ensemble", s, p), HNM)); lows.append(pv.min()); highs.append(pv.max()); meds.append(np.median(pv))
        means, lows, highs = map(np.array, (means, lows, highs))
        for ax, sh in ((axa, 0.0), (axb, -P(QT[w], HNM))):
            ax.bar(x3 + off, means + sh, width=bw3, color=DC[w], alpha=0.82, label=DN[w], edgecolor="white", linewidth=0.4, zorder=3)
            ax.errorbar(x3 + off, means + sh, yerr=[means - lows, highs - means], fmt="none", color="black", capsize=4, linewidth=1.1, zorder=4)
            ax.scatter(x3 + off, np.array(meds) + sh, marker="D", s=24, color="white", edgecolor="black", linewidth=0.8, zorder=5, label="GCM median" if (di == 0 and ax is axa) else "_")
    axa.axhline(P(QT["annual"], HNM), color="black", linewidth=2.0, label=f"Baseline potential, annual ({P(QT['annual'], HNM):.0f} MW)")
    axa.axhline(P_FIRM, color="#E74C3C", linewidth=1.6, linestyle="--", label=f"Firm-power threshold, 2014 minimum ({P_FIRM:.2f} MW)")
    axa.axhline(P_INS, color="#2C3E50", linewidth=1.0, linestyle=":", label=f"Installed capacity ({P_INS:.0f} MW)")
    axb.axhline(0, color="#2C3E50", linewidth=1.3)
    for ax in (axa, axb):
        ax.set_xticks(x3); ax.set_xticklabels([l for _, _, l in SPC], fontsize=11*FS); ax.tick_params(labelsize=10*FS); grid(ax)
    axa.set_ylim(bottom=0); axa.set_ylabel("Projected Hydropower Potential (MW)", fontsize=11*FS); axb.set_ylabel("Projected Potential minus Baseline Potential (MW)", fontsize=11*FS)
    axa.set_title(f"(a)  Hydropower Potential (net head {HNM:g} m)", fontsize=11.5*FS, fontweight="bold"); axb.set_title("(b)  Surplus / Shortfall vs Baseline Potential", fontsize=11.5*FS, fontweight="bold")
    h, l = axa.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3 if n == 1 else 4, fontsize=9*FS, bbox_to_anchor=(0.5, -0.17 if n == 1 else -0.2), framealpha=0.95)
    plt.tight_layout(); save(fig, name)

# ---------------------------------------------------------------- scenario matrix (Fig 4.17)
def fig_matrix(name):
    XR = (-20, 235); dqs = np.linspace(*XR, 500); Hr = np.linspace(*HN, 300); DQ, HH = np.meshgrid(dqs, Hr); PG = P(QT["annual"] * (1 + DQ / 100), HH)
    fig, ax = plt.subplots(figsize=(10, 7.2)); vmax = float(np.ceil(PG.max() / 100) * 100)
    im = ax.contourf(dqs, Hr, PG, levels=np.linspace(0, vmax, 25), cmap="RdYlGn", vmin=0, vmax=vmax)
    iso = [200, 292.44, 400, 500, 600, 760]
    cs = ax.contour(dqs, Hr, PG, levels=iso, colors="black", linewidths=[1.8 if v == 292.44 else 1.4 if v == 760 else 0.8 for v in iso], linestyles=["-" if v in (292.44, 760) else "--" for v in iso])
    ax.clabel(cs, fmt=lambda v: f"{v:g} MW", fontsize=8*FS, inline=True); ax.axvline(0, color="grey", linewidth=0.7, linestyle=":")
    for (s, p, lab, col), dy in zip(COMBOS, [0.0, 0.035, 0.07, 0.105]):
        vals = [dq("annual", g, s, p) for g in GCMS]; yy = HN[0] + (0.035 + dy) * (HN[1] - HN[0])
        ax.scatter(vals, [yy] * len(vals), marker="|", s=110, color=col, linewidths=1.6, zorder=6); ax.scatter([dq("annual", "Ensemble", s, p)], [yy], marker="o", s=55, color=col, edgecolor="black", linewidth=0.9, zorder=7)
    ax.set_xlabel("Change in Annual Mean River Flow, ΔQ (%)", fontsize=11*FS); ax.set_ylabel("Net Head, H (m)", fontsize=11*FS); ax.set_xlim(*XR); ax.set_ylim(*HN); ax.tick_params(labelsize=10*FS)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02); cb.set_label(f"Projected Hydropower Potential (MW)  [baseline Qt = {QT['annual']:.2f} m³/s]", fontsize=11*FS); cb.set_ticks(np.arange(0, vmax + 1, 100)); cb.ax.tick_params(labelsize=10*FS)
    leg = [Line2D([], [], marker="o", color=c, markeredgecolor="black", ls="", markersize=7, label=f"{l}: ensemble mean (|: individual GCMs)") for _, _, l, c in COMBOS]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.5*FS, framealpha=0.92); save(fig, name)

# ---------------------------------------------------------------- flow change needed for the reference levels (Fig 4.18 / A.6)
def fig_critical(show_dry, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True); XR = (-25, 290); dqr = np.linspace(*XR, 700)
    for ai, (hh, Hv) in enumerate((("Hmin", HN[0]), ("Hmax", HN[1]))):
        ax = axes[ai]
        for w, col in (("annual", "#2980B9"),) + ((("dry", "#C0392B"),) if show_dry else ()):
            ax.plot(dqr, P(QT[w] * (1 + dqr / 100), Hv), color=col, lw=2.2, label=f"{DN[w]} baseline Qt = {QT[w]:.1f} m³/s")
            c = crit("net", w, hh)
            for lvl, dqv, tag in ((P_FIRM, 100*(P_FIRM/P(QT[w], Hv)-1), "292.44 MW"), (P_INS, c.dQ_inst_pct, "760 MW")):
                ax.plot([dqv], [lvl], marker="o", color=col, markersize=8, markeredgecolor="black", zorder=5)
                if lvl == P_INS and w == "annual":
                    ax.annotate(f"ΔQ = {dqv:+.1f}%", xy=(dqv, lvl), xytext=(dqv - 14, lvl + 24), fontsize=8.5*FS, color=col, fontweight="bold", ha="right", va="bottom")
                elif lvl == P_INS:
                    ax.annotate(f"ΔQ = {dqv:+.1f}%", xy=(dqv, lvl), xytext=(dqv + 6, lvl - 170), fontsize=8.5*FS, color=col, fontweight="bold", ha="center", va="top", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
                else:
                    ax.annotate(f"ΔQ = {dqv:+.1f}%", xy=(dqv, lvl), xytext=(max(dqv - (62 if w == "annual" else 20), -20) + (40 if dqv < 10 else 0), lvl + (80 if w == "annual" else -95)), fontsize=8.5*FS, color=col, fontweight="bold", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
        ax.axhline(P_FIRM, color="#E74C3C", lw=1.6, ls="--", label=f"Firm-power threshold, 2014 minimum ({P_FIRM:.2f} MW)")
        ax.axhline(P_INS, color="#2C3E50", lw=1.0, ls=":", label=f"Installed capacity ({P_INS:.0f} MW)")
        ax.axhline(OBS_MW, color="#7F8C8D", lw=1.6, label=f"Observed mean generation ({OBS_MW:.0f} MW)")
        ax.axhline(P(QT["annual"], Hv), color="black", lw=1.5, label=f"Baseline potential, annual ({P(QT['annual'], Hv):.0f} MW)"); ax.axvline(0, color="grey", lw=0.7, ls=":")
        bands = [("SSP245", "near", "SSP2-4.5 Mid-century GCM range", "#AED6F1", 0.95, 1.00), ("SSP245", "far", "SSP2-4.5 Late-century GCM range", "#2980B9", 0.90, 0.95), ("SSP585", "near", "SSP5-8.5 Mid-century GCM range", "#F1948A", 0.85, 0.90), ("SSP585", "far", "SSP5-8.5 Late-century GCM range", "#C0392B", 0.80, 0.85)]
        for s, p, bl, c_, y0, y1 in bands:
            v = [dq("annual", g, s, p) for g in GCMS]; ax.axvspan(min(v), max(v), ymin=y0, ymax=y1, facecolor=c_, alpha=0.55, label=bl if ai == 0 else "_", zorder=1)
        ax.set_xlim(*XR); ax.set_ylim(0, 1050); ax.set_xlabel("Change in River Flow, ΔQ (%)", fontsize=11*FS)
        if ai == 0: ax.set_ylabel("Hydropower Potential (MW)", fontsize=11*FS)
        ax.set_title(f"({'ab'[ai]})  H = {Hv:g} m ({'minimum' if ai == 0 else 'maximum'} net head)", fontsize=12*FS, fontweight="bold", color="#1A252F"); ax.tick_params(labelsize=10*FS); grid(ax)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5 if not show_dry else 4, fontsize=8.5*FS, bbox_to_anchor=(0.5, -0.17), framealpha=0.95); plt.tight_layout(); save(fig, name)

# ---------------------------------------------------------------- head definition (Fig 4.19 single panel; Fig A.7 two panels)
def fig_head(two_panels, name):
    fig, axs = plt.subplots(1, 2 if two_panels else 1, figsize=(12.5, 5.3) if two_panels else (8.5, 6.2), gridspec_kw={"width_ratios": [1, 1.6]} if two_panels else None); a = axs[0] if two_panels else axs
    vals = [("Reservoir water-surface\nelevation 129.0-141.9 m\nused as H", P(QT["annual"], np.mean(HU)), (P(QT["annual"], HU[0]), P(QT["annual"], HU[1])), "#E67E22"), ("Net head 33-42 m\n(Mohammed et al., 2019)", P(QT["annual"], HNM), (P(QT["annual"], HN[0]), P(QT["annual"], HN[1])), "#2980B9")]
    for i, (lab, m, (lo, hi), c) in enumerate(vals):
        a.bar(i, m, width=0.55, color=c, alpha=0.85, edgecolor="white", zorder=3); a.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="black", capsize=5, lw=1.3, zorder=4); a.text(i, hi + 25, f"{m:.0f} MW", ha="center", fontsize=11*FS, fontweight="bold")
    a.bar(2, OBS_MW, width=0.55, color="#7F8C8D", alpha=0.85, edgecolor="white", zorder=3); a.text(2, OBS_MW + 25, f"{OBS_MW:.0f} MW", ha="center", fontsize=11*FS, fontweight="bold")
    a.axhline(P_INS, color="#2C3E50", lw=1.0, ls=":", label=f"Installed capacity ({P_INS:.0f} MW)"); a.axhline(P_FIRM, color="#E74C3C", lw=1.6, ls="--", label=f"Firm-power threshold, 2014 minimum ({P_FIRM:.2f} MW)")
    a.set_xticks(range(3)); a.set_xticklabels([v[0] for v in vals] + ["Observed mean\ngeneration\n(2,519 GWh/yr)"], fontsize=9.5*FS); a.set_ylabel("Power at the baseline discharge (MW)", fontsize=11*FS); a.set_ylim(0, 1050)
    a.set_title(f"{'(a)  ' if two_panels else ''}Baseline check: P = η ρ g Qt H, Qt = {QT['annual']:.2f} m³/s", fontsize=11.5*FS, fontweight="bold"); a.legend(fontsize=9*FS, loc="upper right"); grid(a)
    if two_panels:
        b = axs[1]; x = np.arange(4); bw = 0.34
        for hi_, (lab, hh, c) in enumerate([("Reservoir elevation range 129.0-141.9 m (mean 135.5 m)", HUM, "#E67E22"), (f"Net head 33-42 m (mean {HNM:g} m)", HNM, "#2980B9")]):
            m_, lo_, hi2 = [], [], []
            for s, p, _ in SPC:
                pv = [P(qproj("annual", g, s, p), hh) for g in GCMS]; m_.append(P(qproj("annual", "Ensemble", s, p), hh)); lo_.append(min(pv)); hi2.append(max(pv))
            m_, lo_, hi2 = map(np.array, (m_, lo_, hi2)); off = (hi_ - 0.5) * bw
            b.bar(x + off, m_, width=bw, color=c, alpha=0.85, label=lab, edgecolor="white", zorder=3); b.errorbar(x + off, m_, yerr=[m_ - lo_, hi2 - m_], fmt="none", color="black", capsize=4, lw=1.1, zorder=4)
        b.axhline(P_INS, color="#2C3E50", lw=1.0, ls=":", label=f"Installed capacity ({P_INS:.0f} MW)"); b.axhline(P_FIRM, color="#E74C3C", lw=1.6, ls="--", label=f"Firm-power threshold ({P_FIRM:.0f} MW)"); b.axhline(OBS_MW, color="#7F8C8D", lw=1.6, label=f"Observed mean generation ({OBS_MW:.0f} MW)")
        b.set_xticks(x); b.set_xticklabels([l for _, _, l in SPC], fontsize=11*FS); b.set_ylim(0, 3000); b.set_ylabel("Projected hydropower potential (MW)", fontsize=11*FS)
        b.set_title("(b)  Projected potential, annual-mean driver (bars: ensemble mean; whiskers: GCM range)", fontsize=11.5*FS, fontweight="bold"); b.legend(fontsize=9*FS, loc="upper left"); grid(b)
    plt.tight_layout(); save(fig, name)

# ---------------------------------------------------------------- projected potential with both head definitions (Fig A.7)
def fig_head_proj(name):
    fig, b = plt.subplots(figsize=(9.0, 5.8)); x = np.arange(4); bw = 0.34
    for hi_, (lab, hh, c) in enumerate([("Reservoir elevation range 129.0-141.9 m (mean 135.5 m)", HUM, "#E67E22"), (f"Net head 33-42 m (mean {HNM:g} m)", HNM, "#2980B9")]):
        m_, lo_, hi2 = [], [], []
        for s, p, _ in SPC:
            pv = [P(qproj("annual", g, s, p), hh) for g in GCMS]; m_.append(P(qproj("annual", "Ensemble", s, p), hh)); lo_.append(min(pv)); hi2.append(max(pv))
        m_, lo_, hi2 = map(np.array, (m_, lo_, hi2)); off = (hi_ - 0.5) * bw
        b.bar(x + off, m_, width=bw, color=c, alpha=0.85, label=lab, edgecolor="white", zorder=3); b.errorbar(x + off, m_, yerr=[m_ - lo_, hi2 - m_], fmt="none", color="black", capsize=4, lw=1.1, zorder=4)
    b.axhline(P_INS, color="#2C3E50", lw=1.0, ls=":", label=f"Installed capacity ({P_INS:.0f} MW)"); b.axhline(P_FIRM, color="#E74C3C", lw=1.6, ls="--", label=f"Firm-power threshold ({P_FIRM:.0f} MW)"); b.axhline(OBS_MW, color="#7F8C8D", lw=1.6, label=f"Observed mean generation ({OBS_MW:.0f} MW)")
    b.set_xticks(x); b.set_xticklabels([l for _, _, l in SPC], fontsize=11*FS); b.set_ylim(0, 3400); b.set_ylabel("Projected hydropower potential (MW)", fontsize=11*FS); b.tick_params(labelsize=10*FS)
    b.legend(fontsize=9*FS, loc="upper left", framealpha=0.92); grid(b); plt.tight_layout(); save(fig, name)

fig_discharge(["annual"], "fig4_14_projected_discharge_by_gcm.png")
fig_discharge(["annual", "dry", "wet"], "figA4_projected_discharge_by_driver.png")
fig_potential(["annual"], "fig4_16_hydropower_potential_surplus.png")
fig_potential(["annual", "dry", "wet"], "figA5_hydropower_potential_by_driver.png")
fig_matrix("fig4_17_scenario_matrix.png")
fig_critical(False, "fig4_18_critical_discharge_threshold.png")
fig_critical(True, "figA6_critical_threshold_dry_baseline.png")
fig_head(False, "fig4_19_head_baseline_check.png")
fig_head_proj("figA7_head_definition_sensitivity.png")
