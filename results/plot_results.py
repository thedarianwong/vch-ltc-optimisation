"""
results/plot_results.py — Generate all figures for FINAL_RESULTS.md

Run:  PYTHONPATH=. python results/plot_results.py
Outputs: results/figures/*.png
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.data_loader import load_all
from model.core.logistic import prepare_model_inputs
from model.parameters import (
    PRIORITY_LABELS, ARRIVAL_RATE, VACANCY_RATE, ABANDONMENT_RATE,
    INITIAL_QUEUE_SIZE, OFFER_TURNAROUND_DAYS, funding_to_h_n,
)
from model.simulation.queue_model import run_replications, summarise_replications
from model.simulation.des import run_des_replications, summarise_des_replications

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette — consistent across all figures
# ---------------------------------------------------------------------------
C_EA  = "#4C72B0"   # Erlang-A  (blue)
C_CF  = "#DD8452"   # Cyclic FIFO (orange)
C_OPT = "#55A868"   # Optimised  (green)
C_LP  = "#C44E52"   # LP Optimal (red)
C_HIS = "#8172B3"   # Historical (purple)
C_RND = "#937860"   # Random     (brown)

TIER_COLOURS = {
    "Transfer/Site Specific": "#4C72B0",
    "Community High":         "#55A868",
    "Community Emergency":    "#DD8452",
    "Acute Care":             "#C44E52",
}

FIG_DPI = 150
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": FIG_DPI,
})


# ===========================================================================
# Load data and run simulations
# ===========================================================================

print("Loading data...")
data   = load_all()
inputs = prepare_model_inputs(
    data["waitlist_entry"],
    data["facility_details"],
    data["vacancies"],
    verbose=False,
)
facilities_df  = inputs["facilities"]
facility_names = inputs["facility_names"]
capacities     = [int(c) for c in inputs["capacities"]]
u_n_map        = inputs["u_n_map"]
h_n_arr        = np.array([funding_to_h_n(facilities_df.loc[i, "funding"])
                            for i in range(len(facility_names))])
u_n_arr        = np.array([u_n_map.get(n, 0.0) for n in facility_names])
gender_lim_arr = [None] * len(facility_names)
facility_regions = [
    str(facilities_df.loc[i, "community_region_desc"])
    if "community_region_desc" in facilities_df.columns else "Vancouver"
    for i in range(len(facility_names))
]

# --- Queue model (10 reps) ---
print("Running queue model (10 reps × 3 policies)...")
N_REPS = 10; SIM_DAYS = 1095; WARMUP = 180

qm_kwargs = dict(facility_names=facility_names, capacities=capacities,
                 h_n_arr=h_n_arr, u_n_arr=u_n_arr, gender_lim_arr=gender_lim_arr,
                 n_reps=N_REPS, seed_base=0, sim_days=SIM_DAYS, warmup_days=WARMUP)

reps_ea_q  = run_replications(policy="erlang_a",    **qm_kwargs)
reps_cf_q  = run_replications(policy="cyclic_fifo", **qm_kwargs)
reps_opt_q = run_replications(policy="optimised",   **qm_kwargs)

s_ea_q  = summarise_replications(reps_ea_q)
s_cf_q  = summarise_replications(reps_cf_q)
s_opt_q = summarise_replications(reps_opt_q)

# --- DES (30 reps) ---
print("Running DES (30 reps × 3 policies)...")
N_DES = 30
des_kwargs = dict(facility_names=facility_names, capacities=capacities,
                  h_n_arr=h_n_arr, u_n_arr=u_n_arr,
                  facility_regions=facility_regions, gender_lim_arr=gender_lim_arr,
                  n_reps=N_DES, seed_base=0, sim_days=SIM_DAYS)

reps_ea_d  = run_des_replications(policy="erlang_a",    **des_kwargs)
reps_cf_d  = run_des_replications(policy="cyclic_fifo", **des_kwargs)
reps_opt_d = run_des_replications(policy="optimised",   **des_kwargs)

s_ea_d  = summarise_des_replications(reps_ea_d)
s_cf_d  = summarise_des_replications(reps_cf_d)
s_opt_d = summarise_des_replications(reps_opt_d)

print("Simulations complete. Generating figures...")


# ===========================================================================
# Helper
# ===========================================================================

def _m(s, k):   return s[k]["mean"]
def _ci(s, k):  return s[k]["half_ci"]
def _tier_m(s, t, k):  return s["by_tier"][t][k]["mean"]
def _tier_ci(s, t, k): return s["by_tier"][t][k]["half_ci"]


# ===========================================================================
# Figure 1 — System Calibration: Steady-State Queue Length
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Figure 1 — System Calibration: Bed-Scarce Regime", fontweight="bold")

# Panel A: steady-state math diagram
ax = axes[0]
months = np.arange(0, 37)
lam  = 28   # arrivals/month
mu   = 18   # vacancies/month
th   = 0.02 # abandonment rate/month
Q_ss = (lam - mu) / th  # = 500

# Analytical trajectory starting from Q0
Q0 = 500
Q_t = Q_ss + (Q0 - Q_ss) * np.exp(-(th) * months)
ax.plot(months, [500]*len(months), "k--", lw=1.5, label="Q_ss = 500 (VCH reported)")
ax.plot(months, Q_t, color=C_CF, lw=2, label="Analytical approach to Q_ss")
ax.fill_between(months, Q_t - 30, Q_t + 30, alpha=0.15, color=C_CF)
ax.set_xlabel("Months of simulation")
ax.set_ylabel("Queue length (clients)")
ax.set_title("A) Steady-state calibration\nQ_ss = (λ − μ) / θ = (28−18)/0.02 = 500")
ax.legend()
ax.set_xlim(0, 36)
ax.set_ylim(400, 600)

# Panel B: observed queue lengths from simulations
ax = axes[1]
q_means = [
    _m(s_ea_q, "mean_queue_length"),
    _m(s_cf_q, "mean_queue_length"),
    _m(s_opt_q, "mean_queue_length"),
    _m(s_ea_d,  "mean_queue_length"),
    _m(s_cf_d,  "mean_queue_length"),
    _m(s_opt_d, "mean_queue_length"),
]
q_cis = [
    _ci(s_ea_q, "mean_queue_length"),
    _ci(s_cf_q, "mean_queue_length"),
    _ci(s_opt_q, "mean_queue_length"),
    _ci(s_ea_d,  "mean_queue_length"),
    _ci(s_cf_d,  "mean_queue_length"),
    _ci(s_opt_d, "mean_queue_length"),
]
labels = ["EA\n(Q4)", "Cyclic\n(Q4)", "Opt\n(Q4)",
          "EA\n(DES)", "Cyclic\n(DES)", "Opt\n(DES)"]
colours = [C_EA, C_CF, C_OPT, C_EA, C_CF, C_OPT]
bars = ax.bar(labels, q_means, color=colours, alpha=0.85, yerr=q_cis,
              capsize=4, edgecolor="white", linewidth=0.5)
ax.axhline(500, color="k", linestyle="--", lw=1.5, label="VCH reported backlog (500)")
ax.set_ylabel("Mean queue length (clients)")
ax.set_title("B) Simulated mean queue length\nacross all policies and models")
ax.set_ylim(0, 700)
ax.legend()
for bar, val in zip(bars, q_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 12,
            f"{val:.0f}", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_calibration.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved fig1_calibration.png")


# ===========================================================================
# Figure 2 — Declination Rates: All Models + LP
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 2 — Provider Declination Rates Across Methods", fontweight="bold")

# Panel A: overall declination rates
ax = axes[0]
# LP data (from compare.py results)
LP_DECLIN   = 46.43 / 500   # expected declinations / total clients
CF_LP_DECLIN = 113.48 / 500
RND_LP_DECLIN = 94.44 / 500
HIS_LP_DECLIN = 83.01 / 467

methods = ["Erlang-A\n(Q4)", "Cyclic FIFO\n(Q4)", "Optimised\n(Q4)",
           "Erlang-A\n(DES)", "Cyclic FIFO\n(DES)", "Optimised\n(DES)",
           "LP Optimal\n(Step 3)", "Cyclic FIFO\n(Step 3)"]
decl_vals = [
    _m(s_ea_q, "declination_rate"),
    _m(s_cf_q, "declination_rate"),
    _m(s_opt_q,"declination_rate"),
    _m(s_ea_d, "declination_rate"),
    _m(s_cf_d, "declination_rate"),
    _m(s_opt_d,"declination_rate"),
    LP_DECLIN,
    CF_LP_DECLIN,
]
decl_cis = [
    _ci(s_ea_q, "declination_rate"),
    _ci(s_cf_q, "declination_rate"),
    _ci(s_opt_q,"declination_rate"),
    _ci(s_ea_d, "declination_rate"),
    _ci(s_cf_d, "declination_rate"),
    _ci(s_opt_d,"declination_rate"),
    0, 0,
]
colours_decl = [C_EA,C_CF,C_OPT, C_EA,C_CF,C_OPT, C_LP, C_CF]
bars = ax.bar(methods, decl_vals, color=colours_decl, alpha=0.85,
              yerr=decl_cis, capsize=3, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Declination rate (declined / total offers)")
ax.set_title("A) Declination rate by method\n(lower is better)")
ax.set_ylim(0, 0.35)
for bar, val in zip(bars, decl_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.1%}", ha="center", va="bottom", fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Panel B: per-tier declination, DES
ax = axes[1]
tiers = PRIORITY_LABELS[::-1]  # Transfer last
x = np.arange(len(tiers))
w = 0.35
cf_tier_decl  = [_tier_m(s_cf_d, t, "declin_rate") for t in tiers]
opt_tier_decl = [_tier_m(s_opt_d, t, "declin_rate") for t in tiers]
cf_tier_ci    = [_tier_ci(s_cf_d, t, "declin_rate") for t in tiers]
opt_tier_ci   = [_tier_ci(s_opt_d, t, "declin_rate") for t in tiers]

ax.barh(x + w/2, cf_tier_decl, w, label="Cyclic FIFO", color=C_CF, alpha=0.85,
        xerr=cf_tier_ci, capsize=3)
ax.barh(x - w/2, opt_tier_decl, w, label="Optimised", color=C_OPT, alpha=0.85,
        xerr=opt_tier_ci, capsize=3)
ax.set_yticks(x)
ax.set_yticklabels([t.replace("/", "/\n") for t in tiers], fontsize=9)
ax.set_xlabel("Declination rate")
ax.set_title("B) Per-tier declination rate (DES, 30 reps)\n(lower is better)")
ax.legend()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_declination_rates.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved fig2_declination_rates.png")


# ===========================================================================
# Figure 3 — Bed-Days Wasted
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 3 — Bed-Days Wasted Due to Declinations", fontweight="bold")

# Panel A: total bed-days wasted comparison (Q4 vs DES)
ax = axes[0]
bdw_data = {
    "Queue\nErlang-A": (_m(s_ea_q,"bed_days_wasted"),  _ci(s_ea_q,"bed_days_wasted"),  C_EA),
    "Queue\nCyclic":   (_m(s_cf_q,"bed_days_wasted"),  _ci(s_cf_q,"bed_days_wasted"),  C_CF),
    "Queue\nOptimised":(_m(s_opt_q,"bed_days_wasted"), _ci(s_opt_q,"bed_days_wasted"), C_OPT),
    "DES\nErlang-A":   (_m(s_ea_d,"bed_days_wasted"),  _ci(s_ea_d,"bed_days_wasted"),  C_EA),
    "DES\nCyclic":     (_m(s_cf_d,"bed_days_wasted"),  _ci(s_cf_d,"bed_days_wasted"),  C_CF),
    "DES\nOptimised":  (_m(s_opt_d,"bed_days_wasted"), _ci(s_opt_d,"bed_days_wasted"), C_OPT),
}
names = list(bdw_data.keys())
vals  = [v[0] for v in bdw_data.values()]
cis   = [v[1] for v in bdw_data.values()]
cols  = [v[2] for v in bdw_data.values()]
bars = ax.bar(names, vals, color=cols, alpha=0.85, yerr=cis, capsize=4,
              edgecolor="white", linewidth=0.5)
ax.set_ylabel("Bed-days wasted (3-year run)")
ax.set_title("A) Total bed-days wasted per simulation\n(OFFER_TURNAROUND = 2 days/decline)")
for bar, val in zip(bars, vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

# Annotation: annual equivalent
q4_annual   = _m(s_cf_q,"bed_days_wasted") / 3
des_annual  = _m(s_cf_d,"bed_days_wasted") / 3
ax.annotate(f"≈ {q4_annual:.0f} bed-days/yr\n(Cyclic, Q4)",
            xy=(1, _m(s_cf_q,"bed_days_wasted")),
            xytext=(1.5, _m(s_cf_q,"bed_days_wasted") + 50),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")

# Panel B: bed-days wasted per placement
ax = axes[1]
bdw_pp = {
    "Cyclic FIFO\n(Queue)":   (_m(s_cf_q,"bed_days_per_place"),  _ci(s_cf_q,"bed_days_per_place"),  C_CF),
    "Optimised\n(Queue)":     (_m(s_opt_q,"bed_days_per_place"), _ci(s_opt_q,"bed_days_per_place"), C_OPT),
    "Cyclic FIFO\n(DES)":     (_m(s_cf_d,"bed_days_per_place"),  _ci(s_cf_d,"bed_days_per_place"),  C_CF),
    "Optimised\n(DES)":       (_m(s_opt_d,"bed_days_per_place"), _ci(s_opt_d,"bed_days_per_place"), C_OPT),
}
names2 = list(bdw_pp.keys())
vals2  = [v[0] for v in bdw_pp.values()]
cis2   = [v[1] for v in bdw_pp.values()]
cols2  = [v[2] for v in bdw_pp.values()]
bars2 = ax.bar(names2, vals2, color=cols2, alpha=0.85, yerr=cis2, capsize=4,
               edgecolor="white", linewidth=0.5)
ax.set_ylabel("Bed-days wasted per successful placement")
ax.set_title("B) Bed-days wasted per placement\n(normalised efficiency metric)")
for bar, val in zip(bars2, vals2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10)

# Draw Erlang-A = 0 reference line
ax.axhline(0, color="k", linestyle="--", lw=1, alpha=0.4, label="Erlang-A (0 = lower bound)")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_bed_days_wasted.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved fig3_bed_days_wasted.png")


# ===========================================================================
# Figure 4 — Mean Wait Times by Tier
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Figure 4 — Mean Wait Times by Priority Tier (DES, 30 reps)", fontweight="bold")

tiers = list(reversed(PRIORITY_LABELS))

# Panel A: DES cyclic vs optimised
ax = axes[0]
x = np.arange(len(tiers))
w = 0.35
cf_waits  = [_tier_m(s_cf_d, t, "mean_wait") for t in tiers]
opt_waits = [_tier_m(s_opt_d, t, "mean_wait") for t in tiers]
cf_ci     = [_tier_ci(s_cf_d, t, "mean_wait") for t in tiers]
opt_ci    = [_tier_ci(s_opt_d, t, "mean_wait") for t in tiers]

bars1 = ax.barh(x + w/2, cf_waits, w, label="Cyclic FIFO", color=C_CF, alpha=0.85,
                xerr=cf_ci, capsize=3)
bars2 = ax.barh(x - w/2, opt_waits, w, label="Optimised", color=C_OPT, alpha=0.85,
                xerr=opt_ci, capsize=3)
ax.set_yticks(x)
ax.set_yticklabels(tiers, fontsize=9)
ax.set_xlabel("Mean wait time (days)")
ax.set_title("A) DES: Cyclic FIFO vs Optimised\nper priority tier")
ax.legend()
# Add VCH benchmarks
ax.axvline(112, color="purple", lw=1.5, linestyle=":", label="VCH avg (2024): 112d")
ax.axvline(318, color="brown", lw=1.5, linestyle=":", label="VCH non-urgent (2024): 318d")
ax.legend(fontsize=8)
for bar, val in zip(bars1, cf_waits):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{val:.0f}d", va="center", fontsize=8)
for bar, val in zip(bars2, opt_waits):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{val:.0f}d", va="center", fontsize=8)

# Panel B: Improvement percentage by tier
ax = axes[1]
reductions = []
for t in tiers:
    cf_w  = _tier_m(s_cf_d, t, "mean_wait")
    opt_w = _tier_m(s_opt_d, t, "mean_wait")
    if cf_w > 0 and not math.isnan(cf_w) and not math.isnan(opt_w):
        pct = (opt_w - cf_w) / cf_w * 100
    else:
        pct = 0.0
    reductions.append(pct)

tier_colours = [TIER_COLOURS[t] for t in tiers]
bars3 = ax.barh(range(len(tiers)), reductions, color=tier_colours, alpha=0.85)
ax.set_yticks(range(len(tiers)))
ax.set_yticklabels(tiers, fontsize=9)
ax.set_xlabel("Wait time change: Optimised vs Cyclic FIFO (%)")
ax.set_title("B) Wait time improvement by tier\n(negative = shorter wait)")
ax.axvline(0, color="k", lw=0.8)
for bar, val in zip(bars3, reductions):
    offset = -2 if val < 0 else 1
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f"{val:.0f}%", va="center", ha="right" if val < 0 else "left", fontsize=9,
            fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_wait_times.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved fig4_wait_times.png")


# ===========================================================================
# Figure 5 — LP Optimisation Comparison
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 5 — LP Optimisation: Step 3 Results", fontweight="bold")

# Panel A: expected declinations by policy
ax = axes[0]
lp_policies  = ["Historical\n(actual)", "Random\n(feasible)", "Cyclic FIFO\n(VCH)", "LP Optimal"]
lp_declin    = [83.01, 94.44, 113.48, 46.43]
lp_cols      = [C_HIS, C_RND, C_CF, C_LP]
bars = ax.bar(lp_policies, lp_declin, color=lp_cols, alpha=0.85,
              edgecolor="white", linewidth=0.5)
ax.set_ylabel("Expected declinations (500 clients)")
ax.set_title("A) Expected declinations by assignment policy\n(LP snapshot of 500-client waitlist)")
for bar, val in zip(bars, lp_declin):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10)
# Annotations: % improvements
ax.annotate("−59% vs Cyclic", xy=(3, 46.43), xytext=(2.5, 80),
            arrowprops=dict(arrowstyle="->", color="darkred"),
            color="darkred", fontsize=9, fontweight="bold")

# Panel B: mean P(decline) per client by priority tier
ax = axes[1]
tier_order = ["Acute Care", "Community Emergency", "Community High", "Transfer/Site Specific"]
hist_p  = [0.2214, 0.1857, 0.1528, 0.1400]
cf_p    = [0.2453, 0.2351, 0.2120, 0.2092]
rnd_p   = [0.1925, 0.1907, 0.1847, 0.1844]
lp_p    = [0.0944, 0.0944, 0.0911, 0.0916]

x = np.arange(len(tier_order))
w = 0.2
ax.bar(x - 1.5*w, hist_p, w, label="Historical", color=C_HIS, alpha=0.85)
ax.bar(x - 0.5*w, cf_p,   w, label="Cyclic FIFO", color=C_CF, alpha=0.85)
ax.bar(x + 0.5*w, rnd_p,  w, label="Random", color=C_RND, alpha=0.85)
ax.bar(x + 1.5*w, lp_p,   w, label="LP Optimal", color=C_LP, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([t.replace(" ", "\n") for t in tier_order], fontsize=9)
ax.set_ylabel("Mean P(decline) per client")
ax.set_title("B) Mean declination probability by tier\n(LP uniformly best across all tiers)")
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig(f"{OUT}/fig5_lp_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved fig5_lp_comparison.png")


# ===========================================================================
# Figure 6 — DES Extensions: Escalation + Geo + Flu Season
# ===========================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 6 — DES Extensions: Escalation, Geography, and Flu Season", fontweight="bold")

# Panel A: Priority escalations
ax = axes[0]
esc_labels = ["Erlang-A", "Cyclic FIFO", "Optimised"]
esc_vals   = [_m(s_ea_d,"escalations"), _m(s_cf_d,"escalations"), _m(s_opt_d,"escalations")]
esc_cis    = [_ci(s_ea_d,"escalations"), _ci(s_cf_d,"escalations"), _ci(s_opt_d,"escalations")]
esc_cols   = [C_EA, C_CF, C_OPT]
bars = ax.bar(esc_labels, esc_vals, color=esc_cols, alpha=0.85, yerr=esc_cis, capsize=5)
ax.set_ylabel("Priority escalations per run")
ax.set_title("A) Priority escalations\n(clients promoted to higher tier after wait threshold)")
for bar, val in zip(bars, esc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.annotate("High: FCFS doesn't\nprioritise anyone →\nall tiers wait long",
            xy=(0, esc_vals[0]), xytext=(0.3, esc_vals[0]*0.75),
            fontsize=7.5, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# Panel B: Geographic out-of-region fraction
ax = axes[1]
geo_cf  = _m(s_cf_d,  "out_of_region_frac")
geo_opt = _m(s_opt_d, "out_of_region_frac")
geo_ea  = _m(s_ea_d,  "out_of_region_frac")

# Stacked bar: in-region vs out-of-region
width = 0.5
policies_geo = ["Erlang-A\n(FCFS)", "Cyclic FIFO", "Optimised"]
oor = [geo_ea, geo_cf, geo_opt]
ir  = [1-v for v in oor]
ax.bar(policies_geo, ir, width, label="In-region", color="#2196F3", alpha=0.8)
ax.bar(policies_geo, oor, width, bottom=ir, label="Out-of-region", color="#FF7043", alpha=0.8)
ax.set_ylabel("Fraction of placements")
ax.set_title("B) Geographic matching\n(in-region vs out-of-region placements)")
ax.set_ylim(0, 1.1)
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
for i, (oor_v, ir_v) in enumerate(zip(oor, ir)):
    ax.text(i, ir_v/2, f"{ir_v:.0%}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
    ax.text(i, ir_v + oor_v/2, f"{oor_v:.0%}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)

# Panel C: Flu season effect
ax = axes[2]
flu_cf   = _m(s_cf_d, "flu_declination_rate")
nflu_cf  = _m(s_cf_d, "nonflu_declination_rate")
flu_opt  = _m(s_opt_d,"flu_declination_rate")
nflu_opt = _m(s_opt_d,"nonflu_declination_rate")
flu_ci_cf   = _ci(s_cf_d, "flu_declination_rate")
nflu_ci_cf  = _ci(s_cf_d, "nonflu_declination_rate")
flu_ci_opt  = _ci(s_opt_d,"flu_declination_rate")
nflu_ci_opt = _ci(s_opt_d,"nonflu_declination_rate")

x = np.arange(2)
w = 0.3
ax.bar(x - w/2, [flu_cf, flu_opt],   w, label="Flu season (Nov–Jan)", color="#E53935", alpha=0.85,
       yerr=[flu_ci_cf, flu_ci_opt], capsize=4)
ax.bar(x + w/2, [nflu_cf, nflu_opt], w, label="Non-flu season", color="#1E88E5", alpha=0.85,
       yerr=[nflu_ci_cf, nflu_ci_opt], capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(["Cyclic FIFO", "Optimised"], fontsize=10)
ax.set_ylabel("Declination rate")
ax.set_title("C) Flu season vs non-flu season\ndeclination rates (DES)")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
for i, (f, nf) in enumerate([(flu_cf, nflu_cf), (flu_opt, nflu_opt)]):
    ax.text(i - w/2, f + 0.005, f"{f:.1%}", ha="center", va="bottom", fontsize=9)
    ax.text(i + w/2, nf + 0.005, f"{nf:.1%}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig6_des_extensions.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved fig6_des_extensions.png")


# ===========================================================================
# Figure 7 — Integrated Summary: Improvement Waterfall
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 7 — Integrated Summary: Policy Improvement Across All Steps", fontweight="bold")

# Panel A: Declination reduction waterfall
ax = axes[0]
steps = ["Cyclic FIFO\n(baseline)", "LP Optimal\n(Step 3)", "Optimised\n(DES, Step 5)",
         "Optimised\n(Queue, Step 4)", "Erlang-A\n(lower bound)"]
decl_abs = [
    _m(s_cf_d, "declination_rate"),
    LP_DECLIN,
    _m(s_opt_d,"declination_rate"),
    _m(s_opt_q,"declination_rate"),
    0.0,
]
bar_cols = [C_CF, C_LP, C_OPT, C_OPT, C_EA]
bars = ax.bar(steps, [d * 100 for d in decl_abs], color=bar_cols, alpha=0.85,
              edgecolor="white", linewidth=0.5)
ax.set_ylabel("Declination rate (%)")
ax.set_title("A) Declination rates across all steps\n(Cyclic FIFO = baseline to beat)")
for bar, val in zip(bars, decl_abs):
    pct_str = f"{val:.1%}"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            pct_str, ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(0, 32)

# Draw improvement arrows
cf_val = decl_abs[0] * 100
for i, (label, val) in enumerate(zip(steps[1:], decl_abs[1:])):
    if val < decl_abs[0]:
        reduction = (val - decl_abs[0]) / decl_abs[0] * 100
        # ax.annotate(f"{reduction:.0f}%", ...)

# Panel B: Overall summary heatmap
ax = axes[1]
ax.axis("off")

summary_data = [
    ["Metric",              "Cyclic FIFO", "Optimised", "Improvement"],
    ["─"*22,                "─"*12,        "─"*10,       "─"*12],
    ["Declination rate",    "26.0%",        "19.0%",     "−27%"],
    ["Bed-days wasted/run", "388",          "254",       "−34%"],
    ["Mean wait (DES)",     "216 days",     "158 days",  "−27%"],
    ["Acute Care wait",     "765 days",     "265 days",  "−65%"],
    ["Out-of-region",       "74%",          "21%",       "−72%"],
    ["─"*22,                "─"*12,        "─"*10,       "─"*12],
    ["LP declination rate", "22.7%",        "9.3%",      "−59%"],
    ["(batch optimal)",     "(cyclic)",     "(LP)",      "(upper bound)"],
    ["─"*22,                "─"*12,        "─"*10,       "─"*12],
    ["VCH benchmark",       "112d avg",     "",           ""],
    ["VCH non-urgent",      "318d avg",     "",           ""],
]

y_start = 0.97
line_h  = 0.072
for row in summary_data:
    col_positions = [0.01, 0.38, 0.60, 0.78]
    for j, (cell, xpos) in enumerate(zip(row, col_positions)):
        weight = "bold" if row[0].startswith("Metric") or row[0].startswith("─") else "normal"
        color = "#2e7d32" if "−" in str(cell) and "%" in str(cell) else (
                "#c62828" if row[0] in ["Declination rate", "Acute Care wait",
                                         "Bed-days wasted/run"] and j == 3 else "black")
        ax.text(xpos, y_start, cell, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", fontweight=weight, color=color,
                fontfamily="monospace")
    y_start -= line_h

ax.set_title("B) Quantitative results summary\n(DES = 30 reps, Queue = 10 reps)")

plt.tight_layout()
plt.savefig(f"{OUT}/fig7_summary.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved fig7_summary.png")


print(f"\nAll figures saved to {OUT}/")
print("Run: PYTHONPATH=. python results/plot_results.py")
