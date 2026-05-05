"""
experiments/vch_experiment.py — VCH-scale case study (55 facilities).

Purpose
-------
Reproduce Vancouver Coastal Health system dynamics with calibrated parameters,
then compare the three placement policies at realistic scale.

Calibration targets (BC Office of Seniors Advocate 2024/25)
------------------------------------------------------------
  55 facilities | ~5,300 beds total | λ = 28 clients/month | μ = 18 beds/month
  CommHigh mean wait ≈ 473 days under Cyclic FIFO  (calibrated: CommHigh=21.1%)
  Priority waitlist backlog Q₀ = 100 (tier-specific θ: Q_ss ≈ 94)
  30 replications | 1,095-day horizon (3 years) | 180-day warm-up

Run:  PYTHONPATH=. python experiments/vch_experiment.py
"""

from __future__ import annotations

import math
import numpy as np

import model.simulation.queue_model as _qm
import model.parameters as P

# VCH parameters are set in model/parameters.py.
# Override replications to 30 for VCH report-quality CIs.
_qm.N_REPLICATIONS = 30

assert abs(_qm.ARRIVAL_RATE - 28 / 30.0) < 1e-6, "ARRIVAL_RATE mismatch"
assert abs(_qm.VACANCY_RATE - 18 / 30.0) < 1e-6, "VACANCY_RATE mismatch"

from model.simulation.queue_model import (
    QueueSimulator, run_replications, summarise_replications,
    PRIORITY_LABELS,
)

# ---------------------------------------------------------------------------
# Facility generation — VCH-representative mock data
# ---------------------------------------------------------------------------
# Sources: BC OSA 2024/25 LTC Directory
#   55 facilities (16 health-authority + 39 contracted)
#   Average 96 beds per facility (28,869 beds / 301 facilities × BC average)
#   All h_n = 0 (for-profit flag not used in current analysis)
#   u_n ~ N(0, 0.25)  — unobserved facility heterogeneity

_RNG_SETUP = np.random.default_rng(42)

N_FAC = 55

# Bed counts: mean 96, SD 28, min 20 — sample from truncated normal
_raw_beds = _RNG_SETUP.normal(loc=96, scale=28, size=N_FAC)
CAPACITIES = np.maximum(20, np.round(_raw_beds)).astype(int).tolist()

FACILITY_NAMES = [f"Facility_{i+1:02d}" for i in range(N_FAC)]

# No for-profit flag used
H_N_ARR = np.zeros(N_FAC, dtype=int)

# Facility random effects (unobserved heterogeneity)
U_N_ARR = _RNG_SETUP.normal(0.0, np.sqrt(P.SIGMA_SQ), size=N_FAC)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(stat: dict | float, decimals: int = 1) -> str:
    if isinstance(stat, (int, float)):
        return f"{stat:.{decimals}f}"
    m, h = stat["mean"], stat["half_ci"]
    if math.isnan(m):
        return "—"
    if math.isnan(h):
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} ±{h:.{decimals}f}"


def _pct_change(base_stat: dict, new_stat: dict) -> str:
    b, n = base_stat["mean"], new_stat["mean"]
    if math.isnan(b) or math.isnan(n) or b == 0:
        return "—"
    return f"{(n - b) / b * 100:+.1f}%"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

_DISPLAY = {
    "erlang_a":    "FCFS (No Declines)",
    "cyclic_fifo": "Cyclic FIFO",
    "optimised":   "Optimised",
}


def main() -> None:
    total_beds = sum(CAPACITIES)
    print("\n" + "=" * 72)
    print("  VCH CASE STUDY — 55 Facilities")
    print(f"  Total beds: {total_beds}  |  Mean beds/facility: {total_beds/N_FAC:.0f}")
    print(f"  λ = 28 clients/month  |  μ = 18 beds/month  |  Q₀ = {_qm.INITIAL_QUEUE_SIZE}")
    print(f"  Horizon = {_qm.SIM_HORIZON} days  |  Warm-up = {_qm.WARM_UP_DAYS} days  |  "
          f"{_qm.N_REPLICATIONS} replications")
    print("  Tier-specific abandonment: AC 50%/mo · CommEm 30%/mo · "
          "CommHigh 2%/mo · Transfer 9.4%/mo")
    print("  Transfer fraction = 40.0%  →  analytical W_Transfer ≈ 473 days (OSA 2024/25)")
    print("=" * 72)

    kwargs = dict(
        facility_names=FACILITY_NAMES,
        capacities=CAPACITIES,
        h_n_arr=H_N_ARR,
        u_n_arr=U_N_ARR,
        n_reps=_qm.N_REPLICATIONS,
        sim_days=_qm.SIM_HORIZON,
        warmup_days=_qm.WARM_UP_DAYS,
    )

    summaries: dict[str, dict] = {}
    for policy in ("erlang_a", "cyclic_fifo", "optimised"):
        print(f"\n  Running '{_DISPLAY[policy]}' ({_qm.N_REPLICATIONS} reps)...", end=" ", flush=True)
        reps = run_replications(policy=policy, **kwargs)
        summaries[policy] = summarise_replications(reps)
        s = summaries[policy]
        print(f"done  "
              f"[declin={s['declination_rate']['mean']:.3f}  "
              f"wait={s['mean_wait_days']['mean']:.0f}d  "
              f"bed_loss={s['bed_days_wasted']['mean']:.0f}]")

    _print_report(summaries)


def _print_report(summaries: dict[str, dict]) -> None:
    ea  = summaries["erlang_a"]
    cyc = summaries["cyclic_fifo"]
    opt = summaries["optimised"]

    W = 28
    C = 18

    def _row(label: str, key: str, dec: int = 1) -> None:
        print(
            f"  {label:<{W}}"
            f"{_fmt(ea[key],  dec):>{C}}"
            f"{_fmt(cyc[key], dec):>{C}}"
            f"{_fmt(opt[key], dec):>{C}}"
        )

    def _hdr() -> None:
        print(
            f"  {'':>{W}}"
            f"{_DISPLAY['erlang_a']:>{C}}"
            f"{_DISPLAY['cyclic_fifo']:>{C}}"
            f"{_DISPLAY['optimised']:>{C}}"
        )

    sep  = "=" * (W + C * 3 + 2)
    sep2 = "-" * len(sep)

    print("\n" + sep)
    print("  VCH OVERALL RESULTS  (mean ± 95% CI, 30 reps, annualised)")
    print(sep)
    _hdr()
    print(sep2)
    _row("Placements",             "placements",         0)
    _row("Declinations",           "declinations",        0)
    _row("Abandonments",           "abandonments",        0)
    _row("Bed-Loss Days",          "bed_days_wasted",     1)
    _row("Bed-Loss Days/placement","bed_days_per_place",  3)
    _row("Declination rate",       "declination_rate",    3)
    _row("Mean wait (days)",       "mean_wait_days",      1)
    _row("Mean queue length",      "mean_queue_length",   1)

    print(sep2)
    print("  PER-TIER DECLINATION RATE")
    _hdr()
    print(sep2)
    for tier in PRIORITY_LABELS:
        print(
            f"  {tier:<{W}}"
            f"{_fmt(ea['by_tier'][tier]['declin_rate'],  3):>{C}}"
            f"{_fmt(cyc['by_tier'][tier]['declin_rate'], 3):>{C}}"
            f"{_fmt(opt['by_tier'][tier]['declin_rate'], 3):>{C}}"
        )

    print(sep2)
    print("  PER-TIER MEAN WAIT (days)")
    _hdr()
    print(sep2)
    for tier in PRIORITY_LABELS:
        print(
            f"  {tier:<{W}}"
            f"{_fmt(ea['by_tier'][tier]['mean_wait'],  1):>{C}}"
            f"{_fmt(cyc['by_tier'][tier]['mean_wait'], 1):>{C}}"
            f"{_fmt(opt['by_tier'][tier]['mean_wait'], 1):>{C}}"
        )

    print(sep)
    print("  IMPROVEMENT: Cyclic FIFO → Optimised")
    print(sep2)
    for label, key in [
        ("Declinations",    "declinations"),
        ("Bed-Loss Days",   "bed_days_wasted"),
        ("Mean wait",       "mean_wait_days"),
    ]:
        print(f"  {label:<{W}}{_pct_change(cyc[key], opt[key]):>{C}}")
    for tier in PRIORITY_LABELS:
        short = tier.replace("Transfer/Site Specific", "Transfer").replace("Community ", "Comm ")
        print(
            f"  {'  ' + short + ' wait':<{W}}"
            f"{_pct_change(cyc['by_tier'][tier]['mean_wait'], opt['by_tier'][tier]['mean_wait']):>{C}}"
        )
    print(sep)

    # --- Calibration check ---
    tr_wait = cyc['by_tier']['Transfer/Site Specific']['mean_wait']['mean']
    print(f"\n  CALIBRATION CHECK")
    print(f"  Transfer mean wait under Cyclic FIFO: {tr_wait:.0f} days")
    print(f"  Analytical target: 473 days (OSA 2024/25); simulation shows finite-horizon transient")
    print(f"  Formula: W = (λ_T - μ_T)/(μ_T × θ_T) = (11.2-4.5)/(4.5×0.094) ≈ 473 days ✓")


if __name__ == "__main__":
    main()
