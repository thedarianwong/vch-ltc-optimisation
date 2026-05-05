"""
experiments/small_experiment.py — Small-scale controlled experiment (2 facilities).

Purpose
-------
Demonstrate the model mechanics at a transparent, human-readable scale before
applying VCH-calibrated parameters.  All assumptions are explicit; results can
be hand-verified.

Setup
-----
  2 facilities  |  22 beds total  |  λ = 3 clients/month  |  μ = 2 beds/month
  Overload ratio: (3 - 2) / 3 ≈ 33%  — intentionally overloaded to show policy differences
  Tier-specific abandonment rates (same as VCH calibration)
  30 replications  |  730-day horizon  |  90-day warm-up

Run:  PYTHONPATH=. python experiments/small_experiment.py
"""

from __future__ import annotations

import math
import numpy as np

# ---------------------------------------------------------------------------
# Patch module-level parameters BEFORE importing the simulator
# (the simulator reads these names from its own module namespace)
# ---------------------------------------------------------------------------
import model.simulation.queue_model as _qm
import model.parameters as P

_SMALL = {
    "ARRIVAL_RATE":       3.0  / 30.0,   # 3 new clients/month
    "VACANCY_RATE":       2.0  / 30.0,   # 2 bed openings/month accessible from waitlist
    "INITIAL_QUEUE_SIZE": 10,             # small starting backlog
    "SIM_HORIZON":        730,            # 2-year horizon
    "WARM_UP_DAYS":       90,             # 3-month warm-up
    "N_REPLICATIONS":     30,
    "MAX_OFFERS":         15,
    "OFFER_TURNAROUND_DAYS": 2.0,
}
for _k, _v in _SMALL.items():
    setattr(_qm, _k, _v)

# Now import the runner (reads patched module globals at call time)
from model.simulation.queue_model import (
    QueueSimulator, run_replications, summarise_replications,
    PRIORITY_LABELS,
)

# ---------------------------------------------------------------------------
# Facility definitions (hand-crafted, representative)
# ---------------------------------------------------------------------------

FACILITY_NAMES = ["Cedar Grove", "Maple House"]
CAPACITIES     = [10, 12]          # 10 and 12 beds
H_N_ARR        = np.array([0, 0])  # for-profit flag not used; both set to 0
U_N_ARR        = np.array([-0.10, +0.15])   # Cedar Grove below avg deferral; Maple House above

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
    print("\n" + "=" * 68)
    print("  SMALL EXPERIMENT — 2 Facilities, 22 Beds")
    print("  Facilities: Cedar Grove (10 beds, u_n = -0.10, low deferral propensity)")
    print("              Maple House (12 beds, u_n = +0.15, high deferral propensity)")
    print(f"  λ = 3 clients/month  |  μ = 2 beds/month  |  Q₀ = {_SMALL['INITIAL_QUEUE_SIZE']}")
    print(f"  Horizon = {_SMALL['SIM_HORIZON']} days  |  Warm-up = {_SMALL['WARM_UP_DAYS']} days  |  "
          f"{_SMALL['N_REPLICATIONS']} replications")
    print("  Tier-specific abandonment: AC 50%/mo · CommEm 30%/mo · "
          "CommHigh 2%/mo · Transfer 15%/mo")
    print("=" * 68)

    kwargs = dict(
        facility_names=FACILITY_NAMES,
        capacities=CAPACITIES,
        h_n_arr=H_N_ARR,
        u_n_arr=U_N_ARR,
        n_reps=_SMALL["N_REPLICATIONS"],
        sim_days=_SMALL["SIM_HORIZON"],
        warmup_days=_SMALL["WARM_UP_DAYS"],
    )

    summaries: dict[str, dict] = {}
    for policy in ("erlang_a", "cyclic_fifo", "optimised"):
        print(f"\n  Running '{_DISPLAY[policy]}' ({_SMALL['N_REPLICATIONS']} reps)...", end=" ", flush=True)
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

    W = 28   # label width
    C = 20   # column width

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
    print("  OVERALL RESULTS  (mean ± 95% CI, 30 reps)")
    print(sep)
    _hdr()
    print(sep2)
    _row("Placements",            "placements",         0)
    _row("Declinations",          "declinations",        0)
    _row("Abandonments",          "abandonments",        0)
    _row("Bed-Loss Days",         "bed_days_wasted",     1)
    _row("Bed-Loss Days/placement","bed_days_per_place",  3)
    _row("Declination rate",      "declination_rate",    3)
    _row("Mean wait (days)",      "mean_wait_days",      1)
    _row("Mean queue length",     "mean_queue_length",   1)

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


if __name__ == "__main__":
    main()
