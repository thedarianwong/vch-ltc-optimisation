"""
experiments/des_experiment.py — VCH-scale DES case study (55 facilities).

Extends vch_experiment.py with the four DES additions:
  1. Priority escalation  (CE → AC after 45 days; TS → CH after 90 days)
  2. Batch arrivals       (10% of events bring 2–4 clients)
  3. Flu-season calendar  (Nov–Jan deferral adjustment)
  4. Geographic penalty   (γ = 0.25 soft penalty for out-of-region offers)

Uses the identical seed-42 facility setup as vch_experiment.py so results
are directly comparable. Facility regions are drawn from the VCH community
distribution: Vancouver 45%, Richmond 20%, North Shore 20%, Coast 15%.

Calibration targets (BC Office of Seniors Advocate 2024/25)
------------------------------------------------------------
  55 facilities | ~5,300 beds | λ = 28/month | μ = 18/month
  30 replications | 1,095-day horizon (3 years) | 180-day warm-up

Run:  PYTHONPATH=. python experiments/des_experiment.py
"""

from __future__ import annotations

import math
import numpy as np

import model.parameters as P
from model.simulation.des import (
    run_des_replications,
    summarise_des_replications,
    print_des_report,
)

# ---------------------------------------------------------------------------
# Facility generation — identical seed-42 setup as vch_experiment.py
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

N_FAC = 55

# Bed counts: mean 96, SD 28, min 20
_raw_beds  = _RNG.normal(loc=96, scale=28, size=N_FAC)
CAPACITIES = np.maximum(20, np.round(_raw_beds)).astype(int).tolist()

FACILITY_NAMES = [f"Facility_{i+1:02d}" for i in range(N_FAC)]

# No for-profit flag (consistent with vch_experiment.py and paper)
H_N_ARR = np.zeros(N_FAC, dtype=int)

# Facility random effects — same draw order as vch_experiment.py
U_N_ARR = _RNG.normal(0.0, np.sqrt(P.SIGMA_SQ), size=N_FAC)

# Facility regions — VCH community distribution (OSA 2024/25)
# Vancouver 45% | Richmond 20% | North Shore 20% | Coast 15%
FACILITY_REGIONS = list(
    _RNG.choice(P.REGIONS, size=N_FAC, p=[0.45, 0.20, 0.20, 0.15])
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

_DISPLAY = {
    "erlang_a":    "Full Acceptance",
    "cyclic_fifo": "Cyclic FIFO",
    "optimised":   "Optimised",
}


def main() -> None:
    total_beds = sum(CAPACITIES)
    region_counts = {r: FACILITY_REGIONS.count(r) for r in P.REGIONS}

    print("\n" + "=" * 72)
    print("  VCH DES CASE STUDY — 55 Facilities")
    print(f"  Total beds: {total_beds}  |  Mean beds/facility: {total_beds / N_FAC:.0f}")
    print(f"  λ = 28 clients/month  |  μ = 18 beds/month  |  Q₀ = {P.INITIAL_QUEUE_SIZE}")
    print(f"  Horizon = {P.SIM_HORIZON} days  |  Warm-up = {P.WARM_UP_DAYS} days  |  "
          f"{P.DES_N_REPLICATIONS} replications")
    print(f"  Regions: " + "  |  ".join(f"{r}: {n}" for r, n in region_counts.items()))
    print("  DES extensions: escalation · batch arrivals · flu calendar · geo penalty")
    print("=" * 72)

    kwargs = dict(
        facility_names=FACILITY_NAMES,
        capacities=CAPACITIES,
        h_n_arr=H_N_ARR,
        u_n_arr=U_N_ARR,
        facility_regions=FACILITY_REGIONS,
        n_reps=P.DES_N_REPLICATIONS,
        seed_base=0,
        sim_days=P.SIM_HORIZON,
        warmup_days=P.WARM_UP_DAYS,
    )

    summaries: dict[str, dict] = {}
    for policy in ("erlang_a", "cyclic_fifo", "optimised"):
        print(f"\n  Running '{_DISPLAY[policy]}' ({P.DES_N_REPLICATIONS} reps)...",
              end=" ", flush=True)
        reps = run_des_replications(policy=policy, **kwargs)
        summaries[policy] = summarise_des_replications(reps)
        s = summaries[policy]
        print(f"done  "
              f"[declin={s['declination_rate']['mean']:.3f}  "
              f"wait={s['mean_wait_days']['mean']:.0f}d  "
              f"out-of-region={s['out_of_region_frac']['mean']:.2f}]")

    print_des_report(summaries)

    # --- Summary of improvements ---
    cf  = summaries["cyclic_fifo"]
    opt = summaries["optimised"]

    def _pct(base_key: str, new_key: str | None = None) -> str:
        nk = new_key or base_key
        b = cf[base_key]["mean"]
        n = opt[nk]["mean"]
        if b == 0:
            return "—"
        return f"{(n - b) / b * 100:+.1f}%"

    print("\n  IMPROVEMENT: Cyclic FIFO → Optimised")
    print("  " + "-" * 50)
    for label, key in [
        ("Declination rate",  "declination_rate"),
        ("Bed-loss days",     "bed_days_wasted"),
        ("Mean wait",         "mean_wait_days"),
    ]:
        print(f"  {label:<30}{_pct(key):>10}")
    for tier in P.PRIORITY_LABELS:
        short = (tier.replace("Transfer/Site Specific", "Transfer")
                     .replace("Community ", "Comm "))
        b = cf["by_tier"][tier]["mean_wait"]["mean"]
        n = opt["by_tier"][tier]["mean_wait"]["mean"]
        pct = f"{(n - b) / b * 100:+.1f}%" if b != 0 else "—"
        print(f"  {'  ' + short + ' wait':<30}{pct:>10}")

    out_cf  = cf["out_of_region_frac"]["mean"]
    out_opt = opt["out_of_region_frac"]["mean"]
    print(f"\n  Out-of-region:  {out_cf:.0%} → {out_opt:.0%}  "
          f"(in-region: {1-out_cf:.0%} → {1-out_opt:.0%})")


if __name__ == "__main__":
    main()
