"""
model/validation/validate_step4.py — Validation checks for Step 4 (Queue Model).

Checks
------
  Section 1 — Single replication structure (9 checks)
  Section 2 — Erlang-A lower bound (5 checks)
  Section 3 — Cyclic FIFO policy (5 checks)
  Section 4 — Optimised policy (6 checks)
  Section 5 — Policy comparisons (7 checks)

Run:  PYTHONPATH=. python model/validation/validate_step4.py
"""

from __future__ import annotations

import math
import numpy as np

from model.data_loader import load_all
from model.core.logistic import prepare_model_inputs
from model.parameters import (
    PRIORITY_LABELS, ARRIVAL_RATE, VACANCY_RATE,
    INITIAL_QUEUE_SIZE, OFFER_TURNAROUND_DAYS,
    SIM_HORIZON, WARM_UP_DAYS,
)
from model.simulation.queue_model import (
    QueueSimulator, run_replications, summarise_replications,
    run_queue_comparison,
)
from model.parameters import funding_to_h_n

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        _PASS += 1
    else:
        _FAIL += 1


# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------

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

h_n_arr = np.array([
    funding_to_h_n(facilities_df.loc[i, "funding"])
    for i in range(len(facility_names))
])
u_n_arr = np.array([u_n_map.get(name, 0.0) for name in facility_names])
gender_lim_arr = [None] * len(facility_names)

N_REPS  = 3    # use fewer reps for speed in validation
SIM_DAYS = 365  # 1 year for speed


# ---------------------------------------------------------------------------
# Section 1 — Single replication structure
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 1 — Single replication structure")
print("=" * 55)

sim = QueueSimulator(
    policy="cyclic_fifo",
    facility_names=facility_names,
    capacities=capacities,
    h_n_arr=h_n_arr,
    u_n_arr=u_n_arr,
    gender_lim_arr=gender_lim_arr,
    seed=42,
)
result = sim.run(sim_days=SIM_DAYS, warmup_days=WARM_UP_DAYS)

check("Result is dict",
      isinstance(result, dict))

check("Required keys present",
      all(k in result for k in [
          "policy", "placements", "declinations", "abandonments",
          "bed_days_wasted", "bed_days_per_place",
          "declination_rate", "mean_wait_days", "mean_queue_length", "by_tier",
      ]))

check("placements > 0",
      result["placements"] > 0,
      f"placements={result['placements']}")

check("declination_rate in [0, 1]",
      0 <= result["declination_rate"] <= 1,
      f"declin_rate={result['declination_rate']:.4f}")

check("mean_wait_days >= 0",
      result["mean_wait_days"] >= 0,
      f"wait={result['mean_wait_days']:.2f}")

check("mean_queue_length >= 0 and finite",
      not math.isnan(result["mean_queue_length"])
      and result["mean_queue_length"] >= 0,
      f"queue_len={result['mean_queue_length']:.1f}")

check("bed_days_wasted >= 0",
      result["bed_days_wasted"] >= 0,
      f"bed_days_wasted={result['bed_days_wasted']:.1f}")

check("by_tier has all 4 priority labels",
      all(t in result["by_tier"] for t in PRIORITY_LABELS))

check("by_tier sums equal totals (placements)",
      abs(sum(result["by_tier"][t]["placements"] for t in PRIORITY_LABELS)
          - result["placements"]) <= 1)


# ---------------------------------------------------------------------------
# Section 2 — Erlang-A lower bound
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 2 — Erlang-A lower bound")
print("=" * 55)

reps_ea = run_replications(
    policy="erlang_a",
    facility_names=facility_names, capacities=capacities,
    h_n_arr=h_n_arr, u_n_arr=u_n_arr, gender_lim_arr=gender_lim_arr,
    n_reps=N_REPS, seed_base=0, sim_days=SIM_DAYS, warmup_days=WARM_UP_DAYS,
)
summ_ea = summarise_replications(reps_ea)

check("Erlang-A: 0 declinations in every replication",
      all(r["declinations"] == 0 for r in reps_ea))

check("Erlang-A: declination_rate == 0",
      summ_ea["declination_rate"]["mean"] == 0.0,
      f"declin_rate={summ_ea['declination_rate']['mean']}")

check("Erlang-A: bed_days_wasted == 0",
      summ_ea["bed_days_wasted"]["mean"] == 0.0,
      f"bed_days_wasted={summ_ea['bed_days_wasted']['mean']}")

check("Erlang-A: mean placements > 0",
      summ_ea["placements"]["mean"] > 0,
      f"placements={summ_ea['placements']['mean']:.0f}")

check("Erlang-A: has finite mean wait",
      not math.isnan(summ_ea["mean_wait_days"]["mean"]),
      f"wait={summ_ea['mean_wait_days']['mean']:.2f}")


# ---------------------------------------------------------------------------
# Section 3 — Cyclic FIFO policy
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 3 — Cyclic FIFO policy")
print("=" * 55)

reps_cf = run_replications(
    policy="cyclic_fifo",
    facility_names=facility_names, capacities=capacities,
    h_n_arr=h_n_arr, u_n_arr=u_n_arr, gender_lim_arr=gender_lim_arr,
    n_reps=N_REPS, seed_base=0, sim_days=SIM_DAYS, warmup_days=WARM_UP_DAYS,
)
summ_cf = summarise_replications(reps_cf)

check("Cyclic FIFO: declination_rate > 0",
      summ_cf["declination_rate"]["mean"] > 0,
      f"declin_rate={summ_cf['declination_rate']['mean']:.3f}")

check("Cyclic FIFO: declination_rate < 0.5 (sanity upper bound)",
      summ_cf["declination_rate"]["mean"] < 0.5)

check("Cyclic FIFO: bed_days_wasted > 0",
      summ_cf["bed_days_wasted"]["mean"] > 0,
      f"bed_days_wasted={summ_cf['bed_days_wasted']['mean']:.1f}")

check("Cyclic FIFO: placements > 0",
      summ_cf["placements"]["mean"] > 0)

check("Cyclic FIFO: all priority tiers get some placements",
      all(summ_cf["by_tier"][t]["placements"]["mean"] > 0
          for t in PRIORITY_LABELS))


# ---------------------------------------------------------------------------
# Section 4 — Optimised policy
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 4 — Optimised policy")
print("=" * 55)

reps_opt = run_replications(
    policy="optimised",
    facility_names=facility_names, capacities=capacities,
    h_n_arr=h_n_arr, u_n_arr=u_n_arr, gender_lim_arr=gender_lim_arr,
    n_reps=N_REPS, seed_base=0, sim_days=SIM_DAYS, warmup_days=WARM_UP_DAYS,
)
summ_opt = summarise_replications(reps_opt)

check("Optimised: declination_rate > 0 (declinations happen in practice)",
      summ_opt["declination_rate"]["mean"] > 0)

check("Optimised: declination_rate < 0.5",
      summ_opt["declination_rate"]["mean"] < 0.5)

check("Optimised: placements > 0",
      summ_opt["placements"]["mean"] > 0)

check("Optimised: all priority tiers get some placements",
      all(summ_opt["by_tier"][t]["placements"]["mean"] > 0
          for t in PRIORITY_LABELS))

check("Optimised: mean_wait_days >= 0 and finite",
      summ_opt["mean_wait_days"]["mean"] >= 0
      and not math.isnan(summ_opt["mean_wait_days"]["mean"]),
      f"wait={summ_opt['mean_wait_days']['mean']:.2f}")

check("Optimised: bed_days_wasted > 0 (some offers still declined)",
      summ_opt["bed_days_wasted"]["mean"] > 0,
      f"bed_days_wasted={summ_opt['bed_days_wasted']['mean']:.1f}")


# ---------------------------------------------------------------------------
# Section 5 — Policy comparisons
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 5 — Policy comparisons")
print("=" * 55)

check("Erlang-A declination_rate < Cyclic FIFO",
      summ_ea["declination_rate"]["mean"] < summ_cf["declination_rate"]["mean"],
      f"ea={summ_ea['declination_rate']['mean']:.3f} < "
      f"cf={summ_cf['declination_rate']['mean']:.3f}")

check("Erlang-A declination_rate < Optimised",
      summ_ea["declination_rate"]["mean"] < summ_opt["declination_rate"]["mean"],
      f"ea={summ_ea['declination_rate']['mean']:.3f} < "
      f"opt={summ_opt['declination_rate']['mean']:.3f}")

# Note: with only N_REPS=3, variance is high; allow 20% overshoot
check("Optimised declination_rate within 20% of Cyclic FIFO (low-N tolerance)",
      summ_opt["declination_rate"]["mean"] <= summ_cf["declination_rate"]["mean"] * 1.20,
      f"opt={summ_opt['declination_rate']['mean']:.3f} ≤ "
      f"cf×1.2={summ_cf['declination_rate']['mean']*1.20:.3f}")

# Bed-days wasted: erlang_a=0, optimised < cyclic (allow 20% tolerance with N_REPS=3)
check("Erlang-A bed_days_wasted == 0",
      summ_ea["bed_days_wasted"]["mean"] == 0.0)

check("Optimised bed_days_wasted ≤ Cyclic FIFO × 1.20 (low-N tolerance)",
      summ_opt["bed_days_wasted"]["mean"] <= summ_cf["bed_days_wasted"]["mean"] * 1.20,
      f"opt={summ_opt['bed_days_wasted']['mean']:.1f} "
      f"cf={summ_cf['bed_days_wasted']['mean']:.1f}")

check("summarise_replications returns n_reps correctly",
      summ_cf["n_reps"] == N_REPS)

# Calibration check: queue should be large (VCH scenario — vacancy < arrival)
# Steady-state: Q_ss = (arrival - vacancy) / abandon ≈ 500
check("Mean queue length in bed-scarce regime (> 100 clients post warmup)",
      summ_cf["mean_queue_length"]["mean"] > 100,
      f"queue_len={summ_cf['mean_queue_length']['mean']:.0f}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = _PASS + _FAIL
print("\n" + "=" * 55)
print(f"  STEP 4 VALIDATION: {_PASS}/{total} checks passed")
if _FAIL == 0:
    print("  All checks PASS")
else:
    print(f"  {_FAIL} check(s) FAILED")
print("=" * 55 + "\n")
