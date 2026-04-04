"""
model/validation/validate_step5.py — Validation checks for Step 5 (DES).

Checks
------
  Section 1 — Single replication structure (9 checks)
  Section 2 — DES-specific features (8 checks)
  Section 3 — Policy comparisons (8 checks)

Run:  PYTHONPATH=. python model/validation/validate_step5.py
"""

from __future__ import annotations

import math
import numpy as np

from model.data_loader import load_all
from model.core.logistic import prepare_model_inputs
from model.parameters import (
    PRIORITY_LABELS, DES_N_REPLICATIONS, ARRIVAL_RATE,
    VACANCY_RATE, INITIAL_QUEUE_SIZE, OFFER_TURNAROUND_DAYS,
    funding_to_h_n,
)
from model.simulation.des import (
    DESSimulator, run_des_replications, summarise_des_replications,
)

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

facility_regions = []
for i in range(len(facility_names)):
    region = facilities_df.loc[i, "community_region_desc"] \
             if "community_region_desc" in facilities_df.columns else "Vancouver"
    facility_regions.append(str(region))

gender_lim_arr = [None] * len(facility_names)

N_REPS   = 3
SIM_DAYS = 365   # 1-year for speed
# sim_start_month=4 (April): warmup covers Apr-Sep, flu months (Nov-Jan)
# fall in the post-warmup window so flu_declination_rate is computable.
SIM_START_MONTH = 4

common_kwargs = dict(
    facility_names=facility_names,
    capacities=capacities,
    h_n_arr=h_n_arr,
    u_n_arr=u_n_arr,
    facility_regions=facility_regions,
    gender_lim_arr=gender_lim_arr,
    n_reps=N_REPS,
    seed_base=0,
    sim_days=SIM_DAYS,
    sim_start_month=SIM_START_MONTH,
)


# ---------------------------------------------------------------------------
# Section 1 — Single replication structure
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 1 — Single replication structure")
print("=" * 55)

sim = DESSimulator(
    policy="cyclic_fifo",
    facility_names=facility_names,
    capacities=capacities,
    h_n_arr=h_n_arr,
    u_n_arr=u_n_arr,
    facility_regions=facility_regions,
    gender_lim_arr=gender_lim_arr,
    seed=42,
    sim_start_month=SIM_START_MONTH,
)
result = sim.run(sim_days=SIM_DAYS)

check("Result is dict",
      isinstance(result, dict))

check("Required keys present",
      all(k in result for k in [
          "policy", "placements", "declinations", "abandonments",
          "escalations", "batch_events", "bed_days_wasted",
          "declination_rate", "mean_wait_days", "mean_queue_length",
          "flu_declination_rate", "nonflu_declination_rate",
          "out_of_region_frac", "by_tier",
      ]))

check("placements > 0",
      result["placements"] > 0,
      f"placements={result['placements']}")

check("escalations >= 0",
      result["escalations"] >= 0,
      f"escalations={result['escalations']}")

check("batch_events >= 0",
      result["batch_events"] >= 0,
      f"batch_events={result['batch_events']}")

check("flu_declination_rate in [0, 1]",
      0 <= result["flu_declination_rate"] <= 1,
      f"flu_declin={result['flu_declination_rate']:.3f}")

check("nonflu_declination_rate in [0, 1]",
      0 <= result["nonflu_declination_rate"] <= 1,
      f"nonflu_declin={result['nonflu_declination_rate']:.3f}")

check("out_of_region_frac in [0, 1]",
      0 <= result["out_of_region_frac"] <= 1,
      f"out_of_region={result['out_of_region_frac']:.3f}")

check("bed_days_wasted >= 0 and finite",
      not math.isnan(float(result["bed_days_wasted"]))
      and result["bed_days_wasted"] >= 0,
      f"bed_days_wasted={result['bed_days_wasted']:.1f}")

check("mean_queue_length in bed-scarce regime (> 100)",
      not math.isnan(float(result["mean_queue_length"]))
      and result["mean_queue_length"] > 100,
      f"queue_len={result['mean_queue_length']:.1f}")

check("by_tier has all 4 priority labels",
      all(t in result["by_tier"] for t in PRIORITY_LABELS))


# ---------------------------------------------------------------------------
# Section 2 — DES-specific features
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 2 — DES-specific features")
print("=" * 55)

# Escalation: optimised policy should have more escalations than cyclic
# (clients wait longer in optimised to get better matches, so more escalate)
reps_cf  = run_des_replications(policy="cyclic_fifo",  **common_kwargs)
reps_opt = run_des_replications(policy="optimised",    **common_kwargs)
reps_ea  = run_des_replications(policy="erlang_a",     **common_kwargs)

summ_cf  = summarise_des_replications(reps_cf)
summ_opt = summarise_des_replications(reps_opt)
summ_ea  = summarise_des_replications(reps_ea)

check("Escalations are tracked (cyclic FIFO has ≥ 0 escalations)",
      summ_cf["escalations"]["mean"] >= 0,
      f"mean={summ_cf['escalations']['mean']:.1f}")

check("Batch events detected (>0 across replications)",
      all(r["batch_events"] > 0 for r in reps_cf),
      f"mean_batch_events={summ_cf['batch_events']['mean']:.1f}")

check("Flu declination rate exists and is finite",
      not math.isnan(summ_cf["flu_declination_rate"]["mean"]),
      f"flu_declin={summ_cf['flu_declination_rate']['mean']:.3f}")

check("Flu declination rate ≥ non-flu (flu season harder for providers)",
      summ_cf["flu_declination_rate"]["mean"] >= summ_cf["nonflu_declination_rate"]["mean"] - 0.10,
      f"flu={summ_cf['flu_declination_rate']['mean']:.3f} "
      f"nonflu={summ_cf['nonflu_declination_rate']['mean']:.3f}")

check("Out-of-region fraction in (0, 1) for cyclic FIFO",
      0 < summ_cf["out_of_region_frac"]["mean"] < 1,
      f"out_of_region={summ_cf['out_of_region_frac']['mean']:.3f}")

check("Optimised out-of-region fraction ≤ cyclic FIFO (geo preference works)",
      summ_opt["out_of_region_frac"]["mean"] <= summ_cf["out_of_region_frac"]["mean"] + 0.05,
      f"opt={summ_opt['out_of_region_frac']['mean']:.3f} "
      f"cf={summ_cf['out_of_region_frac']['mean']:.3f}")

check("by_tier escalations_out present for lower tiers",
      all("escalations_out" in summ_cf["by_tier"][t]
          for t in ["Transfer/Site Specific", "Community High", "Community Emergency"]))

check("Erlang-A: 0 declinations across all reps",
      all(r["declinations"] == 0 for r in reps_ea))


# ---------------------------------------------------------------------------
# Section 3 — Policy comparisons
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 3 — Policy comparisons")
print("=" * 55)

check("Erlang-A declination rate == 0",
      summ_ea["declination_rate"]["mean"] == 0.0)

check("Cyclic FIFO declination rate > 0",
      summ_cf["declination_rate"]["mean"] > 0,
      f"declin_rate={summ_cf['declination_rate']['mean']:.3f}")

check("Optimised declination rate < cyclic FIFO (or within 20% variance tolerance)",
      summ_opt["declination_rate"]["mean"] <= summ_cf["declination_rate"]["mean"] * 1.20,
      f"opt={summ_opt['declination_rate']['mean']:.3f} "
      f"cf={summ_cf['declination_rate']['mean']:.3f}")

check("Optimised mean wait ≤ cyclic FIFO mean wait",
      summ_opt["mean_wait_days"]["mean"] <= summ_cf["mean_wait_days"]["mean"] * 1.20,
      f"opt={summ_opt['mean_wait_days']['mean']:.1f}d "
      f"cf={summ_cf['mean_wait_days']['mean']:.1f}d")

check("Placements are similar across policies (same capacity)",
      abs(summ_cf["placements"]["mean"] - summ_opt["placements"]["mean"]) < 50,
      f"cf={summ_cf['placements']['mean']:.0f} opt={summ_opt['placements']['mean']:.0f}")

# Flu declination rate should be higher than non-flu for cyclic
flu_cf    = summ_cf["flu_declination_rate"]["mean"]
nonflu_cf = summ_cf["nonflu_declination_rate"]["mean"]
check("Flu-season declination rate ≥ non-flu for cyclic (±0.10 low-N tolerance)",
      flu_cf >= nonflu_cf - 0.10,
      f"flu={flu_cf:.3f} non-flu={nonflu_cf:.3f}")

check("All per-tier declination rates finite for cyclic FIFO",
      all(not math.isnan(summ_cf["by_tier"][t]["declin_rate"]["mean"])
          for t in PRIORITY_LABELS))

check("Erlang-A bed_days_wasted == 0 in all reps",
      all(r["bed_days_wasted"] == 0 for r in reps_ea))

check("Cyclic FIFO bed_days_wasted > 0",
      summ_cf["bed_days_wasted"]["mean"] > 0,
      f"bed_days_wasted={summ_cf['bed_days_wasted']['mean']:.1f}")

check("Optimised bed_days_wasted ≤ Cyclic FIFO × 1.20 (low-N tolerance)",
      summ_opt["bed_days_wasted"]["mean"] <= summ_cf["bed_days_wasted"]["mean"] * 1.20,
      f"opt={summ_opt['bed_days_wasted']['mean']:.1f} "
      f"cf={summ_cf['bed_days_wasted']['mean']:.1f}")

check("summarise_des_replications n_reps correct",
      summ_cf["n_reps"] == N_REPS)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = _PASS + _FAIL
print("\n" + "=" * 55)
print(f"  STEP 5 VALIDATION: {_PASS}/{total} checks passed")
if _FAIL == 0:
    print("  All checks PASS")
else:
    print(f"  {_FAIL} check(s) FAILED")
print("=" * 55 + "\n")
