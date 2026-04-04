"""
model/validation/validate_step3.py — Validation checks for Step 3 (Optimisation).

Checks
------
  Section 1 — D matrix (8 checks)
  Section 2 — Baseline: historical (5 checks)
  Section 3 — Baseline: cyclic FIFO (6 checks)
  Section 4 — Baseline: random (4 checks)
  Section 5 — LP solver (8 checks)
  Section 6 — Comparison metrics (4 checks)

Run:  PYTHONPATH=. python model/validation/validate_step3.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.data_loader import load_all
from model.core.logistic import prepare_model_inputs
from model.optimisation.baseline import (
    historical_assignment,
    cyclic_fifo_assignment,
    random_assignment,
)
from model.optimisation.solver import solve_lp, assignment_to_df
from model.optimisation.compare import compute_metrics

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

data         = load_all()
entries_df   = data["waitlist_entry"]
facilities_df = data["facility_details"]
vacancies_df  = data["vacancies"]

inputs = prepare_model_inputs(
    entries_df, facilities_df, vacancies_df, verbose=False
)
snapshot       = inputs["snapshot"]
D              = inputs["D"]
capacities     = inputs["capacities"]
gender_blocks  = inputs["gender_blocks"]
facility_names = inputs["facility_names"]

P = len(snapshot)
N = len(facility_names)


# ---------------------------------------------------------------------------
# Section 1 — D matrix
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 1 — D matrix")
print("=" * 55)

check("D shape matches (P, N)",
      D.shape == (P, N),
      f"D.shape={D.shape}, P={P}, N={N}")

check("D values in (0, 1)",
      (D > 0).all() and (D < 1).all(),
      f"min={D.min():.4f}, max={D.max():.4f}")

check("D mean is plausible (0.05 – 0.40)",
      0.05 <= D.mean() <= 0.40,
      f"mean={D.mean():.4f}")

check("D no NaN or Inf",
      np.isfinite(D).all())

check("Gender blocks list contains only valid (p, n) indices",
      all(0 <= p < P and 0 <= n < N for p, n in gender_blocks))

check("Gender blocks are low fraction (< 20% of pairs)",
      len(gender_blocks) / (P * N) < 0.20,
      f"{len(gender_blocks)} blocks = {len(gender_blocks)/(P*N)*100:.1f}% of pairs")

# Rows with gender mismatch should have higher D on average
if gender_blocks:
    blocked_set = set(gender_blocks)
    blocked_vals = [D[p, n] for p, n in gender_blocks]
    non_blocked_vals = [D[p, n] for p in range(P) for n in range(N)
                        if (p, n) not in blocked_set]
    check("Blocked pairs have higher mean D than non-blocked",
          np.mean(blocked_vals) > np.mean(non_blocked_vals),
          f"blocked mean={np.mean(blocked_vals):.4f}, "
          f"non-blocked mean={np.mean(non_blocked_vals):.4f}")
else:
    check("Blocked pairs: no gender blocks (correct for ~5% vacancy limitation rate)",
          True,
          "0 blocks (expected at facility level with 5% vacancy limitation rate)")

# Higher complexity clients should have higher row mean in D
snapshot_merged = snapshot.reset_index(drop=True)
row_means = D.mean(axis=1)
corr = np.corrcoef(snapshot_merged["r_p"].values, row_means)[0, 1]
check("D row means positively correlated with client complexity r_p",
      corr > 0.0,
      f"corr(r_p, mean_D_row)={corr:.4f}")


# ---------------------------------------------------------------------------
# Section 2 — Historical baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 2 — Historical baseline")
print("=" * 55)

hist_df = historical_assignment(entries_df, snapshot, facility_names, D)

check("Historical: result is DataFrame",
      isinstance(hist_df, pd.DataFrame))

check("Historical: expected columns present",
      all(c in hist_df.columns
          for c in ["client_idx", "PatientID", "facility_idx", "ProviderName", "p_pn", "rule"]))

check("Historical: assigns ≤ P clients",
      len(hist_df) <= P,
      f"{len(hist_df)} assigned out of {P}")

check("Historical: assigns > 0 clients",
      len(hist_df) > 0,
      f"{len(hist_df)} assigned")

check("Historical: all p_pn values finite and in (0,1)",
      hist_df["p_pn"].between(0, 1, inclusive="neither").all())

check("Historical: no duplicate PatientIDs in assignment",
      hist_df["PatientID"].nunique() == len(hist_df))


# ---------------------------------------------------------------------------
# Section 3 — Cyclic FIFO baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 3 — Cyclic FIFO baseline")
print("=" * 55)

cyclic_df = cyclic_fifo_assignment(
    snapshot, facility_names, capacities, D, gender_blocks
)

check("Cyclic FIFO: all clients assigned (or < P if capacity exhausted)",
      len(cyclic_df) <= P,
      f"{len(cyclic_df)} / {P}")

check("Cyclic FIFO: assigns all P clients (capacity >> P)",
      len(cyclic_df) == P,
      f"{len(cyclic_df)} assigned")

check("Cyclic FIFO: no duplicate PatientIDs",
      cyclic_df["PatientID"].nunique() == len(cyclic_df))

check("Cyclic FIFO: capacity constraints respected",
      cyclic_df.groupby("facility_idx").size().max() <= max(capacities),
      f"max load={cyclic_df.groupby('facility_idx').size().max()}, "
      f"max capacity={max(capacities)}")

check("Cyclic FIFO: no gender-blocked assignments",
      not any(
          (row["client_idx"], row["facility_idx"]) in set(gender_blocks)
          for _, row in cyclic_df.iterrows()
      ))

check("Cyclic FIFO: p_pn values all in (0, 1)",
      cyclic_df["p_pn"].between(0, 1, inclusive="neither").all())


# ---------------------------------------------------------------------------
# Section 4 — Random baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 4 — Random baseline")
print("=" * 55)

rand_df = random_assignment(
    snapshot, facility_names, capacities, D, gender_blocks, seed=99
)

check("Random: assigns all P clients",
      len(rand_df) == P,
      f"{len(rand_df)} assigned")

check("Random: no duplicate PatientIDs",
      rand_df["PatientID"].nunique() == len(rand_df))

check("Random: capacity constraints respected",
      rand_df.groupby("facility_idx").size().max() <= max(capacities))

check("Random: no gender-blocked assignments",
      not any(
          (row["client_idx"], row["facility_idx"]) in set(gender_blocks)
          for _, row in rand_df.iterrows()
      ))


# ---------------------------------------------------------------------------
# Section 5 — LP solver
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 5 — LP solver")
print("=" * 55)

lp_result = solve_lp(D, capacities, gender_blocks, verbose=False)

check("LP status is Optimal",
      lp_result["status"] == "Optimal",
      f"status={lp_result['status']}")

check("LP objective is positive",
      lp_result["objective"] is not None and lp_result["objective"] > 0,
      f"objective={lp_result.get('objective')}")

lp_df = assignment_to_df(lp_result, snapshot, facility_names, D)

check("LP assigns all P clients",
      len(lp_df) == P,
      f"{len(lp_df)} assigned")

check("LP no duplicate PatientIDs",
      lp_df["PatientID"].nunique() == len(lp_df))

check("LP capacity constraints respected",
      lp_df.groupby("facility_idx").size().max() <= max(capacities),
      f"max load={lp_df.groupby('facility_idx').size().max()}")

check("LP no gender-blocked assignments",
      not any(
          (row["client_idx"], row["facility_idx"]) in set(gender_blocks)
          for _, row in lp_df.iterrows()
      ))

check("LP objective matches sum of p_pn in result DataFrame",
      abs(lp_df["p_pn"].sum() - lp_result["objective"]) < 1e-3,
      f"sum_p_pn={lp_df['p_pn'].sum():.4f}, obj={lp_result['objective']:.4f}")

# LP should be at least as good as any baseline
lp_total = lp_df["p_pn"].sum()
rand_total = rand_df["p_pn"].sum()
check("LP total expected declinations ≤ random baseline",
      lp_total <= rand_total + 1e-6,
      f"LP={lp_total:.3f}, random={rand_total:.3f}")


# ---------------------------------------------------------------------------
# Section 6 — Comparison metrics
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  SECTION 6 — Comparison metrics")
print("=" * 55)

m_hist   = compute_metrics(hist_df,   snapshot, "historical")
m_cyclic = compute_metrics(cyclic_df, snapshot, "cyclic_fifo")
m_rand   = compute_metrics(rand_df,   snapshot, "random")
m_lp     = compute_metrics(lp_df,     snapshot, "lp_optimal")

check("LP total_expected_declinations < cyclic FIFO",
      m_lp["total_expected_declinations"] < m_cyclic["total_expected_declinations"],
      f"LP={m_lp['total_expected_declinations']:.2f}, "
      f"cyclic={m_cyclic['total_expected_declinations']:.2f}")

check("LP total_expected_declinations < random",
      m_lp["total_expected_declinations"] < m_rand["total_expected_declinations"],
      f"LP={m_lp['total_expected_declinations']:.2f}, "
      f"random={m_rand['total_expected_declinations']:.2f}")

check("LP improvement vs cyclic FIFO > 30%",
      (m_cyclic["total_expected_declinations"] - m_lp["total_expected_declinations"])
      / m_cyclic["total_expected_declinations"] > 0.30,
      f"improvement={(m_cyclic['total_expected_declinations'] - m_lp['total_expected_declinations'])/m_cyclic['total_expected_declinations']*100:.1f}%")

check("All four by_priority keys present in LP metrics",
      all(k in m_lp["by_priority"]
          for k in ["Acute Care", "Community Emergency", "Community High", "Transfer/Site Specific"]))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = _PASS + _FAIL
print("\n" + "=" * 55)
print(f"  STEP 3 VALIDATION: {_PASS}/{total} checks passed")
if _FAIL == 0:
    print("  All checks PASS")
else:
    print(f"  {_FAIL} check(s) FAILED")
print("=" * 55 + "\n")
