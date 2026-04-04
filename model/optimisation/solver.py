"""
model/optimisation/solver.py — LP relaxation of the BIP assignment problem.

Formulation
-----------
  min   Σ_p Σ_n  D[p,n] · x[p,n]

  s.t.  Σ_n x[p,n]  = 1        ∀ p   (C1: each client assigned once)
        Σ_p x[p,n]  ≤ C_n      ∀ n   (C2: facility capacity)
        x[p,n]      = 0         if g_pn = 1  (C3: gender hard block)
        x[p,n]      ∈ [0, 1]               (LP relaxation — TU guarantees integer)

Why LP = BIP  (Total Unimodularity)
--------------------------------------
The constraint matrix of C1+C2 is the node-arc incidence matrix of a bipartite
graph (clients ↔ facilities). This is always totally unimodular (TU).
Adding C3 = column deletion → TU preserved.
∴  LP optimal ∈ {0,1}^(P·N)  for any integer RHS.  No branch-and-bound needed.

Constraints covered by data
---------------------------
  C1  Assignment        ✅  always applicable
  C2  Capacity          ✅  Total LTC Beds from facility_details
  C3  Gender mismatch   ✅  RoomGenderLimitation from vacancies (facility-level)
  --  Wheelchair        ⚠️  room_characteristics has WHEELCHAIR ACCESSIBLE per
                             vacancy, but no client-side "needs wheelchair" field
                             → cannot enforce without client requirements data
  --  Ceiling lift      ⚠️  same reason — no client-side flag
  --  Accessibility     ⚠️  same reason

Unimplementable constraints are documented here so they can be added the moment
VCH provides client-side care requirements data.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pulp


def solve_lp(
    D: np.ndarray,
    capacities: list[int],
    gender_blocks: list[tuple[int, int]],
    time_limit_s: int = 300,
    verbose: bool = False,
) -> dict:
    """
    Solve the LP relaxation and return the optimal assignment.

    Parameters
    ----------
    D             : (P, N) deferral probability matrix
    capacities    : list of N integer facility capacities (Total LTC Beds)
    gender_blocks : list of (p_idx, n_idx) pairs where x must be 0 (C3)
    time_limit_s  : CBC solver time limit in seconds
    verbose       : print CBC solver output if True

    Returns
    -------
    dict with keys:
        status          : "Optimal" | "Infeasible" | ...
        objective       : float — total expected declinations (minimised)
        assignment      : dict {p_idx: n_idx}
        solve_time_s    : float
        n_vars_active   : int — variables not blocked by C3
        n_gender_blocks : int — variables fixed to 0 by C3
    """
    P, N   = D.shape
    t0     = time.time()
    blocked = set(gender_blocks)

    prob = pulp.LpProblem("VCH_LTC_Matching", pulp.LpMinimize)

    # --- Decision variables (skip blocked pairs — C3 enforced implicitly) ---
    active_pairs = [
        (p, n)
        for p in range(P)
        for n in range(N)
        if (p, n) not in blocked
    ]

    x = {
        (p, n): pulp.LpVariable(f"x_{p}_{n}", lowBound=0, upBound=1, cat="Continuous")
        for (p, n) in active_pairs
    }

    # --- Objective: min Σ D[p,n] · x[p,n] ---
    prob += pulp.lpSum(float(D[p, n]) * x[p, n] for (p, n) in active_pairs)

    # --- C1: each client assigned to exactly one facility ---
    for p in range(P):
        active_n = [n for n in range(N) if (p, n) not in blocked]
        if active_n:
            prob += (
                pulp.lpSum(x[p, n] for n in active_n) == 1,
                f"assign_client_{p}",
            )
        # If ALL facilities blocked for client p (extreme edge case): infeasible

    # --- C2: facility capacity not exceeded ---
    for n in range(N):
        active_p = [p for p in range(P) if (p, n) not in blocked]
        if active_p:
            prob += (
                pulp.lpSum(x[p, n] for p in active_p) <= capacities[n],
                f"cap_fac_{n}",
            )

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit_s)
    prob.solve(solver)

    solve_time = time.time() - t0
    status     = pulp.LpStatus[prob.status]

    if prob.status != 1:   # 1 = Optimal
        return {
            "status": status, "objective": None, "assignment": {},
            "solve_time_s": solve_time, "n_vars_active": len(active_pairs),
            "n_gender_blocks": len(gender_blocks),
        }

    # --- Extract assignment ---
    assignment: dict[int, int] = {}
    for p in range(P):
        best_n, best_val = -1, -1.0
        for n in range(N):
            if (p, n) not in blocked:
                val = x[p, n].value() or 0.0
                if val > best_val:
                    best_val, best_n = val, n
        if best_n >= 0:
            assignment[p] = best_n

    return {
        "status":          status,
        "objective":       float(pulp.value(prob.objective)),
        "assignment":      assignment,
        "solve_time_s":    solve_time,
        "n_vars_active":   len(active_pairs),
        "n_gender_blocks": len(gender_blocks),
    }


def assignment_to_df(
    result: dict,
    snapshot: pd.DataFrame,
    facility_names: list[str],
    D: np.ndarray,
) -> pd.DataFrame:
    """Convert solver result dict → tidy DataFrame (same format as baselines)."""
    rows = []
    for p_idx, n_idx in result["assignment"].items():
        rows.append({
            "client_idx":   p_idx,
            "PatientID":    snapshot.loc[p_idx, "PatientID"],
            "facility_idx": n_idx,
            "ProviderName": facility_names[n_idx],
            "p_pn":         float(D[p_idx, n_idx]),
            "rule":         "lp_optimal",
        })
    return pd.DataFrame(rows)
