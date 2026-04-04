"""
model/optimisation/baseline.py — Baseline assignment policies.

Two baselines to compare against the LP optimum:

  1. historical  — what actually happened in the data (Accepted offers)
  2. cyclic_fifo — simulate VCH's cycling discipline on the client snapshot
  3. random      — random feasible assignment (respects capacity + gender)

All three return the same result format:
    DataFrame with columns:
        client_idx    : row index into snapshot
        PatientID     : client ID
        facility_idx  : column index into D / facilities
        ProviderName  : facility name
        p_pn          : D[client_idx, facility_idx]
        rule          : which rule placed this client
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque

from model.parameters import PRIORITY_LABELS


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _make_result(
    assignments: list[tuple[int, int, str]],   # (p_idx, n_idx, rule)
    snapshot: pd.DataFrame,
    facility_names: list[str],
    D: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for p_idx, n_idx, rule in assignments:
        rows.append({
            "client_idx":   p_idx,
            "PatientID":    snapshot.loc[p_idx, "PatientID"],
            "facility_idx": n_idx,
            "ProviderName": facility_names[n_idx],
            "p_pn":         float(D[p_idx, n_idx]),
            "rule":         rule,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline 1: Historical assignment
# ---------------------------------------------------------------------------

def historical_assignment(
    entries_df: pd.DataFrame,
    snapshot: pd.DataFrame,
    facility_names: list[str],
    D: np.ndarray,
) -> pd.DataFrame:
    """
    Extract what actually happened: for each client in snapshot, find their
    most recent Accepted offer and record that facility.

    Clients with no Accepted row are marked as unassigned (excluded from totals).
    """
    accepted = entries_df[entries_df["WaitlistOfferOutcome"] == "Accepted"].copy()
    # Most recent acceptance per patient
    accepted = (
        accepted.sort_values("DateOfReply", ascending=False)
        .drop_duplicates(subset="PatientID", keep="first")
        [["PatientID", "ProviderName"]]
    )
    hist_map = dict(zip(accepted["PatientID"], accepted["ProviderName"]))

    fac_idx_map = {name: idx for idx, name in enumerate(facility_names)}

    assignments = []
    for p_idx, client in snapshot.iterrows():
        pid      = client["PatientID"]
        provider = hist_map.get(pid)
        if provider and provider in fac_idx_map:
            n_idx = fac_idx_map[provider]
            assignments.append((p_idx, n_idx, "historical"))
        # Unassigned clients are excluded

    return _make_result(assignments, snapshot, facility_names, D)


# ---------------------------------------------------------------------------
# Baseline 2: Cyclic FIFO  (VCH's current cycling discipline)
# ---------------------------------------------------------------------------

def cyclic_fifo_assignment(
    snapshot: pd.DataFrame,
    facility_names: list[str],
    capacities: list[int],
    D: np.ndarray,
    gender_blocks: list[tuple[int, int]],
    cycling_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Simulate VCH's current cycling discipline on the client snapshot.

    Cycling order: Acute Care → Community Emergency → Community High →
                   Transfer/Site Specific → repeat.
    Within each tier: FIFO (order of appearance in snapshot = order on waitlist).
    Facility selection: first facility with remaining capacity that passes
                        gender constraint. No p_pn optimisation.

    Parameters
    ----------
    cycling_order : tier labels in rotation order (default: PRIORITY_LABELS reversed,
                    i.e. high → low priority, matching VCH's stated cycling)
    """
    if cycling_order is None:
        cycling_order = list(reversed(PRIORITY_LABELS))
        # Acute Care first: ["Acute Care", "Community Emergency",
        #                    "Community High", "Transfer/Site Specific"]

    # Build per-tier queues (FIFO order = snapshot order)
    tier_queues: dict[str, deque] = {tier: deque() for tier in cycling_order}
    for p_idx, client in snapshot.iterrows():
        tier = client["WaitlistPriority"]
        if tier in tier_queues:
            tier_queues[tier].append(p_idx)

    # Gender block set for O(1) lookup
    blocked = set(gender_blocks)

    remaining = list(capacities)     # mutable capacity per facility
    assignments: list[tuple[int, int, str]] = []
    placed: set[int] = set()

    cycle_pos = 0
    max_rounds = len(snapshot) * 10   # safety cap

    for _ in range(max_rounds):
        # Pick current tier in rotation
        tier = cycling_order[cycle_pos % len(cycling_order)]
        cycle_pos += 1

        # Skip empty tiers (move to next)
        queue = tier_queues[tier]
        while not queue:
            tier = cycling_order[cycle_pos % len(cycling_order)]
            cycle_pos += 1
            queue = tier_queues[tier]
            if all(len(q) == 0 for q in tier_queues.values()):
                break
        if all(len(q) == 0 for q in tier_queues.values()):
            break

        # Take the FIFO client from this tier
        p_idx = queue.popleft()
        if p_idx in placed:
            continue

        # Find first facility with capacity that passes gender check
        assigned = False
        for n_idx in range(len(facility_names)):
            if remaining[n_idx] > 0 and (p_idx, n_idx) not in blocked:
                remaining[n_idx] -= 1
                assignments.append((p_idx, n_idx, f"cyclic_fifo:{tier}"))
                placed.add(p_idx)
                assigned = True
                break

        if not assigned:
            # No valid facility — client unassigned (rare given large capacity)
            pass

    return _make_result(assignments, snapshot, facility_names, D)


# ---------------------------------------------------------------------------
# Baseline 3: Random feasible assignment
# ---------------------------------------------------------------------------

def random_assignment(
    snapshot: pd.DataFrame,
    facility_names: list[str],
    capacities: list[int],
    D: np.ndarray,
    gender_blocks: list[tuple[int, int]],
    seed: int = 99,
) -> pd.DataFrame:
    """
    Randomly assign each client to a facility, respecting capacity and gender.
    Clients are shuffled before assignment to remove any ordering bias.

    Provides a lower-effort baseline: better than the worst possible
    assignment, but no optimisation of p_pn.
    """
    rng     = np.random.default_rng(seed)
    blocked = set(gender_blocks)

    remaining   = list(capacities)
    client_order = list(snapshot.index)
    rng.shuffle(client_order)

    assignments: list[tuple[int, int, str]] = []

    for p_idx in client_order:
        # Eligible facilities: have capacity + no gender block
        eligible = [
            n for n in range(len(facility_names))
            if remaining[n] > 0 and (p_idx, n) not in blocked
        ]
        if not eligible:
            continue
        n_idx = int(rng.choice(eligible))
        remaining[n_idx] -= 1
        assignments.append((p_idx, n_idx, "random"))

    return _make_result(assignments, snapshot, facility_names, D)
