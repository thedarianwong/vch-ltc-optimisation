"""
model/optimisation/compare.py — Run all policies and produce comparison report.

Policies compared
-----------------
  historical   — what the mock data shows actually happened
  cyclic_fifo  — VCH's current cycling discipline (Acute→CommEm→CommHigh→Transfer)
  random       — random feasible assignment (capacity + gender respected)
  lp_optimal   — LP-optimised assignment (minimum total expected declinations)

Run:  python model/optimisation/compare.py
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    assign_df: pd.DataFrame,
    snapshot: pd.DataFrame,
    label: str,
) -> dict:
    """Compute summary metrics from an assignment DataFrame."""
    n_clients      = len(snapshot)
    n_assigned     = len(assign_df)
    unassigned     = n_clients - n_assigned

    total_p_pn     = assign_df["p_pn"].sum()
    mean_p_pn      = assign_df["p_pn"].mean() if n_assigned > 0 else float("nan")

    # Per-priority metrics
    merged = assign_df.merge(
        snapshot[["PatientID", "WaitlistPriority"]],
        on="PatientID", how="left",
    )
    by_priority = {}
    for tier in ["Acute Care", "Community Emergency", "Community High", "Transfer/Site Specific"]:
        sub = merged[merged["WaitlistPriority"] == tier]
        by_priority[tier] = {
            "n":       len(sub),
            "mean_p":  sub["p_pn"].mean() if len(sub) > 0 else float("nan"),
        }

    # Facility utilisation
    fac_counts = assign_df["ProviderName"].value_counts()

    return {
        "label":          label,
        "n_clients":      n_clients,
        "n_assigned":     n_assigned,
        "n_unassigned":   unassigned,
        "total_expected_declinations": round(total_p_pn, 3),
        "mean_p_pn":      round(mean_p_pn, 4),
        "n_facilities_used": assign_df["ProviderName"].nunique(),
        "max_facility_load": int(fac_counts.max()) if len(fac_counts) > 0 else 0,
        "by_priority":    by_priority,
    }


def print_report(results: dict[str, dict]) -> None:
    """Print side-by-side comparison table."""
    policies = list(results.keys())
    W_label  = 30   # label column width
    W_col    = 13   # data column width

    sep  = "=" * (W_label + W_col * len(policies) + 2)
    sep2 = "-" * len(sep)

    def _fmt(v, fmt):
        try:
            if isinstance(v, float) and np.isnan(v):
                return "—"
            return format(v, fmt)
        except (ValueError, TypeError):
            return "—"

    def _header_row(labels):
        row = "  " + "".ljust(W_label)
        for lbl in labels:
            row += lbl.rjust(W_col)
        print(row)

    def _data_row(label, values, fmt):
        row = "  " + label.ljust(W_label)
        for v in values:
            row += _fmt(v, fmt).rjust(W_col)
        print(row)

    print("\n" + sep)
    print("  VCH LTC ASSIGNMENT — POLICY COMPARISON")
    print(sep)

    _header_row(policies)
    print(sep2)

    rows_spec = [
        ("Clients assigned",           "n_assigned",                  ".0f"),
        ("Clients unassigned",          "n_unassigned",                ".0f"),
        ("Total expected declinations", "total_expected_declinations", ".2f"),
        ("Mean P(decline) per client",  "mean_p_pn",                  ".4f"),
        ("Facilities used",             "n_facilities_used",           ".0f"),
        ("Max clients/facility",        "max_facility_load",           ".0f"),
    ]
    for label, key, fmt in rows_spec:
        vals = [results[p].get(key, float("nan")) for p in policies]
        _data_row(label, vals, fmt)

    print(sep2)

    # Improvement vs cyclic_fifo and random
    if "cyclic_fifo" in results and "lp_optimal" in results:
        base = results["cyclic_fifo"]["total_expected_declinations"]
        opt  = results["lp_optimal"]["total_expected_declinations"]
        if base and base > 0:
            impr = (opt - base) / base * 100   # negative = improvement
            print(f"  LP vs cyclic FIFO:  {base:.2f} → {opt:.2f}  ({impr:+.2f}%)")

    if "random" in results and "lp_optimal" in results:
        base = results["random"]["total_expected_declinations"]
        opt  = results["lp_optimal"]["total_expected_declinations"]
        if base and base > 0:
            impr = (opt - base) / base * 100
            print(f"  LP vs random:       {base:.2f} → {opt:.2f}  ({impr:+.2f}%)")

    print(sep2)
    print("  Per-priority mean P(decline):")
    _header_row(policies)
    print(sep2)

    for tier in ["Acute Care", "Community Emergency", "Community High", "Transfer/Site Specific"]:
        vals = [
            results[p]["by_priority"].get(tier, {}).get("mean_p", float("nan"))
            for p in policies
        ]
        _data_row(tier, vals, ".4f")

    print(sep)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_comparison(verbose: bool = True) -> dict[str, dict]:
    """
    Load data, build D matrix, run all policies, return metrics dict.
    """
    print("\n" + "=" * 55)
    print("  STEP 3 — OPTIMISATION COMPARISON")
    print("=" * 55)

    # --- Load data ---
    print("\n[1/4] Loading data...")
    data         = load_all()
    entries_df   = data["waitlist_entry"]
    facilities_df = data["facility_details"]
    vacancies_df  = data["vacancies"]

    # --- Build D matrix ---
    print("[2/4] Building model inputs (D matrix)...")
    inputs = prepare_model_inputs(
        entries_df, facilities_df, vacancies_df, verbose=True
    )
    snapshot       = inputs["snapshot"]
    D              = inputs["D"]
    capacities     = inputs["capacities"]
    gender_blocks  = inputs["gender_blocks"]
    facility_names = inputs["facility_names"]

    print(f"\n  Clients (P):    {len(snapshot)}")
    print(f"  Facilities (N): {len(facility_names)}")
    print(f"  Total capacity: {sum(capacities):,} beds")
    print(f"  Gender blocks:  {len(gender_blocks):,} pairs")

    # --- Baselines ---
    print("\n[3/4] Running baselines...")

    print("  → Historical assignment...")
    hist_df = historical_assignment(
        entries_df, snapshot, facility_names, D
    )

    print("  → Cyclic FIFO (VCH baseline)...")
    cyclic_df = cyclic_fifo_assignment(
        snapshot, facility_names, capacities, D, gender_blocks
    )

    print("  → Random assignment...")
    rand_df = random_assignment(
        snapshot, facility_names, capacities, D, gender_blocks
    )

    # --- LP optimum ---
    print("\n[4/4] Solving LP optimisation...")
    lp_result = solve_lp(D, capacities, gender_blocks, verbose=False)
    print(f"  Status: {lp_result['status']}  |  "
          f"Solve time: {lp_result['solve_time_s']:.1f}s  |  "
          f"Active vars: {lp_result['n_vars_active']:,}  |  "
          f"Gender blocks: {lp_result['n_gender_blocks']:,}")

    lp_df = assignment_to_df(lp_result, snapshot, facility_names, D)

    # --- Metrics ---
    results = {
        "historical":  compute_metrics(hist_df,    snapshot, "historical"),
        "cyclic_fifo": compute_metrics(cyclic_df,  snapshot, "cyclic_fifo"),
        "random":      compute_metrics(rand_df,    snapshot, "random"),
        "lp_optimal":  compute_metrics(lp_df,      snapshot, "lp_optimal"),
    }

    # --- Report ---
    print_report(results)

    return results


if __name__ == "__main__":
    run_comparison()
