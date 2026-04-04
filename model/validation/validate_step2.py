"""
model/validate_step2.py — Validate parameters.py and data_loader.py.

Checks:
  1. Data loads without errors, correct row counts and dtypes
  2. compute_r_p() matches the _r_p column stored in waitlist_entry
  3. compute_p_pn() matches the _p_pn column stored in waitlist_entry
  4. encode_priority() maps all 4 labels correctly
  5. is_gender_mismatch() logic is correct
  6. is_flu_season() hits the right months
  7. funding_to_h_n() maps funding strings correctly

Run:  python model/validate_step2.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.data_loader  import load_all, load_offers
from model.parameters   import (
    compute_r_p, compute_p_pn, encode_priority,
    is_gender_mismatch, is_flu_season, funding_to_h_n,
    PRIORITY_MAP, PRIORITY_LABELS, PRIORITY_FRACS,
    ALPHA, BETA, SIGMA_SQ, W_MENTAL, W_PHYSICAL,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SEP  = "-" * 55


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"{status}  {label}")
    if not condition and detail:
        print(f"         {detail}")
    return condition


def run_validation() -> None:
    all_pass = True
    print("\n" + "=" * 55)
    print("  STEP 2 VALIDATION")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  1. Data loading")
    print(SEP)

    data = load_all()
    fac   = data["facility_details"]
    vac   = data["vacancies"]
    rooms = data["room_characteristics"]
    we    = data["waitlist_entry"]

    all_pass &= check("facility_details loaded",    len(fac) == 30,
                      f"got {len(fac)} rows, expected 30")
    all_pass &= check("vacancies loaded",           len(vac) == 800,
                      f"got {len(vac)} rows, expected 800")
    all_pass &= check("room_characteristics loaded", len(rooms) > 0)
    all_pass &= check("waitlist_entry loaded",      len(we) > 0,
                      f"got {len(we)} rows")

    # Date columns parsed
    sample_date = we["DateClientOffered"].dropna().iloc[0]
    all_pass &= check("DateClientOffered is date object",
                      isinstance(sample_date, date),
                      f"got type {type(sample_date)}")

    # Foreign key integrity: every ProviderName in waitlist_entry exists in facility_details
    we_providers  = set(we["ProviderName"].unique())
    fac_providers = set(fac["ProviderName"].unique())
    missing = we_providers - fac_providers
    all_pass &= check("FK: all ProviderNames in waitlist_entry exist in facility_details",
                      len(missing) == 0,
                      f"missing: {missing}")

    # Foreign key integrity: VacancyIDs
    we_vids  = set(we["VacancyID"].unique())
    vac_vids = set(vac["VacancyID"].unique())
    missing_v = we_vids - vac_vids
    all_pass &= check("FK: all VacancyIDs in waitlist_entry exist in vacancies",
                      len(missing_v) == 0,
                      f"missing: {list(missing_v)[:5]}")

    # Date ordering: DateClientOffered >= DateOnWaitlist
    bad = we.dropna(subset=["DateClientOffered", "DateOnWaitlist"])
    bad = bad[bad["DateClientOffered"] < bad["DateOnWaitlist"]]
    all_pass &= check("Date order: DateClientOffered >= DateOnWaitlist",
                      len(bad) == 0,
                      f"{len(bad)} violated rows")

    # ------------------------------------------------------------------
    # 2. compute_r_p matches stored _r_p
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  2. compute_r_p() vs stored _r_p")
    print(SEP)

    we["_r_p_check"] = we.apply(
        lambda row: compute_r_p(row["CPS_Score"], row["ADL_SP_Hierarchy"]), axis=1
    )
    max_diff = (we["_r_p_check"] - we["_r_p"]).abs().max()
    all_pass &= check("compute_r_p matches _r_p (tol 1e-6)",
                      max_diff < 1e-6,
                      f"max diff = {max_diff:.2e}")

    # Boundary checks
    all_pass &= check("r_p in [0, 1] for all rows",
                      we["_r_p_check"].between(0, 1).all())
    all_pass &= check("r_p = 0.0 when CPS=0 and ADL=0",
                      abs(compute_r_p(0, 0) - 0.0) < 1e-9)
    all_pass &= check("r_p = 1.0 when CPS=6 and ADL=6",
                      abs(compute_r_p(6, 6) - 1.0) < 1e-9)
    all_pass &= check("r_p weight check: CPS=6, ADL=0 → 0.7",
                      abs(compute_r_p(6, 0) - 0.7) < 1e-9)

    # ------------------------------------------------------------------
    # 3. compute_p_pn matches stored _p_pn
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  3. compute_p_pn() vs stored _p_pn")
    print(SEP)

    # Build facility lookup for u_n and h_n
    # Note: u_n values were drawn randomly in mock_data.py with same seed,
    # so we back them out from the stored _p_pn and _r_p values.
    # Instead of exact u_n recovery, we verify the FORMULA is correct by
    # checking that setting u_n=0 reproduces the deterministic part of p_pn,
    # then verify that the residual (from u_n) is normally distributed.

    # Deterministic part (u_n = 0): should be close to _p_pn on average
    we["_p_pn_det"] = we.apply(
        lambda row: compute_p_pn(
            r_p  = row["_r_p"],
            q_p  = encode_priority(row["WaitlistPriority"]),
            h_n  = row["_h_n"],
            g_pn = row["_g_pn"],
            t    = is_flu_season(row["DateClientOffered"]) if row["DateClientOffered"] else 0,
            u_n  = 0.0,
        ),
        axis=1,
    )

    # The stored _p_pn includes u_n, so they won't match exactly.
    # Verify the formula structure: p_pn should be in (0,1) and
    # the stored values should be recoverable once we recover u_n.
    all_pass &= check("_p_pn in (0, 1) for all rows",
                      we["_p_pn"].between(0, 1).all())
    all_pass &= check("deterministic p_pn (u_n=0) in (0, 1)",
                      we["_p_pn_det"].between(0, 1).all())

    # Back out u_n from stored _p_pn and verify it's ~N(0, SIGMA_SQ)
    # log(p/(1-p)) = α + u_n + Σβ_i x_i  →  u_n = logit(p_pn) - (α + Σβ_i x_i)
    we["_logit_stored"] = np.log(we["_p_pn"] / (1 - we["_p_pn"]))
    we["_logit_det"]    = np.log(we["_p_pn_det"] / (1 - we["_p_pn_det"]))
    we["_u_n_recovered"] = we["_logit_stored"] - we["_logit_det"]

    # u_n should be one value per facility (same u_n for all offers to that facility)
    u_n_by_fac = we.groupby("ProviderName")["_u_n_recovered"].std()
    max_within_fac_std = u_n_by_fac.max()
    all_pass &= check("Recovered u_n is constant within each facility (std < 1e-4)",
                      max_within_fac_std < 1e-4,
                      f"max within-facility std = {max_within_fac_std:.4f}")

    u_n_means = we.groupby("ProviderName")["_u_n_recovered"].mean()
    u_n_var   = u_n_means.var()
    all_pass &= check(f"Recovered u_n variance ≈ SIGMA_SQ={SIGMA_SQ} (tol ±0.15)",
                      abs(u_n_var - SIGMA_SQ) < 0.15,
                      f"recovered var = {u_n_var:.3f}, expected {SIGMA_SQ}")

    print(f"         Recovered u_n: mean={u_n_means.mean():.3f}, "
          f"var={u_n_var:.3f}  (expected mean≈0, var≈{SIGMA_SQ})")

    # ------------------------------------------------------------------
    # 4. encode_priority
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  4. encode_priority()")
    print(SEP)

    expected = {
        "Transfer/Site Specific": 0,
        "Community High":         1,
        "Community Emergency":    2,
        "Acute Care":             3,
    }
    for label, expected_val in expected.items():
        got = encode_priority(label)
        all_pass &= check(f'encode_priority("{label}") == {expected_val}',
                          got == expected_val, f"got {got}")

    try:
        encode_priority("Unknown Priority")
        all_pass &= check("encode_priority raises KeyError for unknown string", False)
    except KeyError:
        all_pass &= check("encode_priority raises KeyError for unknown string", True)

    # All priorities in waitlist_entry are recognised
    unrecognised = set(we["WaitlistPriority"].unique()) - set(PRIORITY_MAP.keys())
    all_pass &= check("All WaitlistPriority values in PRIORITY_MAP",
                      len(unrecognised) == 0,
                      f"unrecognised: {unrecognised}")

    # ------------------------------------------------------------------
    # 5. is_gender_mismatch
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  5. is_gender_mismatch()")
    print(SEP)

    cases = [
        ("Female", None,     0, "no limitation → no mismatch"),
        ("Female", "Female", 0, "same gender → no mismatch"),
        ("Female", "Male",   1, "Female client, Male room → mismatch"),
        ("Male",   "Male",   0, "same gender → no mismatch"),
        ("Male",   "Female", 1, "Male client, Female room → mismatch"),
        ("Female", float("nan"), 0, "NaN limitation → no mismatch"),
    ]
    for client_g, room_lim, expected_val, desc in cases:
        got = is_gender_mismatch(client_g, room_lim)
        all_pass &= check(f"is_gender_mismatch: {desc}", got == expected_val,
                          f"got {got}, expected {expected_val}")

    # Verify _g_pn column matches re-computed values (need vacancies for RoomGenderLimitation)
    we_vac = we.merge(
        vac[["VacancyID", "RoomGenderLimitation"]], on="VacancyID", how="left"
    )
    we_vac["_g_pn_check"] = we_vac.apply(
        lambda row: is_gender_mismatch(row["ClientGender"], row["RoomGenderLimitation"]),
        axis=1,
    )
    match = (we_vac["_g_pn_check"] == we_vac["_g_pn"]).all()
    all_pass &= check("_g_pn column matches is_gender_mismatch() for all rows", match)

    # ------------------------------------------------------------------
    # 6. is_flu_season
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  6. is_flu_season()")
    print(SEP)

    flu_cases = [
        (date(2023, 11, 1), 1, "November → flu"),
        (date(2023, 12, 25), 1, "December → flu"),
        (date(2024,  1, 15), 1, "January → flu"),
        (date(2023,  2,  1), 0, "February → not flu"),
        (date(2023,  6, 15), 0, "June → not flu"),
        (date(2023, 10, 31), 0, "October → not flu"),
    ]
    for d, expected_val, desc in flu_cases:
        got = is_flu_season(d)
        all_pass &= check(f"is_flu_season: {desc}", got == expected_val,
                          f"got {got}, expected {expected_val}")

    # ------------------------------------------------------------------
    # 7. funding_to_h_n
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  7. funding_to_h_n()")
    print(SEP)

    funding_cases = [
        ("Private",       1, "Private → for-profit"),
        ("Affiliate-NP",  0, "Affiliate-NP → non-profit"),
        ("Health Authority", 0, "Health Authority → non-profit"),
        ("Other",         0, "Other → non-profit"),
    ]
    for funding, expected_val, desc in funding_cases:
        got = funding_to_h_n(funding)
        all_pass &= check(f"funding_to_h_n: {desc}", got == expected_val,
                          f"got {got}")

    h_n_by_fac = fac.apply(
        lambda row: funding_to_h_n(row["funding"]), axis=1
    )
    private_count = (fac["funding"] == "Private").sum()
    h_n_count     = h_n_map = h_n_by_fac.sum()
    all_pass &= check("h_n count matches Private facility count",
                      private_count == h_n_count,
                      f"funding_to_h_n gives {h_n_count}, Private count = {private_count}")

    # ------------------------------------------------------------------
    # Summary stats (informational)
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  Summary statistics (informational)")
    print(SEP)

    offers = load_offers()
    n_dec  = (offers["WaitlistOfferOutcome"] == "Provider Declined").sum()
    n_acc  = (offers["WaitlistOfferOutcome"] == "Accepted").sum()
    print(f"  Total offers:            {len(offers)}")
    print(f"  Accepted:                {n_acc} ({n_acc/len(offers)*100:.1f}%)")
    print(f"  Provider Declined:       {n_dec} ({n_dec/len(offers)*100:.1f}%)")
    print(f"  Mean p_pn (all offers):  {offers['_p_pn'].mean():.3f}")
    print(f"  Mean r_p  (all offers):  {offers['_r_p'].mean():.3f}")
    print(f"  Gender mismatch rate:    {offers['_g_pn'].mean()*100:.1f}%")

    print(f"\n  Priority distribution (unique clients):")
    unique_clients = we.drop_duplicates("PatientID")
    for lbl in PRIORITY_LABELS:
        n = (unique_clients["WaitlistPriority"] == lbl).sum()
        print(f"    {lbl:<28}: {n} ({n/len(unique_clients)*100:.1f}%)")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    if all_pass:
        print("  ALL CHECKS PASSED — Step 2 validated.")
    else:
        print("  SOME CHECKS FAILED — see [FAIL] lines above.")
    print("=" * 55)


if __name__ == "__main__":
    run_validation()
