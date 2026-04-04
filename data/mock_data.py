"""
data/mock_data.py — Generate synthetic VCH LTC data matching the exact schema.

Four tables produced (saved as CSVs to data/generated/):
  facility_details      : 30 facilities with room counts and funding type
  vacancies             : ~800 bed vacancy events over 3 years
  room_characteristics  : ceiling lift / accessibility per vacancy
  waitlist_entry        : ~1350 offer events (core table, one row per offer)

Date consistency enforced:
  DateOnWaitlist <= DateClientOffered < DateOfReply <= DateOffWaitlist

Run:
  python data/mock_data.py
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

# ---------------------------------------------------------------------------
# Config — all placeholders, will be replaced by real data / estimated betas
# ---------------------------------------------------------------------------

SEED       = 42
rng        = np.random.default_rng(SEED)

N_FACILITIES = 30
N_CLIENTS    = 500
N_VACANCIES  = 800

START_DATE = date(2022, 4, 1)   # FY2022 start
END_DATE   = date(2025, 3, 31)  # FY2025 end

# --- Logistic model parameters (PLACEHOLDERS) ---
ALPHA   = -1.8
BETAS   = {
    "r_p":        0.9,   # composite clinical complexity
    "q_p":       -0.2,   # priority ordinal (higher urgency → less declined)
    "h_n":        0.1,   # for-profit facility
    "r_p_x_h_n":  0.3,   # for-profit × complexity interaction
    "g_pn":       1.5,   # gender mismatch (strong deferral)
    "t":          0.15,  # flu season (Nov–Jan)
}
SIGMA_SQ   = 0.25   # facility random effect variance
W_MENTAL   = 0.7    # CPS weight in r_p
W_PHYSICAL = 0.3    # ADL weight in r_p

# --- Priority encoding ---
PRIORITY_MAP = {
    "Transfer/Site Specific": 0,
    "Community High":         1,
    "Community Emergency":    2,
    "Acute Care":             3,
}
PRIORITY_LABELS = list(PRIORITY_MAP.keys())
PRIORITY_PROBS  = [0.15, 0.20, 0.30, 0.35]   # Transfer, CommHigh, CommEm, Acute

# --- Score distributions (from LATEST_PLAN.md) ---
CPS_PROBS   = [0.05, 0.10, 0.20, 0.25, 0.20, 0.12, 0.08]   # 0–6
CHESS_PROBS = [0.08, 0.20, 0.30, 0.22, 0.13, 0.07]          # 0–5
MAPLE_PROBS = [0.05, 0.15, 0.25, 0.30, 0.25]                 # 1–5
ADL_PROBS   = [0.03, 0.08, 0.15, 0.22, 0.24, 0.18, 0.10]    # 0–6

# --- Geography ---
REGIONS = ["Vancouver", "Richmond", "North Shore", "Coast"]
REGION_PROBS = [0.40, 0.25, 0.20, 0.15]
LHA_MAP = {
    "Vancouver":   ["Vancouver Community"],
    "Richmond":    ["Richmond"],
    "North Shore": ["North Shore", "West Vancouver"],
    "Coast":       ["Sunshine Coast", "Powell River"],
}

# --- Reject reasons (provider-side) ---
REJECT_REASONS = [
    "High Acuity/Complexity",
    "Behaviour Concern",
    "Bariatric",
    "No Availability",
    "Client Unsuitable",
    "Medical Complexity",
    "Specialty Mismatch",
    "Other",
]
REJECT_PROBS = [0.25, 0.20, 0.08, 0.15, 0.12, 0.10, 0.05, 0.05]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, max(delta, 1))))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def compute_r_p(cps: int, adl: int) -> float:
    """Composite clinical complexity, normalised to [0, 1]."""
    return W_MENTAL * (cps / 6.0) + W_PHYSICAL * (adl / 6.0)


def compute_p_pn(r_p: float, q_p: int, h_n: int, g_pn: int, t: int, u_n: float) -> float:
    """Provider deferral probability from logistic model."""
    z = (
        ALPHA
        + u_n
        + BETAS["r_p"]        * r_p
        + BETAS["q_p"]        * q_p
        + BETAS["h_n"]        * h_n
        + BETAS["r_p_x_h_n"]  * r_p * h_n
        + BETAS["g_pn"]       * g_pn
        + BETAS["t"]          * t
    )
    return _sigmoid(z)


def _is_flu_season(d: date) -> int:
    return 1 if d.month in {11, 12, 1} else 0


def _pick_lha(region: str) -> str:
    opts = LHA_MAP[region]
    return opts[rng.integers(len(opts))]


# ---------------------------------------------------------------------------
# Table 1: facility_details
# ---------------------------------------------------------------------------

def generate_facilities() -> pd.DataFrame:
    rows = []
    for i in range(1, N_FACILITIES + 1):
        region  = rng.choice(REGIONS, p=REGION_PROBS)
        lha     = _pick_lha(region)
        funding = rng.choice(
            ["Affiliate-NP", "Health Authority", "Private"],
            p=[0.45, 0.25, 0.30],
        )
        total_beds = int(rng.integers(40, 201))

        # Room type split (approximate)
        single = int(total_beds * rng.uniform(0.30, 0.65))
        double = int(total_beds * rng.uniform(0.10, 0.35))
        three  = int(total_beds * rng.uniform(0.05, 0.15))
        four   = max(0, total_beds - single - double * 2 - three * 3) // 4

        rows.append({
            "placement_site_code_src":       i,
            "ProviderName":                   f"Care Home {i:03d}",
            "placement_site_reporting_name":  f"Care Home {i:03d} - {region}",
            "community_region_desc":          region,
            "lha_name":                       lha,
            "funding":                        funding,
            "Single LTC Rooms":               single,
            "Double LTC Rooms":               double,
            "3-Bed LTC Rooms":                three,
            "4-Bed LTC Rooms":                four,
            ">4 Bed LTC Rooms":               0,
            "Total LTC Beds":                 total_beds,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 2: vacancies
# ---------------------------------------------------------------------------

def generate_vacancies(facilities_df: pd.DataFrame) -> pd.DataFrame:
    providers = facilities_df["ProviderName"].tolist()
    rows = []
    for i in range(1, N_VACANCIES + 1):
        provider    = providers[rng.integers(N_FACILITIES)]
        fac         = facilities_df.loc[
            facilities_df["ProviderName"] == provider
        ].iloc[0]
        region      = fac["community_region_desc"]

        # Vacancy reported date — spread across study period, leaving room for dates after
        vac_reported = _random_date(START_DATE, END_DATE - timedelta(days=30))
        bed_available = vac_reported + timedelta(days=int(rng.integers(1, 8)))

        gender_lim = rng.choice([None, "Female", "Male"], p=[0.90, 0.05, 0.05])

        rows.append({
            "VacancyID":              i,
            "WaitlistName":           f"PRIORITY ACCESS {region.upper()}",
            "ProviderName":           provider,
            "DateVacancyReported":    vac_reported,
            "DateBedAvailable":       bed_available,
            "IsPrivateRoom":          int(rng.random() < 0.25),
            "RoomGenderLimitation":   gender_lim,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 3: room_characteristics
# ---------------------------------------------------------------------------

def generate_room_characteristics(vacancies_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    uid = 1
    for _, vac in vacancies_df.iterrows():
        if rng.random() < 0.20:
            rows.append({
                "UniqueCharacteristicID": uid,
                "VacancyID":              vac["VacancyID"],
                "RoomCharacteristic":     "CEILING LIFT",
            })
            uid += 1
        if rng.random() < 0.10:
            rows.append({
                "UniqueCharacteristicID": uid,
                "VacancyID":              vac["VacancyID"],
                "RoomCharacteristic":     "WHEELCHAIR ACCESSIBLE",
            })
            uid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 4: waitlist_entry  (core offer-event table)
# ---------------------------------------------------------------------------

def generate_waitlist_entries(
    facilities_df: pd.DataFrame,
    vacancies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (client, offer) event.

    Generation logic:
      - 500 clients, each gets 1–5 offer events (weighted toward fewer)
      - For each offer, sample a vacancy, compute p_pn, draw outcome
      - If accepted early, stop issuing further offers for that client
      - Dates enforced: DateOnWaitlist <= DateClientOffered < DateOfReply
    """
    # Pre-compute fixed per-facility values
    u_n_map = {
        name: float(rng.normal(0, SIGMA_SQ ** 0.5))
        for name in facilities_df["ProviderName"]
    }
    h_n_map = {
        row["ProviderName"]: 1 if row["funding"] == "Private" else 0
        for _, row in facilities_df.iterrows()
    }

    vac_records = vacancies_df.to_dict("records")
    n_vac = len(vac_records)

    rows = []
    entry_id = 1

    # Number-of-offers distribution (mean ≈ 2.3, range 1–5)
    n_offers_choices = [1, 2, 3, 4, 5]
    n_offers_probs   = [0.30, 0.28, 0.22, 0.12, 0.08]

    for patient_id in range(1, N_CLIENTS + 1):
        # --- Client attributes ---
        region   = rng.choice(REGIONS, p=REGION_PROBS)
        lha      = _pick_lha(region)
        priority = rng.choice(PRIORITY_LABELS, p=PRIORITY_PROBS)
        age      = int(np.clip(rng.normal(82, 8), 65, 105))
        gender   = rng.choice(["Female", "Male"], p=[0.55, 0.45])
        cps      = int(rng.choice(range(7),   p=CPS_PROBS))
        chess    = int(rng.choice(range(6),   p=CHESS_PROBS))
        maple    = int(rng.choice(range(1, 6), p=MAPLE_PROBS))
        adl      = int(rng.choice(range(7),   p=ADL_PROBS))
        rug      = int(rng.integers(1, 24))

        r_p = compute_r_p(cps, adl)
        q_p = PRIORITY_MAP[priority]

        # Waitlist entry date — give 60-day buffer before study end
        date_on = _random_date(START_DATE, END_DATE - timedelta(days=60))

        n_offers = int(rng.choice(n_offers_choices, p=n_offers_probs))
        placed   = False

        for offer_num in range(n_offers):
            if placed:
                break

            # Sample a vacancy
            vac     = vac_records[rng.integers(n_vac)]
            provider = vac["ProviderName"]
            h_n      = h_n_map[provider]
            u_n      = u_n_map[provider]

            # Gender mismatch: g_pn = 1 if room has limitation AND client differs
            gender_lim = vac["RoomGenderLimitation"]
            g_pn = 1 if (gender_lim is not None and gender != gender_lim) else 0

            # Offer date: must be >= date_on AND >= DateBedAvailable
            bed_avail   = vac["DateBedAvailable"]
            earliest    = max(date_on, bed_avail)
            offer_date  = earliest + timedelta(days=int(rng.integers(0, 8)))
            if offer_date >= END_DATE:
                offer_date = END_DATE - timedelta(days=5)

            t = _is_flu_season(offer_date)

            # Compute deferral probability and sample outcome
            p_pn    = compute_p_pn(r_p, q_p, h_n, g_pn, t, u_n)
            declined = bool(rng.random() < p_pn)

            reply_date = offer_date + timedelta(days=int(rng.integers(1, 6)))
            if reply_date >= END_DATE:
                reply_date = END_DATE - timedelta(days=1)

            if not declined:
                outcome      = "Accepted"
                date_off     = reply_date
                reason       = None
                placed       = True
            else:
                outcome  = "Provider Declined"
                date_off = None   # still on waitlist after this offer
                reason   = rng.choice(REJECT_REASONS, p=REJECT_PROBS)

            rows.append({
                "SourceWaitlistEntryID":  entry_id,
                "PatientID":              patient_id,
                "CommunityRegion":        region,
                "CommunityLHA":           lha,
                "WaitlistName":           vac["WaitlistName"],
                "WaitlistPriority":       priority,
                "DateOnWaitlist":         date_on,
                "DateOffWaitlist":        date_off,
                # Client detail fields
                "ClientAge":              age,
                "ClientGender":           gender,
                "ClientHomeLHA":          lha,
                "CPS_Score":              cps,
                "CHESS_Score":            chess,
                "MAPLe_Score":            maple,
                "ADL_SP_Hierarchy":       adl,
                "RUG_III_HC_Cat":         rug,
                # Offer / vacancy fields
                "VacancyID":              vac["VacancyID"],
                "ProviderName":           provider,
                "DateVacancyReported":    vac["DateVacancyReported"],
                "DateBedAvailable":       vac["DateBedAvailable"],
                "DateClientOffered":      offer_date,
                "DateOfReply":            reply_date,
                "WaitlistOfferOutcome":   outcome,
                "ReasonRejected":         reason,
                # Derived columns (useful downstream)
                "_r_p":   round(r_p, 8),
                "_p_pn":  round(p_pn, 8),
                "_g_pn":  g_pn,
                "_h_n":   h_n,
            })
            entry_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(
    facilities_df: pd.DataFrame,
    vacancies_df: pd.DataFrame,
    room_chars_df: pd.DataFrame,
    entries_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 60)
    print("  MOCK DATA SUMMARY")
    print("=" * 60)

    print(f"\n[facility_details]  {len(facilities_df)} rows")
    print(f"  Regions:   {facilities_df['community_region_desc'].value_counts().to_dict()}")
    print(f"  Funding:   {facilities_df['funding'].value_counts().to_dict()}")
    print(f"  Beds:      min={facilities_df['Total LTC Beds'].min()}  "
          f"mean={facilities_df['Total LTC Beds'].mean():.0f}  "
          f"max={facilities_df['Total LTC Beds'].max()}")

    print(f"\n[vacancies]  {len(vacancies_df)} rows")
    print(f"  Date range: {vacancies_df['DateVacancyReported'].min()} → "
          f"{vacancies_df['DateVacancyReported'].max()}")
    print(f"  Gender limitation: "
          f"{vacancies_df['RoomGenderLimitation'].value_counts(dropna=False).to_dict()}")
    print(f"  Private rooms: {vacancies_df['IsPrivateRoom'].sum()} "
          f"({vacancies_df['IsPrivateRoom'].mean()*100:.1f}%)")

    print(f"\n[room_characteristics]  {len(room_chars_df)} rows")
    print(f"  {room_chars_df['RoomCharacteristic'].value_counts().to_dict()}")

    print(f"\n[waitlist_entry]  {len(entries_df)} rows")
    n_clients  = entries_df["PatientID"].nunique()
    n_accepted = (entries_df["WaitlistOfferOutcome"] == "Accepted").sum()
    n_declined = (entries_df["WaitlistOfferOutcome"] == "Provider Declined").sum()
    dec_rate   = n_declined / len(entries_df) * 100

    print(f"  Unique patients:      {n_clients}")
    print(f"  Total offer rows:     {len(entries_df)}")
    print(f"  Mean offers/patient:  {len(entries_df)/n_clients:.2f}")
    print(f"  Accepted:             {n_accepted}  ({n_accepted/len(entries_df)*100:.1f}%)")
    print(f"  Provider Declined:    {n_declined}  ({dec_rate:.1f}%)")
    print(f"  Patients placed:      "
          f"{(entries_df['WaitlistOfferOutcome']=='Accepted').groupby(entries_df['PatientID']).any().sum()}")

    print(f"\n  Priority distribution:")
    pct = entries_df.drop_duplicates("PatientID")["WaitlistPriority"].value_counts(normalize=True)
    for lbl, p in pct.items():
        print(f"    {lbl:<28}: {p*100:.1f}%")

    print(f"\n  CPS Score distribution (unique patients):")
    cps_counts = entries_df.drop_duplicates("PatientID")["CPS_Score"].value_counts().sort_index()
    for score, cnt in cps_counts.items():
        print(f"    CPS={score}: {cnt} ({cnt/n_clients*100:.1f}%)")

    print(f"\n  ADL distribution (unique patients):")
    adl_counts = entries_df.drop_duplicates("PatientID")["ADL_SP_Hierarchy"].value_counts().sort_index()
    for score, cnt in adl_counts.items():
        print(f"    ADL={score}: {cnt} ({cnt/n_clients*100:.1f}%)")

    print(f"\n  Mean r_p (composite complexity): {entries_df['_r_p'].mean():.3f}")
    print(f"  Mean p_pn (deferral prob):       {entries_df['_p_pn'].mean():.3f}")
    print(f"  Gender mismatch rate:            "
          f"{entries_df['_g_pn'].mean()*100:.1f}% of offers")

    print(f"\n  Declined offer reasons:")
    declined = entries_df[entries_df["WaitlistOfferOutcome"] == "Provider Declined"]
    if len(declined) > 0:
        for reason, cnt in declined["ReasonRejected"].value_counts().items():
            print(f"    {reason:<30}: {cnt}")

    # Date consistency check
    bad_dates = entries_df[
        entries_df["DateClientOffered"] < entries_df["DateOnWaitlist"]
    ]
    print(f"\n  Date consistency check (DateClientOffered >= DateOnWaitlist): "
          f"{'PASS' if len(bad_dates) == 0 else f'FAIL — {len(bad_dates)} bad rows'}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all(save: bool = True) -> dict:
    print("Generating mock VCH LTC data...")

    print("  [1/4] facility_details...")
    facilities_df = generate_facilities()

    print("  [2/4] vacancies...")
    vacancies_df = generate_vacancies(facilities_df)

    print("  [3/4] room_characteristics...")
    room_chars_df = generate_room_characteristics(vacancies_df)

    print("  [4/4] waitlist_entry (offer events)...")
    entries_df = generate_waitlist_entries(facilities_df, vacancies_df)

    if save:
        out_dir = os.path.join(os.path.dirname(__file__), "generated")
        os.makedirs(out_dir, exist_ok=True)

        facilities_df.to_csv(os.path.join(out_dir, "facility_details.csv"), index=False)
        vacancies_df.to_csv(os.path.join(out_dir, "vacancies.csv"), index=False)
        room_chars_df.to_csv(os.path.join(out_dir, "room_characteristics.csv"), index=False)
        entries_df.to_csv(os.path.join(out_dir, "waitlist_entry.csv"), index=False)

        print(f"\n  Saved 4 CSVs to {out_dir}/")

    print_summary(facilities_df, vacancies_df, room_chars_df, entries_df)

    return {
        "facility_details":     facilities_df,
        "vacancies":            vacancies_df,
        "room_characteristics": room_chars_df,
        "waitlist_entry":       entries_df,
    }


if __name__ == "__main__":
    generate_all(save=True)
