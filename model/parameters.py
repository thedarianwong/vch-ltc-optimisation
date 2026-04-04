"""
model/parameters.py — Central configuration for all model parameters and helpers.

ALL β coefficients are PLACEHOLDERS estimated from mock data distributions.
Replace with fitted values once real VCH offer outcome data is available:
    logistic regression on WaitlistOfferOutcome ~ covariates

Import pattern for all downstream modules (Steps 3–5):
    from model.parameters import ALPHA, BETA, compute_p_pn, encode_priority, ...
"""

from __future__ import annotations
import numpy as np
from datetime import date

# ---------------------------------------------------------------------------
# Composite clinical complexity weights
# ---------------------------------------------------------------------------

W_MENTAL   = 0.7   # CPS weight  — cognitive/behavioural issues (dominant driver)
W_PHYSICAL = 0.3   # ADL weight  — physical dependency

# ---------------------------------------------------------------------------
# Provider deferral logistic model  — ALL VALUES ARE PLACEHOLDERS
# ---------------------------------------------------------------------------

ALPHA: float = -1.8   # baseline log-odds  (σ(-1.8) ≈ 14% base deferral rate)

BETA: dict[str, float] = {
    "r_p":        +0.9,    # composite complexity: higher → more likely declined
    "q_p":        -0.2,    # priority ordinal: higher urgency → less declined
    "h_n":        +0.1,    # for-profit facility: slightly more selective
    "r_p_x_h_n":  +0.3,    # for-profit × complexity interaction
    "g_pn":       +1.5,    # gender mismatch: strong deferral signal
    "t":          +0.15,   # flu season (Nov–Jan): slightly elevated deferral
}

SIGMA_SQ: float = 0.25   # variance of facility random effect  u_n ~ N(0, σ²)

# ---------------------------------------------------------------------------
# Priority encoding  (WaitlistPriority string → ordinal q_p)
# ---------------------------------------------------------------------------

PRIORITY_MAP: dict[str, int] = {
    "Transfer/Site Specific": 0,
    "Community High":         1,
    "Community Emergency":    2,
    "Acute Care":             3,
}

PRIORITY_LABELS: list[str] = [
    "Transfer/Site Specific",
    "Community High",
    "Community Emergency",
    "Acute Care",
]

# Arrival fraction per tier — calibrated to real VCH patient data.
# Transfer/Site Specific set to 40% (dominant tier, ~57% of all VCH LTC patients
# are site-specific transfer requests) so Cyclic FIFO yields W_Transfer ≈ 473 days,
# matching OSA 2024/25 mean wait for non-urgent (Transfer + CommHigh) clients.
# CommHigh at 10% is underloaded under Cyclic FIFO → short wait (realistic).
PRIORITY_FRACS: dict[str, float] = {
    "Acute Care":             0.30,
    "Community Emergency":    0.20,
    "Community High":         0.10,
    "Transfer/Site Specific": 0.40,   # dominant tier; tuned: W_Transfer ≈ 473 days under Cyclic FIFO
}

# ---------------------------------------------------------------------------
# Queue / DES parameters
# ---------------------------------------------------------------------------

ARRIVAL_RATE: float = 28 / 30.0   # clients per day (~28 new clients/month)

# Per-tier abandonment rates (daily).  Each tier has different "patience":
#   Acute Care      — hospital cannot hold patients indefinitely; alternative
#                     placement found if waitlist too slow  (~2-month mean)
#   Community Emerg — needs placement soon but has some community support  (~3-month mean)
#   Community High  — non-urgent, living at home, very patient             (~50-month mean)
#   Transfer/Site   — wants specific facility; patient but has a limit     (~11-month mean)
#                     Calibrated so Cyclic FIFO yields W_Transfer ≈ 473 days (OSA 2024/25).
#                     θ = 9.4%/month derived from: W = (λ_T - μ_T)/(μ_T × θ)
#                     = (11.2 - 4.5)/(4.5 × 0.094) ≈ 15.8 months ≈ 473 days
ABANDONMENT_RATE: dict[str, float] = {
    "Acute Care":             0.50  / 30.0,
    "Community Emergency":    0.30  / 30.0,
    "Community High":         0.02  / 30.0,    # 2%/month ≈ 50-month patience
    "Transfer/Site Specific": 0.094 / 30.0,    # 9.4%/month ≈ 11-month patience; calibrated to W_Transfer ≈ 473 days
}

FLU_MONTHS: set[int] = {11, 12, 1}      # November, December, January

SIM_HORIZON: int    = 3 * 365    # total simulation horizon in days (3 years)
WARM_UP_DAYS: int   = 6 * 30     # warm-up period to discard from stats (6 months)
N_REPLICATIONS: int = 10         # default replications per policy
MAX_OFFERS: int    = 15         # safety cap: max offer attempts before a vacancy is abandoned

# ---------------------------------------------------------------------------
# Bed-scarcity parameters  (the REAL VCH situation)
# ---------------------------------------------------------------------------

# Beds available to the ACTIVE WAITLIST per day.
# VCH total vacancies ≈ 147/month (5300 beds / 1083-day LOS).
# Most go to direct hospital-to-LTC transfers; only ~18/month reach the
# priority waitlist (this model's scope).
#
# Steady-state calibration (dominant Transfer queue, θ = 9.4%/month):
#   Q_Transfer ≈ Δλ_Transfer / θ_Transfer = 6.7 / 0.094 = 71 clients
#   Q_Acute    ≈ Δλ_Acute / θ_Acute       = 3.9 / 0.50  =  8 clients
#   Q_CommEm   ≈ Δλ_CommEm / θ_CommEm     = 1.1 / 0.30  =  4 clients
#   Q_CommHigh ≈ 0  (underloaded: λ_CH = 2.8/mo < μ_CH = 4.5/mo)
#   Q_total    ≈ 83 clients
VACANCY_RATE: float        = 18 / 30.0   # waitlist-accessible bed openings per day

# Number of clients on the waitlist at simulation start.
# Steady-state with tier-specific θ: Q_ss ≈ 83 (Transfer≈71, AC≈8, CommEm≈4).
# We initialise at 90 to reflect a realistic starting backlog.
INITIAL_QUEUE_SIZE: int    = 90

# Days a bed sits empty between a declination and the next offer attempt.
# VCH targets a 4-day bed-turnaround; each offer→response cycle ≈ 2 days.
OFFER_TURNAROUND_DAYS: float = 2.0

# ---------------------------------------------------------------------------
# Policy parameters
# ---------------------------------------------------------------------------

# Policy 1 (baseline): equal cyclic rotation targets across all 4 tiers (25% each)
CYCLING_TARGETS: dict[str, float] = {
    "Acute Care":             0.25,
    "Community Emergency":    0.25,
    "Community High":         0.25,
    "Transfer/Site Specific": 0.25,
}

# Policy 2 (optimised): fairness window to maintain approximate equity
FAIRNESS_WINDOW: int      = 20    # rolling window: last N placements
FAIRNESS_TOLERANCE: float = 0.10  # allowed deviation from each tier's target share

# ---------------------------------------------------------------------------
# DES (Step 5) parameters
# ---------------------------------------------------------------------------

# Days in each tier before escalating to the next higher tier.
# Escalation is clinically triggered (health deterioration); Community High
# has NO automatic time-based escalation — non-urgent clients wait until
# placed or until their condition changes.
ESCALATION_DAYS: dict[str, float | None] = {
    "Transfer/Site Specific": 90.0,   # → Community High after 3 months
    "Community High":         None,   # clinically triggered only — no auto-escalation
    "Community Emergency":    45.0,   # → Acute Care after 6 weeks
    "Acute Care":             None,   # already highest — no escalation
}

# Escalation chain (what tier a client moves to)
ESCALATION_TARGET: dict[str, str] = {
    "Transfer/Site Specific": "Community High",
    "Community Emergency":    "Acute Care",
}

# Compound Poisson batch arrivals: with probability BATCH_PROB, an arrival
# event generates 2–BATCH_MAX_SIZE clients (hospital discharge planning day)
BATCH_PROB:     float = 0.10   # 10% of arrival events are batch
BATCH_MAX_SIZE: int   = 4      # max clients in a batch arrival

# Geographic mismatch soft penalty: logit penalty added to p_pn when client's
# home region ≠ facility region. Models client/family preference for local placement.
GEO_PENALTY: float = 0.25   # small penalty (σ(GEO_PENALTY) effect)

# Community regions in VCH
REGIONS: list[str] = ["Vancouver", "Richmond", "North Shore", "Coast"]

# DES uses more replications for robust CIs
DES_N_REPLICATIONS: int = 30

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_r_p(cps: int, adl: int) -> float:
    """
    Composite clinical complexity score, normalised to [0, 1].

        r_p = W_MENTAL * (CPS / 6) + W_PHYSICAL * (ADL / 6)

    CPS 0–6  Cognitive Performance Scale  (0 = intact, 6 = very severe impairment)
    ADL 0–6  ADL Self-Performance Hierarchy  (0 = independent, 6 = total dependence)
    """
    return W_MENTAL * (cps / 6.0) + W_PHYSICAL * (adl / 6.0)


def compute_p_pn(
    r_p:  float,
    q_p:  int,
    h_n:  int,
    g_pn: int,
    t:    int,
    u_n:  float = 0.0,
) -> float:
    """
    Provider deferral probability for client p offered facility n.

        p_pn = σ(α + u_n + β1·r_p + β2·q_p + β3·h_n + β4·(r_p·h_n) + β5·g_pn + β6·t)

    Parameters
    ----------
    r_p  : composite clinical complexity [0, 1]  — from compute_r_p()
    q_p  : priority ordinal {0, 1, 2, 3}         — from encode_priority()
    h_n  : for-profit flag {0, 1}                — from funding_to_h_n()
    g_pn : gender mismatch flag {0, 1}           — from is_gender_mismatch()
    t    : flu season flag {0, 1}                — from is_flu_season()
    u_n  : facility random effect (default 0.0 for point estimate)

    Returns
    -------
    float in (0, 1) — probability provider declines this offer
    """
    z = (
        ALPHA
        + u_n
        + BETA["r_p"]        * r_p
        + BETA["q_p"]        * q_p
        + BETA["h_n"]        * h_n
        + BETA["r_p_x_h_n"]  * r_p * h_n
        + BETA["g_pn"]       * g_pn
        + BETA["t"]          * t
    )
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))


def encode_priority(priority_str: str) -> int:
    """
    Map WaitlistPriority string → ordinal integer q_p.
    Raises KeyError for unrecognised strings.
    """
    return PRIORITY_MAP[priority_str]


def is_gender_mismatch(client_gender: str, room_limitation) -> int:
    """
    Return g_pn = 1 if the room has a gender limitation that differs from
    the client's gender; 0 if no limitation or genders match.

    room_limitation: "Male" | "Female" | None | NaN
    """
    if room_limitation is None:
        return 0
    if isinstance(room_limitation, float) and np.isnan(room_limitation):
        return 0
    return 1 if client_gender != room_limitation else 0


def is_flu_season(d: date) -> int:
    """Return t = 1 if date d is in flu season (Nov, Dec, Jan), else 0."""
    return 1 if d.month in FLU_MONTHS else 0


def funding_to_h_n(funding: str) -> int:
    """
    Map facility funding type → for-profit binary flag h_n.
    "Private" → 1,  "Affiliate-NP" / "Health Authority" / other → 0.
    """
    return 1 if funding == "Private" else 0
