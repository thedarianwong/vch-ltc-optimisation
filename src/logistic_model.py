"""
logistic_model.py
Computes declination probabilities for (client, facility) pairs
using the two logistic regression models from the Statistical Assumptions doc.

P_prov  : probability provider declines the referral
P_client: probability client declines the offer (given provider accepted)
P_decline = 1 - (1 - P_prov)(1 - P_client)   [combined, assuming independence]
"""
import math
import numpy as np
import pandas as pd

from src.features import compute_features

# ---------------------------------------------------------------------------
# Model coefficients  (from Statistical_Assumptions.docx)
# ---------------------------------------------------------------------------

PROVIDER_INTERCEPT = -1.8
PROVIDER_BETAS = {
    "high_clinical_score": +0.9,
    "is_dementia":         +0.5,
    "lang_mismatch":       +0.3,
    "is_for_profit":       -0.3,
    "is_small_facility":   +0.4,
    "spec_mismatch":       +0.7,
}

CLIENT_INTERCEPT = -1.2
CLIENT_BETAS = {
    "not_preferred":  +1.4,
    "diff_zone":      +0.8,
    "lang_mismatch":  +0.6,
    "is_non_urgent":  +0.3,
    "is_over_85":     -0.3,
    "is_acute":       -0.4,
}


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def compute_p_provider(client: dict, facility: dict) -> float:
    """P(provider declines) for a single (client, facility) pair."""
    feats = compute_features(client, facility)
    z = PROVIDER_INTERCEPT + sum(b * feats[k] for k, b in PROVIDER_BETAS.items())
    return _sigmoid(z)


def compute_p_client(client: dict, facility: dict) -> float:
    """P(client declines) for a single (client, facility) pair."""
    feats = compute_features(client, facility)
    z = CLIENT_INTERCEPT + sum(b * feats[k] for k, b in CLIENT_BETAS.items())
    return _sigmoid(z)


def compute_p_decline(client: dict, facility: dict) -> float:
    """
    Combined P(any declination) = 1 - (1 - P_prov)(1 - P_client).
    Assumes provider and client decisions are independent.
    """
    p_prov   = compute_p_provider(client, facility)
    p_client = compute_p_client(client, facility)
    return 1.0 - (1.0 - p_prov) * (1.0 - p_client)


def build_decline_matrix(
    clients_df: pd.DataFrame,
    facilities_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build the full P(decline) matrix of shape (n_clients, n_facilities).

    Entry [i, j] = P_decline(client i, facility j).

    This is the core input to the optimisation module.
    """
    clients    = clients_df.to_dict("records")
    facilities = facilities_df.to_dict("records")

    n_c = len(clients)
    n_f = len(facilities)
    matrix = np.zeros((n_c, n_f), dtype=np.float32)

    for i, client in enumerate(clients):
        for j, facility in enumerate(facilities):
            matrix[i, j] = compute_p_decline(client, facility)

    return matrix
