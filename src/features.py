"""
features.py
Binary feature engineering for every (client, facility) pair.

All indicator functions return 1 or 0 (int).
"""

# ---------------------------------------------------------------------------
# Language–Zone affinity lookup
# A mismatch means the client's language is NOT supported in the facility's zone.
# English (and any unlisted language) is assumed supported everywhere.
# ---------------------------------------------------------------------------
LANGUAGE_ZONE_SUPPORT: dict[str, set[str]] = {
    "Cantonese": {"Richmond", "Vancouver South", "Vancouver Midtown"},
    "Mandarin":  {"Richmond", "Vancouver South", "Vancouver Midtown"},
    "Punjabi":   {"Vancouver South"},
    "Vietnamese": {"Vancouver Downtown/West End", "Vancouver Midtown"},
    "Korean":    {"Vancouver Midtown", "Vancouver Westside"},
}

# ---------------------------------------------------------------------------
# Diagnosis–Specialisation compatibility lookup
# A mismatch means the facility's specialisation cannot serve the diagnosis.
# ---------------------------------------------------------------------------
DIAGNOSIS_SPECIALISATION_COMPAT: dict[str, set[str]] = {
    "Dementia/Alzheimer's":  {"Dementia/Cognitive", "General"},
    "Stroke/CVA":            {"Complex Medical", "Rehabilitation", "General"},
    "Heart Failure":         {"Complex Medical", "General"},
    "COPD":                  {"Complex Medical", "General"},
    "Diabetes (complex)":    {"Complex Medical", "General"},
    "Cancer (palliative)":   {"Palliative", "General"},
    "Fracture/Fall Recovery":{"Rehabilitation", "General"},
    "Mental Health":         {"Psychogeriatric", "General"},
    "Multi-morbidity":       {"Complex Medical", "General"},
    "Parkinson's":           {"Complex Medical", "General"},
}
_ALL_SPECIALISATIONS = {
    "General", "Dementia/Cognitive", "Complex Medical",
    "Palliative", "Psychogeriatric", "Rehabilitation",
}


def language_mismatch(client_language: str, facility_zone: str) -> int:
    """1 if client's language is NOT supported in the facility's zone."""
    supported_zones = LANGUAGE_ZONE_SUPPORT.get(client_language)
    if supported_zones is None:
        return 0  # English / unlisted → no mismatch
    return int(facility_zone not in supported_zones)


def specialisation_mismatch(diagnosis: str, care_specialisation: str) -> int:
    """1 if facility specialisation cannot serve the diagnosis."""
    compatible = DIAGNOSIS_SPECIALISATION_COMPAT.get(diagnosis, _ALL_SPECIALISATIONS)
    return int(care_specialisation not in compatible)


def compute_features(client: dict, facility: dict) -> dict[str, int]:
    """
    Compute all binary indicator features for a single (client, facility) pair.

    Parameters
    ----------
    client  : dict-like row from clients_df (use df.to_dict('records'))
    facility: dict-like row from facilities_df

    Returns
    -------
    dict with keys matching feature names used in the logistic models.
    """
    prefs = {
        client.get("preferred_care_home_1"),
        client.get("preferred_care_home_2"),
        client.get("preferred_care_home_3"),
    } - {None, float("nan"), ""}

    return {
        # --- Provider decline features ---
        "high_clinical_score": int(client["clinical_assessment_score"] >= 17),
        "is_dementia":         int(client["primary_diagnosis"] == "Dementia/Alzheimer's"),
        "lang_mismatch":       language_mismatch(
                                   client["primary_language"], facility["geographic_zone"]
                               ),
        "is_for_profit":       int(facility["ownership_type"] == "Private For-Profit"),
        "is_small_facility":   int(facility["licensed_bed_count"] < 50),
        "spec_mismatch":       specialisation_mismatch(
                                   client["primary_diagnosis"], facility["care_specialisation"]
                               ),
        # --- Client decline features ---
        "not_preferred":       int(facility["facility_id"] not in prefs),
        "diff_zone":           int(facility["geographic_zone"] != client["geographic_zone"]),
        "is_non_urgent":       int(client["urgency_classification"].startswith("Priority 3")),
        "is_over_85":          int(client["age"] >= 85),
        "is_acute":            int(str(client["referral_source"]).startswith("Acute")),
    }
