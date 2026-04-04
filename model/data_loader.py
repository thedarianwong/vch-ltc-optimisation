"""
model/data_loader.py — Load the four VCH LTC tables from data/generated/.

When real VCH data arrives, only this file changes (paths + any field renames).
All downstream code (Steps 3–5) imports load_all() from here.
"""

from __future__ import annotations
import os
import pandas as pd

# Resolve data/generated/ relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DIR  = os.path.join(_PROJECT_ROOT, "data", "generated")

# Columns to parse as dates per table
_DATE_COLS: dict[str, list[str]] = {
    "waitlist_entry": [
        "DateOnWaitlist",
        "DateOffWaitlist",
        "DateVacancyReported",
        "DateBedAvailable",
        "DateClientOffered",
        "DateOfReply",
    ],
    "vacancies": [
        "DateVacancyReported",
        "DateBedAvailable",
    ],
}


def load_table(name: str, data_dir: str = _DEFAULT_DIR) -> pd.DataFrame:
    """
    Load a single table CSV and parse date columns.

    Parameters
    ----------
    name     : table name without .csv extension
               ("facility_details" | "vacancies" | "room_characteristics" | "waitlist_entry")
    data_dir : directory containing the CSV files
    """
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Table '{name}' not found at {path}. "
            "Run `python data/mock_data.py` to generate it."
        )

    date_cols = _DATE_COLS.get(name, [])
    df = pd.read_csv(path, parse_dates=date_cols, low_memory=False)

    # Convert datetime columns to plain date objects for consistency
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    return df


def load_all(data_dir: str = _DEFAULT_DIR) -> dict[str, pd.DataFrame]:
    """
    Load all four tables and return as a dict.

    Returns
    -------
    {
      "facility_details"     : DataFrame (30 rows)
      "vacancies"            : DataFrame (800 rows)
      "room_characteristics" : DataFrame (~264 rows)
      "waitlist_entry"       : DataFrame (~578 rows — core offer-event table)
    }
    """
    return {
        "facility_details":     load_table("facility_details",     data_dir),
        "vacancies":            load_table("vacancies",            data_dir),
        "room_characteristics": load_table("room_characteristics", data_dir),
        "waitlist_entry":       load_table("waitlist_entry",       data_dir),
    }


def load_offers(data_dir: str = _DEFAULT_DIR) -> pd.DataFrame:
    """
    Convenience: load waitlist_entry and filter to provider-side rows only.
    Excludes any non-provider outcomes (e.g. client refusals) if present in real data.

    Returns
    -------
    DataFrame with WaitlistOfferOutcome in {"Accepted", "Provider Declined"}
    """
    df = load_table("waitlist_entry", data_dir)
    provider_outcomes = {"Accepted", "Provider Declined"}
    return df[df["WaitlistOfferOutcome"].isin(provider_outcomes)].copy()
