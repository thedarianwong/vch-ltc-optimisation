"""
data_loader.py
Loads and validates the four LTC datasets.
Returns typed DataFrames ready for feature engineering and modelling.
"""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

DATE_COLS_CLIENTS    = ["waitlist_entry_date"]
DATE_COLS_REFERRALS  = ["referral_date", "admission_date"]
DATE_COLS_TIMELINE   = ["waitlist_entry_date", "final_admission_date"]


def load_facilities(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "facilities.csv")
    assert set(["facility_id", "ownership_type", "licensed_bed_count",
                "geographic_zone", "care_specialisation"]).issubset(df.columns)
    return df


def load_clients(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "clients.csv", parse_dates=DATE_COLS_CLIENTS)
    assert set(["client_id", "age", "gender", "primary_language", "geographic_zone",
                "referral_source", "urgency_classification", "clinical_assessment_score",
                "primary_diagnosis"]).issubset(df.columns)
    return df


def load_referrals(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "referrals.csv", parse_dates=DATE_COLS_REFERRALS)
    return df


def load_timeline(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "waitlist_timeline.csv", parse_dates=DATE_COLS_TIMELINE)
    return df


def load_all(data_dir: pathlib.Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Return all four datasets in a dict keyed by short name."""
    return {
        "facilities": load_facilities(data_dir),
        "clients":    load_clients(data_dir),
        "referrals":  load_referrals(data_dir),
        "timeline":   load_timeline(data_dir),
    }
