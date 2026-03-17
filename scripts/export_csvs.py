"""
Run once to export the four sheets from the Excel workbook to data/.
Usage: python scripts/export_csvs.py
"""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
XL_PATH = ROOT / "VCH Long Term Care - Mock Data.xlsx"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SHEET_TO_CSV = {
    "Facilities":        "facilities.csv",
    "Clients":           "clients.csv",
    "Referral_Events":   "referrals.csv",
    "Waitlist_Timeline": "waitlist_timeline.csv",
}

xl = pd.ExcelFile(XL_PATH)
for sheet, csv_name in SHEET_TO_CSV.items():
    df = xl.parse(sheet)
    out = DATA_DIR / csv_name
    df.to_csv(out, index=False)
    print(f"  {sheet} ({len(df)} rows) -> {out.relative_to(ROOT)}")

print("Done.")
