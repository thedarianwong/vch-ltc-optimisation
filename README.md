# vch-ltc-optimisation

Optimising Long-Term Care Placement Matching to Reduce Service Provider Declinations
**MATH 402W Capstone** — Team LTC (Darian, Nick, Kuncen, Paul) × Vancouver Coastal Health

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo
git clone <repo-url>
cd vch-ltc-optimisation

# 2. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Export the Excel data to CSVs (run once)
python3 scripts/export_csvs.py
```

To deactivate the venv when you're done:

```bash
deactivate
```