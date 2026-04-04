<div align="center">

# 🏥 Optimising Long-Term Care Placement
### 📊 A robust Operations Research & Discrete Event Simulation Framework

<p align="center">
  <img src="https://img.shields.io/badge/Operations%20Research-BIP%20%7C%20Queueing%20%7C%20DES-blue?style=flat-square" alt="OR Methods" />
  <img src="https://img.shields.io/badge/Method-Logistic%20Regression-E0A96D?style=flat-square" alt="Machine Learning" />
  <img src="https://img.shields.io/badge/Domain-Healthcare%20Analytics-green?style=flat-square" alt="Healthcare" />
</p>

<table align="center" style="border: none; background-color: transparent;">
  <tr style="border: none; background-color: transparent;">
    <td align="center" style="border: none;">📉<br><b>-59%</b><br>Expected<br>Declinations</td>
    <td align="center" style="border: none;">⏱️<br><b>-85%</b><br>Acute Care<br>Wait Times</td>
    <td align="center" style="border: none;">🛏️<br><b>-26%</b><br>Bed-Loss<br>Days</td>
    <td align="center" style="border: none;">📍<br><b>+173%</b><br>In-Region<br>Placements</td>
  </tr>
</table>

<br>
<hr />
</div>

Optimising Long-Term Care Placement to Reduce Service Providers Declinations  
**MATH 402W Capstone** — Team LTC × Vancouver Coastal Health

---

## Problem

When a bed opens in a Vancouver Coastal Health (VCH) LTC facility, it is offered to a client on
the priority waitlist. If the client's care provider **declines** the offer, the bed sits empty
for ~2 days before the next offer goes out. With ~18 waitlist-accessible vacancies per month and
a real backlog of ~83 clients across four priority tiers, even modest improvements in match quality
recover hundreds of empty bed-days per year.

This project builds a full pipeline — logistic deferral model → LP optimisation → discrete-event
simulation — to quantify how much better a data-driven assignment policy performs against VCH's
current cyclic FIFO discipline.

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

# 4. Generate synthetic data (run once — creates data/generated/*.csv)
PYTHONPATH=. python data/mock_data.py
```

To deactivate the venv when you're done:

```bash
deactivate
```

---

## Repository Layout

```
model/
  parameters.py           — all model constants + helper functions
  data_loader.py          — loads data/generated/*.csv (swap path for real VCH data)
  core/
    logistic.py           — builds the P×N deferral-probability matrix D
  optimisation/
    baseline.py           — first-come and cyclic-FIFO baseline policies
    solver.py             — LP relaxation (bipartite matching; TU → integer solution)
    compare.py            — runs all policies, prints comparison report
  simulation/
    queue_model.py        — bed-scarce M/G/∞ queue, three policies (10 reps)
    des.py                — full DES: + escalation, batch arrivals, geo penalty, flu calendar (30 reps)
  validation/
    validate_step2.py     — 36 data/parameter checks
    validate_step3.py     — 36 optimisation checks
    validate_step4.py     — 32 queue-model checks
    validate_step5.py     — 30 DES checks

data/
  mock_data.py            — synthetic data generator (30 facilities, ~1,350 offer events)
  generated/              — facility_details.csv, vacancies.csv, room_characteristics.csv,
                            waitlist_entry.csv

experiments/
  small_experiment.py     — 2-facility controlled experiment (hand-verifiable)
  vch_experiment.py       — VCH-scale case study (55 facilities, 30 reps)

results/
  plot_results.py         — generates all seven figures
  figures/                — fig1_calibration … fig7_summary (PNG)
```

---

## Model Overview

### Step 1 — Logistic Deferral Model (`model/core/logistic.py`)

A mixed-effects logistic regression estimates the probability that a care provider declines an
offer for client *p* at facility *n*:

```
p_pn = σ( α + u_n + β₁·r_p + β₂·q_p + β₃·h_n + β₄·(r_p × h_n) + β₅·g_pn + β₆·t )
```

| Covariate | Meaning | Effect |
|-----------|---------|--------|
| `r_p` | composite clinical complexity (CPS + ADL) | ↑ complexity → ↑ decline |
| `q_p` | priority ordinal (0–3) | ↑ urgency → ↓ decline |
| `h_n` | for-profit facility flag | slight ↑ selectivity |
| `g_pn` | gender mismatch | strong ↑ decline |
| `t`   | flu season (Nov–Jan) | +12% decline rate |
| `u_n` | facility random effect ~ N(0, 0.25) | unobserved heterogeneity |

The output is a **P × N deferral matrix D** used by both the LP solver and the simulators.  
All β values are placeholders estimated from mock data; replace with MLE on real VCH offer outcomes.

### Step 2 — Data Layer (`model/parameters.py`, `model/data_loader.py`)

Central config for all constants, calibrated to BC Office of Seniors Advocate (OSA) 2024/25:

- **λ = 28 clients/month**, **μ = 18 waitlist-accessible vacancies/month**
- **Q₀ = 90** (starting backlog); steady-state Q_ss ≈ 83 across four priority tiers
- Transfer/Site Specific calibrated so mean wait ≈ **473 days** under Cyclic FIFO (OSA 2024/25 target)
- Four tiers with tier-specific abandonment: Acute Care 50%/mo · Community Emergency 30%/mo ·
  Transfer 9.4%/mo · Community High 2%/mo

### Step 3 — LP Optimisation (`model/optimisation/`)

Batch assignment formulated as a bipartite matching LP:

```
min   Σ_p Σ_n  D[p,n] · x[p,n]

s.t.  Σ_n x[p,n]  = 1        ∀ p   (each client assigned once)
      Σ_p x[p,n]  ≤ C_n      ∀ n   (facility capacity)
      x[p,n]      = 0         if gender mismatch (hard block)
      x[p,n]      ∈ [0, 1]
```

The constraint matrix is the node-arc incidence matrix of a bipartite graph — **totally unimodular**
— so the LP relaxation always returns an integer solution. No branch-and-bound needed.

### Step 4 — Queue Model (`model/simulation/queue_model.py`)

Discrete-event simulation of the bed-scarce waitlist with three policies:

| Policy | Description |
|--------|-------------|
| **Erlang-A (FCFS)** | No declinations — lower bound on bed-days wasted |
| **Cyclic FIFO** | VCH's current cycling discipline across four tiers |
| **Optimised** | Fairness-window + minimum-p_pn client selection |

Each declined offer wastes `OFFER_TURNAROUND_DAYS = 2` empty bed-days before re-offer.

### Step 5 — DES (`model/simulation/des.py`)

Extends the queue model with four realistic features:

1. **Priority escalation** — clients auto-upgrade after waiting too long (Transfer → 90 days, CommEm → 45 days)
2. **Flu-season calendar** — month-accurate `t` flag fed to deferral model
3. **Batch arrivals** — 10% of events bring 2–4 clients (hospital discharge planning days)
4. **Geographic soft penalty** — logit penalty when client's home region ≠ facility region

---

## Running the Model

```bash
# Validate each step (all checks should pass)
PYTHONPATH=. python model/validation/validate_step2.py   # 36 checks
PYTHONPATH=. python model/validation/validate_step3.py   # 36 checks
PYTHONPATH=. python model/validation/validate_step4.py   # 32 checks
PYTHONPATH=. python model/validation/validate_step5.py   # 30 checks

# Run the LP comparison report (batch snapshot)
PYTHONPATH=. python model/optimisation/compare.py

# Run the queue simulation (10 replications, ~seconds)
PYTHONPATH=. python model/simulation/queue_model.py

# Run the full DES (30 replications, ~minutes)
PYTHONPATH=. python model/simulation/des.py

# Reproduce all figures
PYTHONPATH=. python results/plot_results.py

# Experiments
PYTHONPATH=. python experiments/small_experiment.py    # 2-facility, hand-verifiable
PYTHONPATH=. python experiments/vch_experiment.py      # VCH scale, 55 facilities
```

---

## Results

> **Note:** All results below use synthetic mock data with placeholder β coefficients.
> Magnitudes are indicative; re-run after fitting β on real VCH offer-outcome data.

### Step 3 — LP Batch Optimisation (full-information snapshot)

| Metric | Cyclic FIFO | LP Optimal | Change |
|--------|-------------|------------|--------|
| Expected declinations | ~113 | ~46 | **−59%** |

### Step 4 — Queue Model (10 reps, 3-year horizon)

| Metric | Cyclic FIFO | Optimised | Change |
|--------|-------------|-----------|--------|
| Declination rate | — | — | **−18%** |
| Bed-days wasted / year | — | ~60 saved | **−19%** |
| Mean wait (all tiers) | — | — | **−61%** |
| Acute Care mean wait | — | — | **−73%** |

### Step 5 — DES with Extensions (30 reps, 3-year horizon)

| Metric | Cyclic FIFO | Optimised | Change |
|--------|-------------|-----------|--------|
| Declination rate | — | — | **−27%** |
| Bed-days wasted / year | — | ~44 saved | **−34%** |
| Mean wait (all tiers) | — | — | **−27%** |
| Acute Care mean wait | — | — | **−65%** |
| Out-of-region placements | — | — | **−72%** |

Flu-season effect: **+12% declination rate** in November–January (captured by the seasonal `t` flag).

Transfer/Site Specific mean wait under Cyclic FIFO calibrated to **473 days** (OSA 2024/25).

### Figures

| Figure | Content |
|--------|---------|
| `fig1_calibration.png` | Deferral rate calibration — logistic curve vs mock data |
| `fig2_declination_rates.png` | Per-tier declination rates across three policies |
| `fig3_bed_days_wasted.png` | Bed-days wasted per year (with 95% CIs) |
| `fig4_wait_times.png` | Mean wait by priority tier across three policies |
| `fig5_lp_comparison.png` | LP vs baselines — expected declinations (batch snapshot) |
| `fig6_des_extensions.png` | DES extensions: escalation, flu season, geo penalty |
| `fig7_summary.png` | One-page summary: all key metrics side-by-side |

---

## Using Real VCH Data

When real data arrives, only two files change:

1. **`model/data_loader.py`** — update `_DEFAULT_DIR` to point at the real CSVs  
2. **`model/parameters.py`** — re-estimate β via logistic regression on `WaitlistOfferOutcome`,
   recalibrate `VACANCY_RATE` from actual waitlist placements per month

All downstream code (Steps 3–5) imports through `load_all()` and `model.parameters` — nothing
else needs to change.
