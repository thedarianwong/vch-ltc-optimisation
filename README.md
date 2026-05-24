# Optimising Long-Term Care Placement to Reduce Service Provider Declinations
 
**MATH 402W Capstone · Simon Fraser University · Spring 2026**
 
---
 
### 🏆 Recognition
 
> **Top 5 National Finalist — 2026 CORS Undergraduate Paper Competition (All Canada 🇨🇦)**
>
> Selected as a **top 5 finalist nationally** by the [Canadian Operational Research Society](https://cors.ca/). Hosted at CORS 2026 in Kingston, Ontario (June 8–10, 2026).
>
> 📄 [Research Paper (PDF)](https://drive.google.com/file/d/1dIwS2VFXPDLI-SDxqwhn-Y01DX2q2IWt/view?usp=sharing) · 📰 [LinkedIn Announcement](https://www.linkedin.com/feed/update/urn:li:activity:7464161139032215552/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEOwE3MB72WZTo_C4_V7ooMBFgUZFf-xZTo)
 
<img width="512" height="768" alt="CORS 2026 Undergraduate Paper Competition Finalists" src="https://github.com/user-attachments/assets/7a180fc3-a454-42cd-96ff-b55cae9275df" />

---
 
## Overview
 
When a bed opens in a VCH long-term care facility, it is offered to a client on the priority waitlist. If the care provider **declines** the offer, the bed sits empty for ~2 days before the next offer goes out. This project builds a data-driven assignment policy — a fairness-window selector guided by predicted declination probabilities — and evaluates it against VCH's current Cyclic FIFO discipline using queueing analysis and discrete-event simulation.
 
**All paper results come from the three scripts in `experiments/`.** The `model/` folder is the supporting library those scripts depend on.
 
---

## Quickstart

```bash
git clone <repo-url>
cd vch-ltc-optimisation

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the Experiments

These three scripts reproduce every number in the paper. Run them from the repo root with the venv active.

### 1. Small-Scale Validation (Step 4)
2 facilities · 22 beds · λ = 3/month · μ = 2/month · 30 reps · ~30 seconds

```bash
PYTHONPATH=. python experiments/small_experiment.py
```

Validates model behaviour at a transparent, hand-checkable scale before applying VCH parameters.

| Metric | Cyclic FIFO | Optimised | Change |
|--------|-------------|-----------|--------|
| Declination rate | 0.200 | 0.170 | **−19%** |
| Bed-loss days | 21.3 | 17.2 | **−19%** |
| Mean wait | 81.1 d | 57.2 d | **−29%** |
| Acute Care wait | 46.8 d | 22.8 d | **−51%** |

---

### 2. VCH Case Study — Queue Model (Step 4)
55 facilities · ~5,300 beds · λ = 28/month · μ = 18/month · 30 reps · ~2 minutes

```bash
PYTHONPATH=. python experiments/vch_experiment.py
```

Calibrated to BC Office of Seniors Advocate 2024/25 data.

| Metric | Cyclic FIFO | Optimised | Change |
|--------|-------------|-----------|--------|
| Declination rate | 0.180 | 0.160 | **−14%** |
| Bed-loss days | ~243 | ~210 | **−14%** |
| Acute Care wait | 38.0 d | 5.9 d | **−85%** |

---

### 3. VCH Case Study — Discrete Event Simulation (Step 5)
55 facilities · same calibration + 4 extensions · 30 reps · ~5 minutes

```bash
PYTHONPATH=. python experiments/des_experiment.py
```

Adds priority escalation, batch arrivals, flu-season calendar, and geographic soft penalty.

| Metric | Cyclic FIFO | Optimised | Change |
|--------|-------------|-----------|--------|
| Declination rate | 0.210 | 0.160 | **−24%** |
| Bed-loss days | ~291 | ~204 | **−30%** |
| Acute Care wait | 53.2 d | 17.3 d | **−67%** |
| In-region placement | 31% | 78% | **+152%** |

---

## How the Model Works

The modelling pipeline has five steps. Steps 1–3 produce the declination probabilities and optimal static assignment used to calibrate the simulations. Steps 4–5 are what the experiments run.

```
Step 1  model/core/logistic.py        Logistic model — estimates p_pn for each client-facility pair
Step 2  model/optimisation/solver.py  BIP — minimises expected declinations over a static snapshot
Step 3  model/optimisation/           Fairness Window — makes the BIP dynamic (online policy)
Step 4  model/simulation/queue_model.py   Queue simulation — 3-year horizon, tier-specific abandonment
Step 5  model/simulation/des.py           DES — adds escalation, batch arrivals, flu, geography
```

### Declination probability model

```
p_pn = σ( α + u_n + β₁·r_p + β₂·h_n + β₃·t )
```

| Term | Meaning |
|------|---------|
| `α = −1.8` | Baseline log-odds (~14% base declination rate) |
| `u_n ~ N(0, 0.25)` | Facility random effect (unobserved heterogeneity) |
| `β₁·r_p` | Client complexity from CPS + ADL scores (+0.9) |
| `β₂·h_n` | Facility staffing / for-profit flag (+0.1) |
| `β₃·t` | Flu-season indicator Nov–Jan (+0.15) |

All β values are placeholders calibrated to synthetic data mirroring VCH's PARIS system. Replace with MLE estimates once real offer-outcome data is available.

### BIP result (Step 2)

On a VCH snapshot of 500 clients across 30 facilities, expected declinations fall from **113.5 → 46.4** — a theoretical **−59%** reduction over Cyclic FIFO.

---

## Repository Layout

```
experiments/
  small_experiment.py     — 2-facility controlled validation (paper Section 5.1)
  vch_experiment.py       — VCH-scale queue model (paper Section 5.2)
  des_experiment.py       — VCH-scale DES with extensions (paper Section 5.3)

model/
  parameters.py           — all constants, β coefficients, helper functions
  data_loader.py          — loads data/generated/*.csv
  core/
    logistic.py           — builds the P×N deferral matrix D (used by BIP)
  optimisation/
    solver.py             — LP/BIP solver (bipartite matching, TU → integer)
    baseline.py           — Cyclic FIFO and random baseline implementations
    compare.py            — batch comparison runner for Steps 1–3
  simulation/
    queue_model.py        — bed-scarce queue simulator (used by small + vch experiments)
    des.py                — extended DES simulator (used by des_experiment)
  validation/             — unit/integration test suites for each step

data/
  mock_data.py            — synthetic data generator
  generated/              — facility_details.csv, vacancies.csv, waitlist_entry.csv, …

results/
  plot_results.py         — generates all paper figures
  figures/                — fig1_calibration … fig7_summary (PNG)
```

---

## Updating with Real VCH Data

Only two files need to change:

1. **`model/data_loader.py`** — point `_DEFAULT_DIR` at the real CSVs
2. **`model/parameters.py`** — re-fit β via logistic regression on `WaitlistOfferOutcome`, recalibrate `VACANCY_RATE` from actual monthly placements

Everything else: the BIP, the simulators, and the experiments, imports through those two files and requires no other changes.
