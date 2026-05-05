"""
model/simulation/des.py — Discrete Event Simulation for VCH LTC.

Same bed-scarce model as queue_model.py, plus four extensions
--------------------------------------------------------------
  1. Priority escalation  — auto-upgrade after waiting too long
  2. Sim calendar         — month-accurate flu-season flag
  3. Batch arrivals       — 10% of events bring 2–4 clients
  4. Geographic preference— soft logit penalty for out-of-region
  5. 30 replications      — tighter 95% CIs

Shared design with queue_model.py
----------------------------------
  500-client initial queue  |  ~18 vacancies/month  |  ~28 arrivals/month
  2-day penalty per declined offer (OFFER_TURNAROUND_DAYS)
  Steady-state queue ≈ 500 (mirrors VCH's real waitlist)

Run:  PYTHONPATH=. python model/simulation/des.py
"""

from __future__ import annotations

import heapq
import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Any

from model.parameters import (
    ARRIVAL_RATE, ABANDONMENT_RATE,
    VACANCY_RATE, INITIAL_QUEUE_SIZE, OFFER_TURNAROUND_DAYS,
    PRIORITY_FRACS, PRIORITY_LABELS, PRIORITY_MAP,
    FAIRNESS_WINDOW, FAIRNESS_TOLERANCE,
    SIM_HORIZON, WARM_UP_DAYS, MAX_OFFERS, FLU_MONTHS,
    ESCALATION_DAYS, ESCALATION_TARGET,
    BATCH_PROB, BATCH_MAX_SIZE, GEO_PENALTY, REGIONS,
    DES_N_REPLICATIONS,
    compute_r_p, compute_p_pn, funding_to_h_n,
)

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

_CPS_PROBS     = [.05, .10, .20, .25, .20, .12, .08]
_ADL_PROBS     = [.03, .08, .15, .22, .24, .18, .10]
_REGION_PROBS  = [0.25, 0.25, 0.25, 0.25]   # uniform across four VCH communities

_CYCLIC_ORDER = list(reversed(PRIORITY_LABELS))

# Event types
_ARRIVAL  = 0
_VACANCY  = 1
_RETRY    = 2
_ABANDON  = 3
_ESCALATE = 4


# ---------------------------------------------------------------------------
# Extended Client
# ---------------------------------------------------------------------------

@dataclass
class DESClient:
    cid:          int
    priority:     str
    q_p:          int
    r_p:          float
    region:       str
    arrival_time: float
    escalations:  int = 0
    abandon_time: float = math.inf


# ---------------------------------------------------------------------------
# DES Simulator
# ---------------------------------------------------------------------------

class DESSimulator:
    def __init__(
        self,
        policy:           str,
        facility_names:   list[str],
        capacities:       list[int],
        h_n_arr:          np.ndarray,
        u_n_arr:          np.ndarray,
        facility_regions: list[str],
        seed:             int = 0,
        sim_start_month:  int = 10,
    ):
        self.policy           = policy
        self.N                = len(facility_names)
        self.facility_names   = facility_names
        self.capacities       = capacities
        self.h_n_arr          = h_n_arr
        self.u_n_arr          = u_n_arr
        self.facility_regions = facility_regions
        self.sim_start_month  = sim_start_month

        total_cap = sum(capacities)
        self._fac_weights = np.array(capacities, dtype=float) / total_cap

        self.rng = np.random.default_rng(seed)
        self._cid_counter   = 0
        self._event_counter = 0

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def run(self, sim_days: int = SIM_HORIZON, warmup_days: int = WARM_UP_DAYS) -> dict:
        self._init_state(warmup_days)
        end_time = float(sim_days)

        while self.events:
            t, _, etype, data = heapq.heappop(self.events)
            if t > end_time:
                break
            self.now          = t
            self.after_warmup = (t >= warmup_days)

            if   etype == _ARRIVAL:  self._on_arrival(data)
            elif etype == _VACANCY:  self._on_vacancy(data["facility_idx"])
            elif etype == _RETRY:    self._on_retry(data["facility_idx"], data["attempt"])
            elif etype == _ABANDON:  self._on_abandon(data["cid"])
            elif etype == _ESCALATE: self._on_escalate(data["cid"])

            if self.after_warmup and int(t) % 7 == 0:
                self.queue_snapshots.append(sum(len(q) for q in self.queues.values()))

        return self._collect_metrics()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _init_state(self, warmup_days: float) -> None:
        self.now          = 0.0
        self.after_warmup = False
        self.events: list = []
        self.queue_snapshots: list[int] = []

        self.queues: dict[str, list[DESClient]] = {t: [] for t in PRIORITY_LABELS}
        self.in_queue: set[int] = set()
        self.cycle_pos = 0
        self.placement_window: deque[str] = deque(maxlen=FAIRNESS_WINDOW)

        self._reset_metrics()

        # Pre-populate queue
        for i in range(INITIAL_QUEUE_SIZE):
            wait_already = self.rng.uniform(0, 180)
            client = self._generate_client(arrival_time=-wait_already)
            self.queues[client.priority].append(client)
            self.in_queue.add(client.cid)
            t_abandon = self.rng.exponential(1.0 / ABANDONMENT_RATE[client.priority])
            client.abandon_time = t_abandon
            self._push(t_abandon, _ABANDON, {"cid": client.cid})
            esc = ESCALATION_DAYS.get(client.priority)
            if esc is not None:
                self._push(esc - wait_already, _ESCALATE, {"cid": client.cid})

        # First vacancy and arrival
        t_vac = self.rng.exponential(1.0 / VACANCY_RATE)
        n_fac = int(self.rng.choice(self.N, p=self._fac_weights))
        self._push(t_vac, _VACANCY, {"facility_idx": n_fac})
        self._push(self.rng.exponential(1.0 / ARRIVAL_RATE), _ARRIVAL, {})

    def _reset_metrics(self) -> None:
        self.m = {
            "arrivals": 0, "placements": 0, "declinations": 0,
            "abandonments": 0, "escalations": 0, "batch_events": 0,
            "bed_days_wasted": 0.0, "out_of_region": 0,
            "flu_declinations": 0, "nonflu_declinations": 0,
            "flu_placements": 0, "nonflu_placements": 0,
            "wait_times": [],
            "by_tier": {
                tier: {
                    "arrivals": 0, "placements": 0, "declinations": 0,
                    "abandonments": 0, "escalations_out": 0, "wait_times": [],
                }
                for tier in PRIORITY_LABELS
            },
        }

    def _push(self, t: float, etype: int, data: dict) -> None:
        self._event_counter += 1
        heapq.heappush(self.events, (t, self._event_counter, etype, data))

    # ------------------------------------------------------------------ #
    # Calendar helpers
    # ------------------------------------------------------------------ #

    def _current_month(self) -> int:
        months_elapsed = int(self.now // 30)
        return ((self.sim_start_month - 1 + months_elapsed) % 12) + 1

    def _is_flu_season(self) -> int:
        return 1 if self._current_month() in FLU_MONTHS else 0

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_arrival(self, _data: dict) -> None:
        # Compound Poisson: batch arrivals
        n_batch = 1
        if self.rng.random() < BATCH_PROB:
            n_batch = int(self.rng.integers(2, BATCH_MAX_SIZE + 1))
            if self.after_warmup:
                self.m["batch_events"] += 1

        for _ in range(n_batch):
            client = self._generate_client(arrival_time=self.now)
            self.queues[client.priority].append(client)
            self.in_queue.add(client.cid)

            if self.after_warmup:
                self.m["arrivals"] += 1
                self.m["by_tier"][client.priority]["arrivals"] += 1

            t_abandon = self.now + self.rng.exponential(1.0 / ABANDONMENT_RATE[client.priority])
            client.abandon_time = t_abandon
            self._push(t_abandon, _ABANDON, {"cid": client.cid})

            esc = ESCALATION_DAYS.get(client.priority)
            if esc is not None:
                self._push(self.now + esc, _ESCALATE, {"cid": client.cid})

        self._push(
            self.now + self.rng.exponential(1.0 / ARRIVAL_RATE),
            _ARRIVAL, {},
        )

    def _on_vacancy(self, facility_idx: int) -> None:
        self._attempt_fill(facility_idx, attempt=0)
        t_next = self.now + self.rng.exponential(1.0 / VACANCY_RATE)
        n_next = int(self.rng.choice(self.N, p=self._fac_weights))
        self._push(t_next, _VACANCY, {"facility_idx": n_next})

    def _on_retry(self, facility_idx: int, attempt: int) -> None:
        self._attempt_fill(facility_idx, attempt=attempt)

    def _on_abandon(self, cid: int) -> None:
        if cid not in self.in_queue:
            return
        for tier in PRIORITY_LABELS:
            for i, c in enumerate(self.queues[tier]):
                if c.cid == cid:
                    self.queues[tier].pop(i)
                    self.in_queue.discard(cid)
                    if self.after_warmup:
                        self.m["abandonments"] += 1
                        self.m["by_tier"][tier]["abandonments"] += 1
                    return

    def _on_escalate(self, cid: int) -> None:
        if cid not in self.in_queue:
            return
        for tier in PRIORITY_LABELS:
            for i, c in enumerate(self.queues[tier]):
                if c.cid == cid:
                    target = ESCALATION_TARGET.get(tier)
                    if target is None:
                        return
                    self.queues[tier].pop(i)
                    c.priority  = target
                    c.q_p       = PRIORITY_MAP[target]
                    c.escalations += 1
                    self.queues[target].append(c)

                    if self.after_warmup:
                        self.m["escalations"] += 1
                        self.m["by_tier"][tier]["escalations_out"] += 1

                    next_esc = ESCALATION_DAYS.get(target)
                    if next_esc is not None:
                        self._push(self.now + next_esc, _ESCALATE, {"cid": cid})
                    return

    # ------------------------------------------------------------------ #
    # Core offer logic
    # ------------------------------------------------------------------ #

    def _attempt_fill(self, facility_idx: int, attempt: int) -> None:
        if attempt >= MAX_OFFERS:
            return

        client = self._select_client(facility_idx)
        if client is None:
            return

        n = facility_idx
        if self.policy == "erlang_a":
            p_pn = 0.0
        else:
            t_flu   = self._is_flu_season()
            geo     = 0 if client.region == self.facility_regions[n] else 1
            u_n_adj = float(self.u_n_arr[n]) + GEO_PENALTY * geo
            p_pn    = compute_p_pn(
                client.r_p, client.q_p,
                int(self.h_n_arr[n]), t_flu, u_n_adj,
            )

        if self.rng.random() >= p_pn:
            self._place_client(client, n, p_pn)
        else:
            self.queues[client.priority].append(client)
            self.in_queue.add(client.cid)
            if self.after_warmup:
                t_flu = self._is_flu_season()
                self.m["declinations"] += 1
                self.m["bed_days_wasted"] += OFFER_TURNAROUND_DAYS
                self.m["by_tier"][client.priority]["declinations"] += 1
                if t_flu:
                    self.m["flu_declinations"] += 1
                else:
                    self.m["nonflu_declinations"] += 1
            self._push(
                self.now + OFFER_TURNAROUND_DAYS,
                _RETRY,
                {"facility_idx": n, "attempt": attempt + 1},
            )

    def _place_client(self, client: DESClient, facility_idx: int, p_pn: float) -> None:
        self.in_queue.discard(client.cid)
        t_flu = self._is_flu_season()
        geo_miss = (client.region != self.facility_regions[facility_idx])

        if self.after_warmup:
            wait = max(0.0, self.now - client.arrival_time)
            tier = client.priority
            self.m["placements"] += 1
            self.m["wait_times"].append(wait)
            self.m["by_tier"][tier]["placements"] += 1
            self.m["by_tier"][tier]["wait_times"].append(wait)
            if t_flu:
                self.m["flu_placements"] += 1
            else:
                self.m["nonflu_placements"] += 1
            if geo_miss:
                self.m["out_of_region"] += 1

        if self.policy == "cyclic_fifo":
            self.cycle_pos += 1
        if self.policy == "optimised":
            self.placement_window.append(client.priority)

    # ------------------------------------------------------------------ #
    # Client selection
    # ------------------------------------------------------------------ #

    def _select_client(self, n: int) -> DESClient | None:
        if self.policy == "erlang_a":
            return self._select_fcfs(n)
        elif self.policy == "cyclic_fifo":
            return self._select_cyclic(n)
        elif self.policy == "optimised":
            return self._select_optimised(n)
        return None

    def _eligible_in_tier(self, tier: str) -> list[DESClient]:
        return list(self.queues[tier])

    def _pop_from_tier(self, tier: str, client: DESClient) -> None:
        q = self.queues[tier]
        for i, c in enumerate(q):
            if c.cid == client.cid:
                q.pop(i)
                self.in_queue.discard(client.cid)
                return

    def _select_fcfs(self, n: int) -> DESClient | None:
        best: DESClient | None = None
        for tier in PRIORITY_LABELS:
            for c in self._eligible_in_tier(tier):
                if best is None or c.arrival_time < best.arrival_time:
                    best = c
        if best is not None:
            self._pop_from_tier(best.priority, best)
        return best

    def _select_cyclic(self, n: int) -> DESClient | None:
        for _ in range(len(_CYCLIC_ORDER)):
            tier = _CYCLIC_ORDER[self.cycle_pos % len(_CYCLIC_ORDER)]
            eligible = self._eligible_in_tier(tier)
            if eligible:
                client = min(eligible, key=lambda c: c.arrival_time)
                self._pop_from_tier(tier, client)
                return client
            self.cycle_pos += 1
        return None

    def _select_optimised(self, n: int) -> DESClient | None:
        t_flu = self._is_flu_season()
        window_list = list(self.placement_window)
        wn = len(window_list) if window_list else 1
        wc = {t: window_list.count(t) / wn for t in PRIORITY_LABELS}

        underserved = [
            t for t in PRIORITY_LABELS
            if wc[t] < (0.25 - FAIRNESS_TOLERANCE) and self.queues[t]
        ]
        eligible_tiers = underserved if underserved else [
            t for t in PRIORITY_LABELS if self.queues[t]
        ]
        if not eligible_tiers:
            return None

        candidates = []
        for tier in eligible_tiers:
            candidates.extend(self._eligible_in_tier(tier))
        if not candidates:
            return None

        def _score(c: DESClient) -> float:
            geo = 0 if c.region == self.facility_regions[n] else 1
            return compute_p_pn(
                c.r_p, c.q_p, int(self.h_n_arr[n]), t_flu,
                float(self.u_n_arr[n]) + GEO_PENALTY * geo,
            )

        best = min(candidates, key=_score)
        self._pop_from_tier(best.priority, best)
        return best

    # ------------------------------------------------------------------ #
    # Client generation
    # ------------------------------------------------------------------ #

    def _generate_client(self, arrival_time: float = 0.0) -> DESClient:
        self._cid_counter += 1
        tiers  = list(PRIORITY_FRACS.keys())
        probs  = [PRIORITY_FRACS[t] for t in tiers]
        priority = str(self.rng.choice(tiers, p=probs))
        cps    = int(self.rng.choice(range(7), p=_CPS_PROBS))
        adl    = int(self.rng.choice(range(7), p=_ADL_PROBS))
        self.rng.random()  # preserve RNG sequence (formerly gender draw)
        region = str(self.rng.choice(REGIONS, p=_REGION_PROBS))
        return DESClient(
            cid=self._cid_counter,
            priority=priority,
            q_p=PRIORITY_MAP[priority],
            r_p=compute_r_p(cps, adl),
            region=region,
            arrival_time=arrival_time,
        )

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def _collect_metrics(self) -> dict:
        m = self.m
        placements   = m["placements"]
        total_offers = placements + m["declinations"]
        declin_rate  = m["declinations"] / total_offers if total_offers > 0 else float("nan")

        wait_arr  = np.array(m["wait_times"])
        mean_wait = float(wait_arr.mean()) if len(wait_arr) > 0 else float("nan")
        bdw_per_p = m["bed_days_wasted"] / placements if placements > 0 else float("nan")

        ql_arr  = np.array(self.queue_snapshots)
        mean_ql = float(ql_arr.mean()) if len(ql_arr) > 0 else float("nan")

        flu_off    = m["flu_placements"]    + m["flu_declinations"]
        nonflu_off = m["nonflu_placements"] + m["nonflu_declinations"]
        flu_dr    = m["flu_declinations"]    / flu_off    if flu_off    > 0 else float("nan")
        nonflu_dr = m["nonflu_declinations"] / nonflu_off if nonflu_off > 0 else float("nan")
        oor_frac  = m["out_of_region"] / placements if placements > 0 else float("nan")

        by_tier = {}
        for tier in PRIORITY_LABELS:
            tm = m["by_tier"][tier]
            wt = np.array(tm["wait_times"])
            t_off = tm["placements"] + tm["declinations"]
            by_tier[tier] = {
                "placements":      tm["placements"],
                "declinations":    tm["declinations"],
                "abandonments":    tm["abandonments"],
                "escalations_out": tm["escalations_out"],
                "declin_rate":     tm["declinations"] / t_off if t_off > 0 else float("nan"),
                "mean_wait":       float(wt.mean()) if len(wt) > 0 else float("nan"),
            }

        return {
            "policy":              self.policy,
            "placements":          placements,
            "declinations":        m["declinations"],
            "abandonments":        m["abandonments"],
            "escalations":         m["escalations"],
            "batch_events":        m["batch_events"],
            "bed_days_wasted":     round(m["bed_days_wasted"], 1),
            "bed_days_per_place":  round(bdw_per_p, 3) if not math.isnan(bdw_per_p) else float("nan"),
            "declination_rate":    round(declin_rate, 4),
            "mean_wait_days":      round(mean_wait, 1) if not math.isnan(mean_wait) else float("nan"),
            "mean_queue_length":   round(mean_ql, 1) if not math.isnan(mean_ql) else float("nan"),
            "flu_declination_rate":    round(flu_dr, 4) if not math.isnan(flu_dr) else float("nan"),
            "nonflu_declination_rate": round(nonflu_dr, 4) if not math.isnan(nonflu_dr) else float("nan"),
            "out_of_region_frac":  round(oor_frac, 4) if not math.isnan(oor_frac) else float("nan"),
            "by_tier":             by_tier,
        }


# ---------------------------------------------------------------------------
# Multi-replication runner + summariser
# ---------------------------------------------------------------------------

def run_des_replications(
    policy:           str,
    facility_names:   list[str],
    capacities:       list[int],
    h_n_arr:          np.ndarray,
    u_n_arr:          np.ndarray,
    facility_regions: list[str],
    n_reps:           int = DES_N_REPLICATIONS,
    seed_base:        int = 0,
    sim_days:         int = SIM_HORIZON,
    warmup_days:      int = WARM_UP_DAYS,
    sim_start_month:  int = 10,
) -> list[dict]:
    return [
        DESSimulator(
            policy=policy,
            facility_names=facility_names,
            capacities=capacities,
            h_n_arr=h_n_arr,
            u_n_arr=u_n_arr,
            facility_regions=facility_regions,
            seed=seed_base + rep * 100,
            sim_start_month=sim_start_month,
        ).run(sim_days=sim_days, warmup_days=warmup_days)
        for rep in range(n_reps)
    ]


def summarise_des_replications(results: list[dict]) -> dict:
    def _ci(vals: list) -> dict:
        arr = np.array([v for v in vals if not (isinstance(v, float) and math.isnan(v))])
        if len(arr) == 0:
            return {"mean": float("nan"), "half_ci": float("nan")}
        mean = float(arr.mean())
        se   = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        return {
            "mean":    round(mean, 2),
            "half_ci": round(1.96 * se, 2) if not math.isnan(se) else float("nan"),
        }

    scalar_keys = [
        "placements", "declinations", "abandonments", "escalations",
        "batch_events", "bed_days_wasted", "bed_days_per_place",
        "declination_rate", "mean_wait_days", "mean_queue_length",
        "flu_declination_rate", "nonflu_declination_rate", "out_of_region_frac",
    ]
    summary: dict[str, Any] = {"policy": results[0]["policy"], "n_reps": len(results)}
    for k in scalar_keys:
        summary[k] = _ci([r[k] for r in results])

    summary["by_tier"] = {}
    for tier in PRIORITY_LABELS:
        summary["by_tier"][tier] = {}
        for tm in ["placements", "declinations", "abandonments",
                   "escalations_out", "declin_rate", "mean_wait"]:
            summary["by_tier"][tier][tm] = _ci([r["by_tier"][tier][tm] for r in results])

    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_des_report(summaries: dict[str, dict]) -> None:
    policies = list(summaries.keys())
    W_label, W_col = 34, 20
    sep  = "=" * (W_label + W_col * len(policies) + 2)
    sep2 = "-" * len(sep)

    def _fmt(stat) -> str:
        if not isinstance(stat, dict):
            return str(stat)
        m, h = stat["mean"], stat["half_ci"]
        if math.isnan(m):
            return "—"
        if math.isnan(h):
            return f"{m:.2f}"
        return f"{m:.2f} ±{h:.2f}"

    def _hdr():
        row = "  " + "".ljust(W_label)
        for p in policies:
            row += p.rjust(W_col)
        print(row)

    print("\n" + sep)
    print("  VCH LTC DES — BED-SCARCE MODEL  (30 reps, 95% CI)")
    print(sep)
    _hdr()
    print(sep2)

    for label, key in [
        ("Placements (post-warmup)",     "placements"),
        ("Declinations",                 "declinations"),
        ("Abandonments",                 "abandonments"),
        ("Priority escalations",         "escalations"),
        ("Batch arrival events",         "batch_events"),
        ("Bed-days wasted (total)",      "bed_days_wasted"),
        ("Bed-days wasted / placement",  "bed_days_per_place"),
        ("Declination rate",             "declination_rate"),
        ("Mean wait (days)",             "mean_wait_days"),
        ("Mean queue length",            "mean_queue_length"),
        ("Flu-season declin. rate",      "flu_declination_rate"),
        ("Non-flu declin. rate",         "nonflu_declination_rate"),
        ("Out-of-region fraction",       "out_of_region_frac"),
    ]:
        row = "  " + label.ljust(W_label)
        for p in policies:
            row += _fmt(summaries[p][key]).rjust(W_col)
        print(row)

    print(sep2)
    print("  Per-tier declination rate:")
    _hdr(); print(sep2)
    for tier in PRIORITY_LABELS:
        row = "  " + tier.ljust(W_label)
        for p in policies:
            row += _fmt(summaries[p]["by_tier"][tier]["declin_rate"]).rjust(W_col)
        print(row)

    print(sep2)
    print("  Per-tier mean wait (days):")
    _hdr(); print(sep2)
    for tier in PRIORITY_LABELS:
        row = "  " + tier.ljust(W_label)
        for p in policies:
            row += _fmt(summaries[p]["by_tier"][tier]["mean_wait"]).rjust(W_col)
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_des_comparison(
    inputs:      dict | None = None,
    n_reps:      int = DES_N_REPLICATIONS,
    sim_days:    int = SIM_HORIZON,
    warmup_days: int = WARM_UP_DAYS,
) -> dict[str, dict]:
    print("\n" + "=" * 65)
    print("  STEP 5 — DES (BED-SCARCE, 30 REPLICATIONS)")
    print("=" * 65)

    if inputs is None:
        from model.data_loader import load_all
        from model.core.logistic import prepare_model_inputs
        data   = load_all()
        inputs = prepare_model_inputs(
            data["waitlist_entry"],
            data["facility_details"],
            data["vacancies"],
            verbose=False,
        )

    facilities_df  = inputs["facilities"]
    facility_names = inputs["facility_names"]
    capacities     = [int(c) for c in inputs["capacities"]]
    u_n_map        = inputs["u_n_map"]

    h_n_arr = np.array([
        funding_to_h_n(facilities_df.loc[i, "funding"])
        for i in range(len(facility_names))
    ])
    u_n_arr = np.array([u_n_map.get(name, 0.0) for name in facility_names])

    facility_regions = []
    for i in range(len(facility_names)):
        r = facilities_df.loc[i, "community_region_desc"] \
            if "community_region_desc" in facilities_df.columns else "Vancouver"
        facility_regions.append(str(r))

    print(f"\n  Initial queue: {INITIAL_QUEUE_SIZE}  |  "
          f"Vacancy: {VACANCY_RATE*30:.0f}/month  |  "
          f"Arrivals: {ARRIVAL_RATE*30:.0f}/month")
    print(f"  Offer turnaround: {OFFER_TURNAROUND_DAYS}d  |  "
          f"Sim: {sim_days}d  |  Warm-up: {warmup_days}d  |  Reps: {n_reps}")
    print(f"  Extras: escalation + batch arrivals + geo penalty + flu calendar")

    summaries: dict[str, dict] = {}
    for policy in ("erlang_a", "cyclic_fifo", "optimised"):
        print(f"\n  Running '{policy}' ({n_reps} reps)...")
        reps = run_des_replications(
            policy=policy,
            facility_names=facility_names, capacities=capacities,
            h_n_arr=h_n_arr, u_n_arr=u_n_arr,
            facility_regions=facility_regions,
            n_reps=n_reps, sim_days=sim_days, warmup_days=warmup_days,
        )
        summaries[policy] = summarise_des_replications(reps)
        s = summaries[policy]
        print(f"    declin={s['declination_rate']['mean']:.3f}  "
              f"wait={s['mean_wait_days']['mean']:.0f}d  "
              f"bed_days_wasted={s['bed_days_wasted']['mean']:.0f}  "
              f"queue={s['mean_queue_length']['mean']:.0f}  "
              f"esc={s['escalations']['mean']:.0f}")

    print_des_report(summaries)
    return summaries


if __name__ == "__main__":
    run_des_comparison()
