"""
model/simulation/queue_model.py — Realistic bed-scarce LTC queue for VCH.

Real scenario modelled
----------------------
  500 clients on the waitlist at t=0 (VCH's reported backlog)
  ~26 beds become available to the waitlist each month  (<  28 arrivals → queue grows)
  Each declined offer wastes OFFER_TURNAROUND_DAYS = 2 days of empty bed time

  This is the core bottleneck VCH faces: when a bed finally opens,
  a poor match (high p_pn) burns 2 extra days per declination before
  the bed gets filled. Multiply by hundreds of placements per year
  and you lose hundreds of bed-days annually.

Three policies compared
-----------------------
  erlang_a    — FCFS, no declinations (lower bound: 0 bed-days wasted)
  cyclic_fifo — VCH's cycling discipline, real p_pn (Policy 1)
  optimised   — Fairness-window + min-p_pn selection, real p_pn (Policy 2)

Key metrics
-----------
  bed_days_wasted   — total empty-bed days due to declinations
  mean_wait_days    — average days from arrival to placement
  queue_length_mean — average backlog (should mirror VCH's ~500)
  declination_rate  — declined offers / total offers

Run:  PYTHONPATH=. python model/simulation/queue_model.py
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
    SIM_HORIZON, WARM_UP_DAYS, N_REPLICATIONS, MAX_OFFERS,
    FLU_MONTHS, compute_r_p, compute_p_pn, funding_to_h_n,
)

# ---------------------------------------------------------------------------
# Client attribute distributions
# ---------------------------------------------------------------------------

_CPS_PROBS = [.05, .10, .20, .25, .20, .12, .08]   # CPS 0–6
_ADL_PROBS = [.03, .08, .15, .22, .24, .18, .10]   # ADL 0–6

_CYCLIC_ORDER = list(reversed(PRIORITY_LABELS))   # Acute Care first

# Event types
_ARRIVAL = 0
_VACANCY = 1   # a bed opens at a facility
_RETRY   = 2   # reoffer after a decline (fired after OFFER_TURNAROUND_DAYS)
_ABANDON = 3


# ---------------------------------------------------------------------------
# Client dataclass
# ---------------------------------------------------------------------------

@dataclass
class Client:
    cid:          int
    priority:     str
    q_p:          int
    r_p:          float
    arrival_time: float
    n_offers:     int = 0
    abandon_time: float = math.inf


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class QueueSimulator:
    """
    Bed-scarce LTC queue simulation.

    Parameters
    ----------
    policy          : "erlang_a" | "cyclic_fifo" | "optimised"
    facility_names  : list[str] length N
    capacities      : list[int] length N  (used for vacancy weighting)
    h_n_arr         : np.ndarray(N) — for-profit flags
    u_n_arr         : np.ndarray(N) — facility random effects
    seed            : int
    """

    def __init__(
        self,
        policy:         str,
        facility_names: list[str],
        capacities:     list[int],
        h_n_arr:        np.ndarray,
        u_n_arr:        np.ndarray,
        seed:           int = 0,
    ):
        self.policy         = policy
        self.N              = len(facility_names)
        self.facility_names = facility_names
        self.capacities     = capacities
        self.h_n_arr        = h_n_arr
        self.u_n_arr        = u_n_arr

        total_cap = sum(capacities)
        # Vacancy weight per facility: proportional to capacity
        self._fac_weights = np.array(capacities, dtype=float) / total_cap

        self.rng = np.random.default_rng(seed)
        self._cid_counter   = 0
        self._event_counter = 0

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def run(
        self,
        sim_days:    int = SIM_HORIZON,
        warmup_days: int = WARM_UP_DAYS,
    ) -> dict:
        self._init_state(warmup_days)
        end_time = float(sim_days)

        while self.events:
            t, _, etype, data = heapq.heappop(self.events)
            if t > end_time:
                break
            self.now          = t
            self.after_warmup = (t >= warmup_days)

            if   etype == _ARRIVAL: self._on_arrival()
            elif etype == _VACANCY: self._on_vacancy(data["facility_idx"])
            elif etype == _RETRY:   self._on_retry(data["facility_idx"], data["attempt"])
            elif etype == _ABANDON: self._on_abandon(data["cid"])

            # Snapshot queue length periodically for mean calculation
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

        # Per-tier queues (lists for flexible selection)
        self.queues: dict[str, list[Client]] = {t: [] for t in PRIORITY_LABELS}
        self.in_queue: set[int] = set()

        # Cycling state
        self.cycle_pos = 0
        self.placement_window: deque[str] = deque(maxlen=FAIRNESS_WINDOW)

        # Metrics
        self._reset_metrics()

        # --- Pre-populate queue with INITIAL_QUEUE_SIZE clients ---
        # These represent VCH's current ~500-person waitlist.
        # Assign negative arrival times (they've been waiting before t=0).
        # Spread them out: longest-waiting arrived ~SIM_HORIZON/2 days ago.
        for i in range(INITIAL_QUEUE_SIZE):
            wait_already = self.rng.uniform(0, 180)   # been waiting 0–180 days
            client = self._generate_client(arrival_time=-wait_already)
            self.queues[client.priority].append(client)
            self.in_queue.add(client.cid)
            # Schedule abandonment from now (not from arrival_time)
            t_abandon = self.rng.exponential(1.0 / ABANDONMENT_RATE[client.priority])
            client.abandon_time = t_abandon
            self._push(t_abandon, _ABANDON, {"cid": client.cid})

        # --- Schedule first vacancy ---
        t_vac = self.rng.exponential(1.0 / VACANCY_RATE)
        n_fac = int(self.rng.choice(self.N, p=self._fac_weights))
        self._push(t_vac, _VACANCY, {"facility_idx": n_fac})

        # --- Schedule first new arrival ---
        t_arr = self.rng.exponential(1.0 / ARRIVAL_RATE)
        self._push(t_arr, _ARRIVAL, {})

    def _reset_metrics(self) -> None:
        self.m = {
            "arrivals":       0,
            "placements":     0,
            "declinations":   0,
            "abandonments":   0,
            "bed_days_wasted": 0.0,
            "wait_times":     [],
            "by_tier": {
                tier: {
                    "placements": 0, "declinations": 0,
                    "abandonments": 0, "wait_times": [],
                }
                for tier in PRIORITY_LABELS
            },
        }

    def _push(self, t: float, etype: int, data: dict) -> None:
        self._event_counter += 1
        heapq.heappush(self.events, (t, self._event_counter, etype, data))

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_arrival(self) -> None:
        client = self._generate_client(arrival_time=self.now)
        self.queues[client.priority].append(client)
        self.in_queue.add(client.cid)

        if self.after_warmup:
            self.m["arrivals"] += 1

        # Abandonment timer
        t_abandon = self.now + self.rng.exponential(1.0 / ABANDONMENT_RATE[client.priority])
        client.abandon_time = t_abandon
        self._push(t_abandon, _ABANDON, {"cid": client.cid})

        # Schedule next arrival
        self._push(
            self.now + self.rng.exponential(1.0 / ARRIVAL_RATE),
            _ARRIVAL, {},
        )

    def _on_vacancy(self, facility_idx: int) -> None:
        """A bed has opened at facility_idx. Begin the offer process."""
        self._attempt_fill(facility_idx, attempt=0)

        # Schedule the NEXT vacancy at any facility (independent of this one)
        t_next = self.now + self.rng.exponential(1.0 / VACANCY_RATE)
        n_next = int(self.rng.choice(self.N, p=self._fac_weights))
        self._push(t_next, _VACANCY, {"facility_idx": n_next})

    def _on_retry(self, facility_idx: int, attempt: int) -> None:
        """Re-offer a vacancy after a prior declination."""
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

    # ------------------------------------------------------------------ #
    # Core offer logic
    # ------------------------------------------------------------------ #

    def _attempt_fill(self, facility_idx: int, attempt: int) -> None:
        """
        Try to fill a vacancy at facility_idx on offer attempt #attempt.
        If declined → waste OFFER_TURNAROUND_DAYS, schedule retry.
        If queue empty or MAX_OFFERS exceeded → vacancy unfilled.
        """
        if attempt >= MAX_OFFERS:
            return   # give up on this vacancy

        client = self._select_client(facility_idx)
        if client is None:
            return   # queue empty or no eligible client

        # Compute p_pn
        if self.policy == "erlang_a":
            p_pn = 0.0
        else:
            month = int((self.now % 365) // 30) + 1
            t_flu = 1 if month in FLU_MONTHS else 0
            p_pn  = compute_p_pn(
                client.r_p, client.q_p,
                int(self.h_n_arr[facility_idx]), t_flu,
                float(self.u_n_arr[facility_idx]),
            )

        if self.rng.random() >= p_pn:
            # --- Accepted ---
            self._place_client(client, facility_idx)
        else:
            # --- Declined ---
            # Return client to queue; bed sits empty for OFFER_TURNAROUND_DAYS
            self.queues[client.priority].append(client)
            self.in_queue.add(client.cid)

            if self.after_warmup:
                self.m["declinations"] += 1
                self.m["bed_days_wasted"] += OFFER_TURNAROUND_DAYS
                self.m["by_tier"][client.priority]["declinations"] += 1

            # Schedule next offer attempt after the turnaround gap
            self._push(
                self.now + OFFER_TURNAROUND_DAYS,
                _RETRY,
                {"facility_idx": facility_idx, "attempt": attempt + 1},
            )

    def _place_client(self, client: Client, facility_idx: int) -> None:
        self.in_queue.discard(client.cid)
        if self.after_warmup:
            wait = max(0.0, self.now - client.arrival_time)
            tier = client.priority
            self.m["placements"] += 1
            self.m["wait_times"].append(wait)
            self.m["by_tier"][tier]["placements"] += 1
            self.m["by_tier"][tier]["wait_times"].append(wait)

        if self.policy == "cyclic_fifo":
            self.cycle_pos += 1
        if self.policy == "optimised":
            self.placement_window.append(client.priority)

    # ------------------------------------------------------------------ #
    # Client selection
    # ------------------------------------------------------------------ #

    def _select_client(self, facility_idx: int) -> Client | None:
        if self.policy == "erlang_a":
            return self._select_fcfs(facility_idx)
        elif self.policy == "cyclic_fifo":
            return self._select_cyclic(facility_idx)
        elif self.policy == "optimised":
            return self._select_optimised(facility_idx)
        return None

    def _eligible_in_tier(self, tier: str) -> list[Client]:
        return list(self.queues[tier])

    def _pop_from_tier(self, tier: str, client: Client) -> None:
        q = self.queues[tier]
        for i, c in enumerate(q):
            if c.cid == client.cid:
                q.pop(i)
                self.in_queue.discard(client.cid)
                return

    def _select_fcfs(self, n: int) -> Client | None:
        """Longest-waiting client across all tiers."""
        best: Client | None = None
        for tier in PRIORITY_LABELS:
            for c in self._eligible_in_tier(tier):
                if best is None or c.arrival_time < best.arrival_time:
                    best = c
        if best is not None:
            self._pop_from_tier(best.priority, best)
        return best

    def _select_cyclic(self, n: int) -> Client | None:
        """Rotate through tiers; FIFO within tier.

        cycle_pos advances by exactly 1 per placement (in _place_client).
        Skipping empty tiers here must NOT advance cycle_pos, otherwise
        a single skip burns two positions and over-serves the following tier.
        """
        for i in range(len(_CYCLIC_ORDER)):
            tier = _CYCLIC_ORDER[(self.cycle_pos + i) % len(_CYCLIC_ORDER)]
            eligible = self._eligible_in_tier(tier)
            if eligible:
                client = min(eligible, key=lambda c: c.arrival_time)
                self._pop_from_tier(tier, client)
                return client
        return None

    def _select_optimised(self, n: int) -> Client | None:
        """Fairness window + pick lowest-p_pn from eligible tiers."""
        month = int((self.now % 365) // 30) + 1
        t_flu = 1 if month in FLU_MONTHS else 0

        window_list = list(self.placement_window)
        wn = len(window_list) if window_list else 1
        window_counts = {t: window_list.count(t) / wn for t in PRIORITY_LABELS}

        underserved = [
            t for t in PRIORITY_LABELS
            if window_counts[t] < (0.25 - FAIRNESS_TOLERANCE)
            and self.queues[t]
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

        def _score(c: Client) -> float:
            return compute_p_pn(
                c.r_p, c.q_p, int(self.h_n_arr[n]), t_flu,
                float(self.u_n_arr[n]),
            )

        best = min(candidates, key=_score)
        self._pop_from_tier(best.priority, best)
        return best

    # ------------------------------------------------------------------ #
    # Client generation
    # ------------------------------------------------------------------ #

    def _generate_client(self, arrival_time: float = 0.0) -> Client:
        self._cid_counter += 1
        tiers = list(PRIORITY_FRACS.keys())
        probs = [PRIORITY_FRACS[t] for t in tiers]
        priority = str(self.rng.choice(tiers, p=probs))
        cps = int(self.rng.choice(range(7), p=_CPS_PROBS))
        adl = int(self.rng.choice(range(7), p=_ADL_PROBS))
        self.rng.random()  # preserve RNG sequence (formerly gender draw)
        return Client(
            cid=self._cid_counter,
            priority=priority,
            q_p=PRIORITY_MAP[priority],
            r_p=compute_r_p(cps, adl),
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

        # Bed-days wasted per placement
        bdw_per_place = (
            m["bed_days_wasted"] / placements if placements > 0 else float("nan")
        )

        ql_arr = np.array(self.queue_snapshots)
        mean_ql = float(ql_arr.mean()) if len(ql_arr) > 0 else float("nan")

        by_tier = {}
        for tier in PRIORITY_LABELS:
            tm = m["by_tier"][tier]
            wt = np.array(tm["wait_times"])
            t_off = tm["placements"] + tm["declinations"]
            by_tier[tier] = {
                "placements":   tm["placements"],
                "declinations": tm["declinations"],
                "abandonments": tm["abandonments"],
                "declin_rate":  tm["declinations"] / t_off if t_off > 0 else float("nan"),
                "mean_wait":    float(wt.mean()) if len(wt) > 0 else float("nan"),
            }

        return {
            "policy":               self.policy,
            "placements":           placements,
            "declinations":         m["declinations"],
            "abandonments":         m["abandonments"],
            "bed_days_wasted":      round(m["bed_days_wasted"], 1),
            "bed_days_per_place":   round(bdw_per_place, 3) if not math.isnan(bdw_per_place) else float("nan"),
            "declination_rate":     round(declin_rate, 4),
            "mean_wait_days":       round(mean_wait, 1) if not math.isnan(mean_wait) else float("nan"),
            "mean_queue_length":    round(mean_ql, 1) if not math.isnan(mean_ql) else float("nan"),
            "by_tier":              by_tier,
        }


# ---------------------------------------------------------------------------
# Multi-replication runner + summariser
# ---------------------------------------------------------------------------

def run_replications(
    policy:         str,
    facility_names: list[str],
    capacities:     list[int],
    h_n_arr:        np.ndarray,
    u_n_arr:        np.ndarray,
    n_reps:         int = N_REPLICATIONS,
    seed_base:      int = 0,
    sim_days:       int = SIM_HORIZON,
    warmup_days:    int = WARM_UP_DAYS,
) -> list[dict]:
    results = []
    for rep in range(n_reps):
        sim = QueueSimulator(
            policy=policy,
            facility_names=facility_names,
            capacities=capacities,
            h_n_arr=h_n_arr,
            u_n_arr=u_n_arr,
            seed=seed_base + rep * 100,
        )
        results.append(sim.run(sim_days=sim_days, warmup_days=warmup_days))
    return results


def summarise_replications(results: list[dict]) -> dict:
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
        "placements", "declinations", "abandonments",
        "bed_days_wasted", "bed_days_per_place",
        "declination_rate", "mean_wait_days", "mean_queue_length",
    ]
    summary: dict[str, Any] = {"policy": results[0]["policy"], "n_reps": len(results)}
    for k in scalar_keys:
        summary[k] = _ci([r[k] for r in results])

    summary["by_tier"] = {}
    for tier in PRIORITY_LABELS:
        summary["by_tier"][tier] = {}
        for tm in ["placements", "declinations", "abandonments", "declin_rate", "mean_wait"]:
            summary["by_tier"][tier][tm] = _ci([r["by_tier"][tier][tm] for r in results])

    return summary


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_queue_report(summaries: dict[str, dict]) -> None:
    policies = list(summaries.keys())
    W_label  = 32
    W_col    = 20
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
    print("  VCH LTC — BED-SCARCE QUEUE MODEL  (10 reps, 95% CI)")
    print(sep)
    _hdr()
    print(sep2)

    specs = [
        ("Placements (post-warmup)",    "placements"),
        ("Declinations",                "declinations"),
        ("Abandonments",                "abandonments"),
        ("Bed-days wasted (total)",     "bed_days_wasted"),
        ("Bed-days wasted / placement", "bed_days_per_place"),
        ("Declination rate",            "declination_rate"),
        ("Mean wait (days)",            "mean_wait_days"),
        ("Mean queue length (clients)", "mean_queue_length"),
    ]
    for label, key in specs:
        row = "  " + label.ljust(W_label)
        for p in policies:
            row += _fmt(summaries[p][key]).rjust(W_col)
        print(row)

    print(sep2)
    print("  Per-tier declination rate:")
    _hdr()
    print(sep2)
    for tier in PRIORITY_LABELS:
        row = "  " + tier.ljust(W_label)
        for p in policies:
            row += _fmt(summaries[p]["by_tier"][tier]["declin_rate"]).rjust(W_col)
        print(row)

    print(sep2)
    print("  Per-tier mean wait (days):")
    _hdr()
    print(sep2)
    for tier in PRIORITY_LABELS:
        row = "  " + tier.ljust(W_label)
        for p in policies:
            row += _fmt(summaries[p]["by_tier"][tier]["mean_wait"]).rjust(W_col)
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_queue_comparison(
    inputs:      dict | None = None,
    n_reps:      int = N_REPLICATIONS,
    sim_days:    int = SIM_HORIZON,
    warmup_days: int = WARM_UP_DAYS,
) -> dict[str, dict]:
    print("\n" + "=" * 60)
    print("  STEP 4 — BED-SCARCE QUEUE SIMULATION")
    print("=" * 60)

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

    print(f"\n  Initial queue: {INITIAL_QUEUE_SIZE} clients  |  "
          f"Vacancy rate: {VACANCY_RATE*30:.0f}/month  |  "
          f"Arrival rate: {ARRIVAL_RATE*30:.0f}/month")
    print(f"  Offer turnaround: {OFFER_TURNAROUND_DAYS} days/decline  |  "
          f"Sim: {sim_days}d  |  Warm-up: {warmup_days}d  |  Reps: {n_reps}")

    kwargs = dict(
        facility_names=facility_names, capacities=capacities,
        h_n_arr=h_n_arr, u_n_arr=u_n_arr,
        n_reps=n_reps, sim_days=sim_days, warmup_days=warmup_days,
    )

    summaries: dict[str, dict] = {}
    for policy in ("erlang_a", "cyclic_fifo", "optimised"):
        print(f"\n  Running '{policy}' ({n_reps} reps)...")
        reps = run_replications(policy=policy, **kwargs)
        summaries[policy] = summarise_replications(reps)
        s = summaries[policy]
        print(f"    placements={s['placements']['mean']:.0f}  "
              f"declin_rate={s['declination_rate']['mean']:.3f}  "
              f"mean_wait={s['mean_wait_days']['mean']:.0f}d  "
              f"bed_days_wasted={s['bed_days_wasted']['mean']:.0f}  "
              f"queue_len={s['mean_queue_length']['mean']:.0f}")

    print_queue_report(summaries)
    return summaries


if __name__ == "__main__":
    run_queue_comparison()
