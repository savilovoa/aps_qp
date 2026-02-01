from __future__ import annotations

from typing import Dict, Tuple, List

import pandas as pd
from ortools.sat.python import cp_model

from src.config import settings, logger


def _safe_int(val, default: int = 0) -> int:
    """Safe int conversion handling NaN/None/invalid values."""
    try:
        if val is None:
            return default
        if isinstance(val, float) and pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def build_master_model(
    products_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    count_days: int,
) -> Tuple[cp_model.CpModel, Dict[Tuple[int, int], cp_model.IntVar], Dict[int, int]]:
    """Build master CP-SAT model over x[p,m] = days of product p on machine m.

    This is a simplified prototype:
      - respects machine_type/div compatibility,
      - enforces machine capacity (sum_p x[p,m] <= count_days),
      - enforces minimal volumes from qty_minus_min (in days),
      - keeps plan_days in the objective |total_days_p - plan_days_p| with
        higher weight for strict products (qty_minus=0).
    """

    model = cp_model.CpModel()

    num_products = len(products_df)
    num_machines = len(machines_df)

    all_p = range(1, num_products)  # skip idx 0 (service product)
    all_m = range(num_machines)

    # Plan in shifts and model days (3 shifts per day)
    shifts_per_day = 3
    plan_shifts: Dict[int, int] = {
        p: _safe_int(products_df.iloc[p].get("qty", 0), 0) for p in all_p
    }
    plan_days: Dict[int, int] = {
        p: (plan_shifts[p] + shifts_per_day - 1) // shifts_per_day for p in all_p
    }

    # qty_minus and qty_minus_min
    qty_minus: Dict[int, int] = {
        p: _safe_int(products_df.iloc[p].get("qty_minus", 0), 0) for p in all_p
    }
    qty_minus_min_shifts: Dict[int, int] = {
        p: _safe_int(products_df.iloc[p].get("qty_minus_min", 0), 0) for p in all_p
    }

    # div and machine_type compatibility
    product_divs = [
        _safe_int(products_df.iloc[p].get("div", 0), 0) for p in range(num_products)
    ]
    machine_divs = [
        _safe_int(machines_df.iloc[m].get("div", 1), 1) for m in range(num_machines)
    ]

    product_types = [
        _safe_int(products_df.iloc[p].get("machine_type", 0), 0)
        for p in range(num_products)
    ]
    machine_types = [
        _safe_int(machines_df.iloc[m].get("type", 0), 0) for m in range(num_machines)
    ]

    # Decision vars x[p,m]
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for p in all_p:
        for m in all_m:
            x[p, m] = model.NewIntVar(0, count_days, f"x_{p}_{m}")

    # Type/div compatibility
    for p in all_p:
        prod_type = product_types[p]
        prod_div = product_divs[p]
        for m in all_m:
            m_type = machine_types[m]
            m_div = machine_divs[m]
            type_incompatible = prod_type > 0 and m_type != prod_type
            div_incompatible = prod_div in (1, 2) and m_div != prod_div
            if type_incompatible or div_incompatible:
                model.Add(x[p, m] == 0)

    # Machine capacity
    for m in all_m:
        model.Add(sum(x[p, m] for p in all_p) <= count_days)

    # Volume constraints and objective terms
    max_total_days = num_machines * count_days
    deviation_terms: List[cp_model.LinearExpr] = []

    for p in all_p:
        total_p = model.NewIntVar(0, max_total_days, f"tot_{p}")
        model.Add(total_p == sum(x[p, m] for m in all_m))

        plan_d = plan_days[p]
        qm_flag = qty_minus[p]
        qm_min_sh = qty_minus_min_shifts[p]
        min_days = (
            (qm_min_sh + shifts_per_day - 1) // shifts_per_day
            if qm_min_sh > 0
            else 0
        )

        # Hard lower bound only from qty_minus_min; plan_d is enforced softly.
        if min_days > 0:
            model.Add(total_p >= min_days)

        # Soft deviation from plan: |total_p - plan_d|
        dev = model.NewIntVar(-max_total_days, max_total_days, f"dev_{p}")
        model.Add(dev == total_p - plan_d)
        abs_dev = model.NewIntVar(0, max_total_days, f"absdev_{p}")
        model.AddAbsEquality(abs_dev, dev)

        w = 2 if qm_flag == 0 else 1
        deviation_terms.append(abs_dev * w)

    if deviation_terms:
        model.Minimize(sum(deviation_terms))
    else:
        model.Minimize(0)

    return model, x, plan_days


def solve_master(
    model: cp_model.CpModel,
    x: Dict[Tuple[int, int], cp_model.IntVar],
    time_limit: int,
) -> Dict[Tuple[int, int], int]:
    """Solve the master model and return x[p,m] values."""

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = settings.LOOM_NUM_WORKERS

    status = solver.Solve(model)
    logger.info(
        "TWOLEVEL MASTER status=%s objective=%.3f",
        solver.StatusName(status),
        solver.ObjectiveValue(),
    )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Two-level master model is infeasible or unknown")

    x_val: Dict[Tuple[int, int], int] = {}
    for (p, m), var in x.items():
        x_val[p, m] = solver.Value(var)
    return x_val


def build_machine_sequence(
    m_idx: int,
    x_pm: Dict[int, int],
    products_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    count_days: int,
) -> List[int]:
    """Greedy per-machine sequence for LONG_SIMPLE.

    Attempts to respect x_pm (days per product on this machine) and produce
    a non-decreasing product idx sequence (INDEX_UP-like) without zeros/cleans.
    """

    num_products = len(products_df)
    all_p = range(1, num_products)

    remaining = dict(x_pm)

    prod_type = [
        _safe_int(products_df.iloc[p].get("machine_type", 0), 0)
        for p in range(num_products)
    ]
    prod_div = [
        _safe_int(products_df.iloc[p].get("div", 0), 0)
        for p in range(num_products)
    ]
    m_type = _safe_int(machines_df.iloc[m_idx].get("type", 0), 0)
    m_div = _safe_int(machines_df.iloc[m_idx].get("div", 1), 1)

    def can_run(p: int) -> bool:
        if p <= 0 or p >= num_products:
            return False
        pt = prod_type[p]
        pd = prod_div[p]
        if pt > 0 and m_type != pt:
            return False
        if pd in (1, 2) and m_div != pd:
            return False
        return True

    seq: List[int | None] = [None for _ in range(count_days)]

    # Day 0: pick product with max remaining, if any.
    best_p = None
    best_rem = -1
    for p in all_p:
        if not can_run(p):
            continue
        if remaining.get(p, 0) > best_rem:
            best_rem = remaining[p]
            best_p = p
    if best_p is not None and best_rem > 0:
        seq[0] = best_p
        remaining[best_p] -= 1
    else:
        seq[0] = next((p for p in all_p if can_run(p)), 1)

    # Next days
    for d in range(1, count_days):
        prev_p = seq[d - 1]
        chosen = None

        # 1) Try to continue previous product if remaining
        if prev_p is not None and prev_p > 0 and can_run(prev_p):
            if remaining.get(prev_p, 0) > 0:
                chosen = prev_p

        # 2) Otherwise choose product with max remaining among p >= prev_p
        if chosen is None:
            start_idx = prev_p if (prev_p is not None and prev_p > 0) else 1
            best_p = None
            best_rem = -1
            for p in range(start_idx, num_products):
                if not can_run(p):
                    continue
                rem = remaining.get(p, 0)
                if rem > best_rem:
                    best_rem = rem
                    best_p = p
            if best_p is not None and best_rem > 0:
                chosen = best_p

        # 3) If no plan left, continue prev_p or pick first compatible product
        if chosen is None:
            if prev_p is not None and prev_p > 0 and can_run(prev_p):
                chosen = prev_p
            else:
                chosen = next((p for p in all_p if can_run(p)), 1)

        seq[d] = chosen
        if chosen is not None and chosen > 0 and remaining.get(chosen, 0) > 0:
            remaining[chosen] -= 1

    # Replace any None with first compatible product (safety)
    for d in range(count_days):
        if seq[d] is None or seq[d] <= 0:
            seq[d] = next((p for p in all_p if can_run(p)), 1)

    return [int(p) for p in seq]  # internal product idx (in products_df / products_new)


def build_twolevel_schedule(
    remains: list,
    products_new: list[tuple],
    machines_new: list[tuple],
    cleans_new: list[tuple],
    count_days: int,
    products_df_orig: pd.DataFrame,
    machines_orig: list[tuple],
    data: dict,
):
    """Build a two-level LONG_SIMPLE schedule.

    - Master level: CP-SAT over x[p,m] (days per product/machine) using
      products_df_orig and machines_orig (as DataFrames).
    - Sub level: greedy per-machine sequencing over model days.

    Returns (schedule, products_schedule, internal_obj, external_prop_penalty).
    """

    # Build DataFrames for master from original JSON data
    products_df = products_df_orig.copy().reset_index(drop=True)
    machines_df = pd.DataFrame(data["machines"]).copy().reset_index(drop=True)

    # Build and solve master model
    model, x_vars, plan_days = build_master_model(products_df, machines_df, count_days)
    x_val = solve_master(model, x_vars, time_limit=int(settings.LOOM_MAX_TIME))

    # Greedy per-machine sequences in internal idx space (of products_df)
    num_machines = len(machines_df)
    seq_by_machine_internal: List[List[int]] = []
    for m in range(num_machines):
        x_pm = {p: x_val.get((p, m), 0) for p in range(1, len(products_df))}
        seq = build_machine_sequence(m, x_pm, products_df, machines_df, count_days)
        seq_by_machine_internal.append(seq)

    # Map internal product idx (in products_df) back to original idx in products_df_orig
    # Build id -> orig_idx from products_df_orig
    id_to_orig_idx: Dict[str, int] = {}
    if "id" in products_df_orig.columns and "idx" in products_df_orig.columns:
        for _, row in products_df_orig.iterrows():
            pid = row.get("id")
            if pid is None:
                continue
            try:
                orig_idx = int(row.get("idx"))
            except Exception:
                continue
            id_to_orig_idx[str(pid)] = orig_idx

    # Build id -> internal_idx from products_df (master space)
    id_to_internal_idx: Dict[str, int] = {}
    if "id" in products_df.columns and "idx" in products_df.columns:
        for _, row in products_df.iterrows():
            pid = row.get("id")
            if pid is None:
                continue
            try:
                internal_idx = int(row.get("idx"))
            except Exception:
                continue
            id_to_internal_idx[str(pid)] = internal_idx

    # Build internal_idx -> orig_idx map via id
    internal_to_orig_idx: Dict[int, int] = {}
    for pid_str, internal_idx in id_to_internal_idx.items():
        if pid_str in id_to_orig_idx:
            internal_to_orig_idx[internal_idx] = id_to_orig_idx[pid_str]

    # Build schedule list of dicts in original index space
    schedule: List[dict] = []

    # Map internal machine index back to original machine_idx via id
    internal_to_orig_machine_idx: Dict[int, int] = {}
    machines_new_ids = [m[2] for m in machines_new]
    machines_orig_ids = [m[2] for m in machines_orig]
    for new_idx, mid in enumerate(machines_new_ids):
        try:
            orig_idx = next(i for i, m in enumerate(machines_orig_ids) if m == mid)
        except StopIteration:
            orig_idx = new_idx
        internal_to_orig_machine_idx[new_idx] = orig_idx

    # We build schedule over model days [0..count_days-1]; solver_result will
    # interpret LONG_SIMPLE days as 3 shifts per day.
    for m_internal, seq in enumerate(seq_by_machine_internal):
        m_old = internal_to_orig_machine_idx.get(m_internal, m_internal)
        for d in range(count_days):
            p_internal = seq[d]
            p_orig = internal_to_orig_idx.get(p_internal, 0)
            schedule.append(
                {
                    "machine_idx": m_old,
                    "day_idx": d,  # model day index (will be expanded to shifts outside)
                    "product_idx": p_orig,
                    "days_in_batch": None,
                    "prev_lday": None,
                }
            )

    # Build products_schedule (per-product stats) in original idx space
    shifts_per_day = 3
    # Plan in shifts from original data
    plan_by_orig: Dict[int, int] = {}
    if "idx" in products_df_orig.columns and "qty" in products_df_orig.columns:
        for _, row in products_df_orig.iterrows():
            try:
                idx_orig = int(row.get("idx"))
            except Exception:
                continue
            plan_by_orig[idx_orig] = _safe_int(row.get("qty", 0), 0)

    # Fact in shifts from sequences (internal -> orig)
    fact_by_orig: Dict[int, int] = {k: 0 for k in plan_by_orig.keys()}
    for m_internal, seq in enumerate(seq_by_machine_internal):
        for d, p_internal in enumerate(seq):
            p_orig = internal_to_orig_idx.get(p_internal)
            if p_orig is None or p_orig not in fact_by_orig:
                continue
            fact_by_orig[p_orig] += shifts_per_day

    # External proportional penalties
    total_plan = sum(v for v in plan_by_orig.values() if v > 0)
    total_fact = sum(
        fact_by_orig[idx] for idx, v in plan_by_orig.items() if v > 0
    )
    external_penalty = 0
    per_prod_penalty: Dict[int, int] = {}
    if total_plan > 0 and total_fact > 0:
        for idx_orig, p_plan in plan_by_orig.items():
            if p_plan <= 0:
                continue
            f = fact_by_orig.get(idx_orig, 0)
            term1 = f * total_plan
            term2 = total_fact * p_plan
            contrib = abs(term1 - term2)
            per_prod_penalty[idx_orig] = contrib
            external_penalty += contrib

    products_schedule: List[dict] = []
    for idx_orig, p_plan in plan_by_orig.items():
        p_fact = fact_by_orig.get(idx_orig, 0)
        pen = per_prod_penalty.get(idx_orig, 0)
        products_schedule.append(
            {
                "product_idx": idx_orig,
                "plan_qty": p_plan,
                "qty": p_fact,
                "penalty": pen,
                "penalty_strategy": 0,
                "machines_start": 0,
                "machines_end": 0,
            }
        )

    # Internal objective (sum |total_days - plan_days|) is not returned from master
    internal_obj = 0

    return schedule, products_schedule, internal_obj, external_penalty
