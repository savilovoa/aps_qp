import json
import math
from pathlib import Path
from typing import Any

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.auto_relax_qty_minus import relax_qty_minus


def compute_deficit(data: dict[str, Any]) -> tuple[int, int, int, int]:
    """Run solver with APPLY_QTY_MINUS=False and estimate capacity deficit.

    Returns (deficit, machine_days_total, machine_days_work, zeros_total).
    """

    machines_raw = data["machines"]
    products_raw = data["products"]
    cleans_raw = data.get("cleans", [])
    remains = data.get("remains", [])
    count_days = int(data["count_days"])

    # Build tuples in the same format as calc_test_data_from
    machines = [
        (m["name"], m["product_idx"], m["id"], m["type"], m["remain_day"], m.get("reserve", 0))
        for m in machines_raw
    ]
    products = [
        (
            p["name"],
            p["qty"],
            p["id"],
            p["machine_type"],
            p["qty_minus"],
            p["lday"],
            p.get("src_root", -1),
            p.get("qty_minus_min", 0),
            p.get("sr", False),
            p.get("strategy", "--"),
        )
        for p in products_raw
    ]
    cleans = [(c["machine_idx"], c["day_idx"]) for c in cleans_raw]

    # Temporarily disable qty-minus constraints for this run
    settings.APPLY_QTY_MINUS = False

    result = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=count_days,
        data=data,
    )

    status = result["status_str"]
    if status not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"Solver status {status} when APPLY_QTY_MINUS=False; cannot derive deficit.")

    products_schedule = result["products"]  # list of dicts from solver_result

    # Machine stats
    M = len(machines_raw)
    D = count_days
    C = len(cleans_raw)
    machine_days_total = M * D
    machine_days_work = machine_days_total - C

    zeros_total = 0
    required = 0

    # Map product_idx -> (qty, ...)
    for ps in products_schedule:
        p_old = int(ps["product_idx"])
        qty_planned = int(ps["qty"])
        if p_old == 0:
            zeros_total = qty_planned

    # Required lower bound from input data (qty / qty_minus_min)
    for p in products_raw:
        idx = int(p["idx"])
        if idx == 0:
            continue
        qty = int(p.get("qty", 0))
        qty_minus = int(p.get("qty_minus", 0))
        qty_minus_min = int(p.get("qty_minus_min", 0))

        if qty > 0:
            if qty_minus == 0:
                required += qty
            elif qty_minus_min > 0:
                required += qty_minus_min

    prod_capacity = machine_days_work - zeros_total
    deficit = required - prod_capacity

    return deficit, machine_days_total, machine_days_work, zeros_total


def auto_relax_with_deficit(
    data: dict[str, Any],
    top_k: int = 5,
) -> tuple[dict[str, Any], int, int]:
    """Compute deficit and choose step_minus so that top-K lday*step_minus covers it.

    Returns (new_data, deficit, step_minus).
    """

    deficit, machine_days_total, machine_days_work, zeros_total = compute_deficit(data)

    if deficit <= 0:
        # No relaxation strictly required from capacity point of view.
        return data, deficit, 0

    # Collect top-K products by qty (idx>0)
    products_raw = data["products"]
    candidates = [
        p for p in products_raw
        if int(p.get("idx", 0)) > 0 and int(p.get("qty", 0)) > 0
    ]
    candidates.sort(key=lambda p: int(p.get("qty", 0)), reverse=True)
    candidates = candidates[:top_k]

    sum_lday = sum(int(p.get("lday", 0)) for p in candidates)
    if sum_lday <= 0:
        # Fallback: cannot compute reasonable step, just set 1.
        step_minus = 1
    else:
        step_minus = math.ceil(deficit / sum_lday)
        if step_minus < 1:
            step_minus = 1

    # Apply relaxation using computed step_minus
    new_data = relax_qty_minus(data, step_minus=step_minus, top_k=top_k)

    return new_data, deficit, step_minus


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Auto-relax qty_minus based on deficit estimated from a run without qty_minus."
        )
    )
    parser.add_argument(
        "input",
        help="Path to input JSON file (e.g. example/middle_test_in.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output JSON file (default: overwrite input)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Relax only top-K products by qty (default: 5)",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    with in_path.open(encoding="utf8") as f:
        data = json.load(f)

    new_data, deficit, step_minus = auto_relax_with_deficit(data, top_k=args.top_k)

    with out_path.open("w", encoding="utf8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(
        f"Auto-relax completed. deficit={deficit}, step_minus={step_minus}, "
        f"top_k={args.top_k}. Saved to {out_path}"
    )


if __name__ == "__main__":
    main()
