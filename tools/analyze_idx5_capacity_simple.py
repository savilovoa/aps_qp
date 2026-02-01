from __future__ import annotations

from pathlib import Path

from src.config import settings, logger
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def run_long_simple_no_qty_minus(data, machines, products, cleans, remains):
    """Базовый прогон LONG_SIMPLE без qty_minus, LS_OVER-режим."""
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = False
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False
    settings.APPLY_INDEX_UP = True
    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = None

    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    if isinstance(res, dict):
        status = res.get("status")
        status_str = res.get("status_str")
        schedule = res.get("schedule", [])
        prod_stats = res.get("products", [])
    else:
        status = res.status
        status_str = res.status_str
        schedule = res.schedule
        prod_stats = res.products

    return status, status_str, schedule, prod_stats


def extract_fact_days_for_idx(prod_stats, target_idx: int) -> tuple[int, int]:
    """Вернуть (plan, fact) по product_idx == target_idx из prod_stats."""
    for ps in prod_stats:
        try:
            if int(ps.get("product_idx")) == target_idx:
                plan = int(ps.get("plan_qty", 0))
                fact = int(ps.get("qty", 0))
                return plan, fact
        except Exception:
            continue
    return 0, 0


def run_max_capacity_for_idx(data, machines, products, cleans, remains, target_idx: int):
    """Запуск LONG_SIMPLE без qty_minus с целью Maximize(product_counts[target_idx])."""
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = False
    settings.APPLY_PROP_OBJECTIVE = False
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False
    settings.APPLY_INDEX_UP = True
    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = target_idx

    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    if isinstance(res, dict):
        status = res.get("status")
        status_str = res.get("status_str")
        schedule = res.get("schedule", [])
        prod_stats = res.get("products", [])
    else:
        status = res.status
        status_str = res.status_str
        schedule = res.schedule
        prod_stats = res.products

    return status, status_str, schedule, prod_stats


def main() -> None:
    data, machines, products, cleans, remains = load_data()
    target_idx = 5

    print("=== LONG_SIMPLE без qty_minus (LS_OVER) ===")
    status, status_str, schedule, prod_stats = run_long_simple_no_qty_minus(
        data, machines, products, cleans, remains
    )
    print(f"status={status} ({status_str})")
    plan, fact = extract_fact_days_for_idx(prod_stats, target_idx)
    print(f"idx={target_idx}: plan_days≈{plan}, fact_days={fact} (без qty_minus)")

    print("\n=== LONG_SIMPLE Maximize(product_counts[5]) без qty_minus ===")
    status2, status_str2, schedule2, prod_stats2 = run_max_capacity_for_idx(
        data, machines, products, cleans, remains, target_idx
    )
    print(f"status={status2} ({status_str2})")
    plan2, fact2 = extract_fact_days_for_idx(prod_stats2, target_idx)
    print(f"idx={target_idx}: plan_days≈{plan2}, fact_days_max={fact2} (максимальная ёмкость)")


if __name__ == "__main__":  # pragma: no cover
    main()
