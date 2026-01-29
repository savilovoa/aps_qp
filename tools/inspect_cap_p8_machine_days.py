from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


TARGET_IDX = 10
OVERPROD_IDX = 8


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # plan_shifts для OVERPROD_IDX
    plan_shifts_over = int(products[OVERPROD_IDX][1])
    shifts_per_day = 3
    plan_days_over = (plan_shifts_over + shifts_per_day - 1) // shifts_per_day

    print(f"OVERPROD_IDX={OVERPROD_IDX}, name={products[OVERPROD_IDX][0]}, plan_shifts={plan_shifts_over}, plan_days≈{plan_days_over}")

    # Настройки как в cap_overprod_p8_for_p10
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300

    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = None

    settings.APPLY_PROP_OBJECTIVE = False
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False
    settings.APPLY_STRATEGY_PENALTY = False

    settings.APPLY_INDEX_UP = True

    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = TARGET_IDX
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = {OVERPROD_IDX: plan_days_over}

    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    # Сброс флагов
    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = None
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = None

    if isinstance(res, dict):
        status = res.get("status", -1)
        status_str = res.get("status_str", "")
        schedule = res.get("schedule", [])
        prod_stats = res.get("products", [])
    else:
        status = res.status
        status_str = res.status_str
        schedule = res.schedule
        prod_stats = res.products

    print(f"status={status} ({status_str})")

    # Подсчитываем machine-day count для OVERPROD_IDX напрямую по расписанию.
    machine_days_over = sum(1 for s in schedule if s["product_idx"] == OVERPROD_IDX)
    print(f"machine_days_over (product_idx={OVERPROD_IDX}) = {machine_days_over}")

    # Сравним с статистикой qty
    stats = {ps["product_idx"]: ps for ps in prod_stats}
    if OVERPROD_IDX in stats:
        fact_shifts_o = int(stats[OVERPROD_IDX]["qty"])
        print(f"stat_qty_over (shifts) = {fact_shifts_o}")
        # Покажем отношение shifts per machine-day как float
        if machine_days_over > 0:
            print(f"avg_shifts_per_machine_day_over ≈ {fact_shifts_o / machine_days_over:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
