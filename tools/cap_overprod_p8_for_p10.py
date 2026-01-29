from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


TARGET_IDX = 10  # ст10417RSt
OVERPROD_IDX = 8  # ст78426t П, 0, 6, П2 — сильное перепроизводство в LS_PROP без qty_minus


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # plan_shifts для OVERPROD_IDX берём из входных данных (qty в сменах)
    plan_shifts_over = int(products[OVERPROD_IDX][1])
    shifts_per_day = 3
    plan_days_over = (plan_shifts_over + shifts_per_day - 1) // shifts_per_day

    print(f"OVERPROD_IDX={OVERPROD_IDX}, name={products[OVERPROD_IDX][0]}, plan_shifts={plan_shifts_over}, plan_days≈{plan_days_over}")

    # Настройки LONG_SIMPLE: включаем qty_minus для всех продуктов, как в боевом режиме.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300

    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = None

    # Отключаем внутреннюю пропорциональную цель и стратегии, чтобы они не
    # мешали максимизации TARGET_IDX.
    settings.APPLY_PROP_OBJECTIVE = False
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False
    settings.APPLY_STRATEGY_PENALTY = False

    settings.APPLY_INDEX_UP = True

    # Максимизируем product_counts[TARGET_IDX].
    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = TARGET_IDX

    # Точечный верхний cap для OVERPROD_IDX: не больше плановых дней.
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

    # Сбрасываем debug-флаги/капы.
    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = None
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = None

    if isinstance(res, dict):
        status = res.get("status", -1)
        status_str = res.get("status_str", "")
        prod_stats = res.get("products", [])
        schedule = res.get("schedule", [])
    else:
        status = res.status
        status_str = res.status_str
        prod_stats = res.products
        schedule = res.schedule

    print(f"status={status} ({status_str})")

    # Строим индекс -> (plan, fact).
    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
        }

    # Информация по TARGET_IDX.
    if TARGET_IDX not in stats:
        print(f"TARGET_IDX={TARGET_IDX} отсутствует в статистике продуктов.")
        return

    target_name = data["products"][TARGET_IDX]["name"]
    plan_shifts_t = int(stats[TARGET_IDX]["plan"])
    fact_shifts_t = int(stats[TARGET_IDX]["fact"])

    plan_days_t = (plan_shifts_t + shifts_per_day - 1) // shifts_per_day
    fact_days_t = (fact_shifts_t + shifts_per_day - 1) // shifts_per_day
    missing_days_t = max(0, plan_days_t - fact_days_t)

    print("\nРезультат для TARGET_IDX при cap на OVERPROD_IDX:")
    print(f"  idx={TARGET_IDX}, name={target_name}")
    print(f"  plan_shifts={plan_shifts_t}, plan_days≈{plan_days_t}")
    print(f"  max_fact_shifts={fact_shifts_t}, max_fact_days≈{fact_days_t}")
    print(f"  missing_days≈{missing_days_t}")

    # Информация по OVERPROD_IDX после введения cap.
    if OVERPROD_IDX in stats:
        over_name = data["products"][OVERPROD_IDX]["name"]
        plan_shifts_o = int(stats[OVERPROD_IDX]["plan"])
        fact_shifts_o = int(stats[OVERPROD_IDX]["fact"])
        plan_days_o = (plan_shifts_o + shifts_per_day - 1) // shifts_per_day
        fact_days_o = (fact_shifts_o + shifts_per_day - 1) // shifts_per_day
        print("\nФакт по OVERPROD_IDX после cap:")
        print(f"  idx={OVERPROD_IDX}, name={over_name}")
        print(f"  plan_shifts={plan_shifts_o}, plan_days≈{plan_days_o}")
        print(f"  fact_shifts={fact_shifts_o}, fact_days≈{fact_days_o}")


if __name__ == "__main__":  # pragma: no cover
    main()
