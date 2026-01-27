from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


TARGET_IDX = 10  # ст10417RSt


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настройки LONG_SIMPLE: хотим чисто структурную максимизацию объёма по TARGET_IDX
    # при включённых qty_minus для всех продуктов.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300

    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = None  # применяем qty_minus ко всем продуктам

    # Отключаем внутреннюю пропорциональную цель и штрафы по стратегиям, чтобы не
    # влияли на максимизацию количества TARGET_IDX.
    settings.APPLY_PROP_OBJECTIVE = False
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False
    settings.APPLY_STRATEGY_PENALTY = False

    settings.APPLY_INDEX_UP = True

    # Отладочный флаг: максимизируем product_counts[TARGET_IDX] в SIMPLE.
    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = TARGET_IDX

    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    # Сбрасываем debug-флаг, чтобы не оставался в глобальном состоянии.
    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = None

    if isinstance(res, dict):
        status = res.get("status", -1)
        status_str = res.get("status_str", "")
        prod_stats = res.get("products", [])
    else:
        status = res.status
        status_str = res.status_str
        prod_stats = res.products

    print(f"status={status} ({status_str})")

    # Строим индекс -> (plan, fact) по продуктам.
    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
        }

    # Достаём информацию по TARGET_IDX.
    if TARGET_IDX not in stats:
        print(f"TARGET_IDX={TARGET_IDX} отсутствует в статистике продуктов.")
        return

    target_name = data["products"][TARGET_IDX]["name"]
    plan_shifts = int(stats[TARGET_IDX]["plan"])
    fact_shifts = int(stats[TARGET_IDX]["fact"])

    shifts_per_day = 3
    plan_days = (plan_shifts + shifts_per_day - 1) // shifts_per_day
    fact_days = (fact_shifts + shifts_per_day - 1) // shifts_per_day

    missing_days = max(0, plan_days - fact_days)

    print("\nРезультат для TARGET_IDX:")
    print(f"  idx={TARGET_IDX}, name={target_name}")
    print(f"  plan_shifts={plan_shifts}, plan_days≈{plan_days}")
    print(f"  max_fact_shifts={fact_shifts}, max_fact_days≈{fact_days}")
    print(f"  missing_days≈{missing_days}")

    # Разложим, где именно стоят дни TARGET_IDX по машинам и типам.
    schedule = res["schedule"] if isinstance(res, dict) else res.schedule

    # Собираем по машинам и типам стоянку TARGET_IDX.
    days_by_machine: dict[int, int] = {}
    days_by_type: dict[int, int] = {}

    for s in schedule:
        if s["product_idx"] != TARGET_IDX:
            continue
        m = s["machine_idx"]
        days_by_machine[m] = days_by_machine.get(m, 0) + 1
        m_type = machines[m][3]
        days_by_type[m_type] = days_by_type.get(m_type, 0) + 1

    print("\nРаспределение TARGET_IDX по машинам:")
    for m in sorted(days_by_machine.keys()):
        name = machines[m][0]
        m_type = machines[m][3]
        print(f"  machine={m} ({name}), type={m_type}, days={days_by_machine[m]}")

    print("\nРаспределение TARGET_IDX по типам (цехам):")
    for t in sorted(days_by_type.keys()):
        print(f"  type={t}: days={days_by_type[t]}")


if __name__ == "__main__":  # pragma: no cover
    main()
