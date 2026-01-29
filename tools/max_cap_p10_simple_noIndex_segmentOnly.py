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

    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300

    # qty_minus включен для всех (боевой режим)
    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = None

    # Отключаем пропорциональную цель и стратегии, чтобы они не мешали
    # максимизации TARGET_IDX.
    settings.APPLY_PROP_OBJECTIVE = False
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False
    settings.APPLY_STRATEGY_PENALTY = False

    # INDEX_UP глобально включен, но для SIMPLE мы включим флаг его игнорировать.
    settings.APPLY_INDEX_UP = True
    settings.SIMPLE_DISABLE_INDEX_UP = True

    # Эвристики начального плана: H1+H2, без H3
    settings.SIMPLE_DEBUG_H_START = True
    settings.SIMPLE_DEBUG_H_MODE = "H12"

    # Максимизируем product_counts[TARGET_IDX].
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

    # Сбрасываем debug-флаги
    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = None
    settings.SIMPLE_DISABLE_INDEX_UP = False
    settings.SIMPLE_DEBUG_H_START = False
    settings.SIMPLE_DEBUG_H_MODE = None

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

    # Статистика по продуктам
    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
        }

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

    print("\nРезультат для TARGET_IDX (без INDEX_UP, но с ограничением одного сегмента):")
    print(f"  idx={TARGET_IDX}, name={target_name}")
    print(f"  plan_shifts={plan_shifts}, plan_days≈{plan_days}")
    print(f"  max_fact_shifts={fact_shifts}, max_fact_days≈{fact_days}")
    print(f"  missing_days≈{missing_days}")

    # Проверим, что для TARGET_IDX и для пары других продуктов не нарушается
    # ограничение "один сегмент на машину" (грубо по расписанию).
    num_days = data["count_days"]
    machines_with_target = {s["machine_idx"] for s in schedule if s["product_idx"] == TARGET_IDX}
    print("\nПроверка сегментов для TARGET_IDX по машинам:")
    for m in sorted(machines_with_target):
        m_name = machines[m][0]
        # Собираем последовательность по дням
        seq = [s["product_idx"] for s in schedule if s["machine_idx"] == m]
        # Выделяем позиции TARGET_IDX
        positions = [d for d, p in enumerate(seq) if p == TARGET_IDX]
        if positions:
            # Проверим, есть ли разрывы
            gaps = 0
            prev = positions[0]
            for pos in positions[1:]:
                if pos != prev + 1:
                    gaps += 1
                prev = pos
            print(f"  machine={m} ({m_name}): positions={positions}, gaps={gaps}")


if __name__ == "__main__":  # pragma: no cover
    main()
