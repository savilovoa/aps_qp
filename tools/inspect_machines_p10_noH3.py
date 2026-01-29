from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


TARGET_IDX = 10  # ст10417RSt
MACHINES_TO_INSPECT = [2, 4, 6]


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300

    # qty_minus включен для всех продуктов
    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = None

    # Отключаем внутреннюю пропорциональную цель и стратегии
    settings.APPLY_PROP_OBJECTIVE = False
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False
    settings.APPLY_STRATEGY_PENALTY = False

    settings.APPLY_INDEX_UP = True

    # Отладка эвристик начального плана: только H1+H2, без H3
    settings.SIMPLE_DEBUG_H_START = True
    settings.SIMPLE_DEBUG_H_MODE = "H12"

    # Максимизируем объём TARGET_IDX, как в max_cap_p10_simple_noH3
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
    settings.SIMPLE_DEBUG_H_START = False
    settings.SIMPLE_DEBUG_H_MODE = None

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

    num_days = data["count_days"]

    # Карта (m,d) -> product_idx
    pd_map: dict[tuple[int, int], int] = {}
    for s in schedule:
        m = s["machine_idx"]
        d = s["day_idx"]
        p = s["product_idx"]
        pd_map[(m, d)] = p

    # Для быстрого доступа к плану/факту и флагам
    stats = {ps["product_idx"]: ps for ps in prod_stats}

    for m in MACHINES_TO_INSPECT:
        if not (0 <= m < len(machines)):
            print(f"\nMachine {m} вне диапазона")
            continue
        m_name, m_init_p, _id, m_type, _remain_day, *_rest = machines[m]
        print("\n=== Машина", m, f"({m_name}), type={m_type}, init_product_idx={m_init_p} ===")
        print("day\tprod_idx\tname\tqty_minus\tplan_shifts\tfact_shifts")

        for d in range(num_days):
            p = pd_map.get((m, d), None)
            if p is None:
                print(f"{d}\t-1\t<none>\t-\t0\t0")
                continue

            prod = data["products"][p]
            name = prod["name"]
            qty_minus_flag = prod["qty_minus"]
            plan_shifts = int(stats.get(p, {}).get("plan_qty", prod["qty"]))
            fact_shifts = int(stats.get(p, {}).get("qty", 0))

            print(f"{d}\t{p}\t{name}\t{qty_minus_flag}\t{plan_shifts}\t{fact_shifts}")


if __name__ == "__main__":  # pragma: no cover
    main()
