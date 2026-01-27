from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


# Жёсткие продукты в subset до первой невыполнимости (без idx=10)
SUBSET_PRE10 = {
    1, 2, 3, 5, 8, 9,
    32, 24, 23, 25, 11, 22, 15, 16, 17, 21, 14, 19, 18,
}


def run_ls_prop_with_subset():
    """Запускает LS_PROP (LONG_SIMPLE) с APPLY_QTY_MINUS=True и подмножеством
    жёстких продуктов SUBSET_PRE10. Возвращает schedule и исходные data.
    """
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 120
    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = set(SUBSET_PRE10)

    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True
    settings.APPLY_INDEX_UP = True
    settings.SIMPLE_DEBUG_H_START = False
    settings.SIMPLE_DEBUG_H_MODE = None

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
        status = res.get("status", -1)
        status_str = res.get("status_str", "")
        schedule = res.get("schedule", [])
    else:
        status = res.status
        status_str = res.status_str
        schedule = res.schedule

    return status, status_str, schedule, machines, products, data


def main() -> None:
    status, status_str, schedule, machines, products, data = run_ls_prop_with_subset()
    num_days = data["count_days"]

    print(f"LS_PROP с qty_minus для SUBSET_PRE10: status={status} ({status_str})")

    # Индекс -> тип машины (будем трактовать как цех).
    machine_types = {m_idx: m[3] for m_idx, m in enumerate(machines)}

    # 1) Одностаночные машины: на машине используется только один жёсткий продукт (qty_minus=0).
    #    Считаем по фактическому расписанию.
    qty_minus_flags = {idx: (p["qty_minus"] if isinstance(p, dict) else p[4]) for idx, p in enumerate(products)}

    products_by_machine: dict[int, set[int]] = {}
    for s in schedule:
        m = s["machine_idx"]
        p = s["product_idx"]
        if qty_minus_flags.get(p, 1) != 0:
            continue  # интересуют только жёсткие
        products_by_machine.setdefault(m, set()).add(p)

    mono_machines: set[int] = set()
    for m, pset in products_by_machine.items():
        if len(pset) == 1:
            mono_machines.add(m)

    print("\nОдностаночные машины (по фактическому расписанию, только жёсткие продукты):")
    for m in sorted(mono_machines):
        pset = products_by_machine[m]
        names = [products[p]["name"] if isinstance(products[p], dict) else products[p][0] for p in pset]
        print(f"  machine={m} ({machines[m][0]}), hard_products={list(pset)}, names={names}")

    # 2) Дни, зарезервированные на начало под стартовые продукты (prefix, где product_idx == initial_product).
    initial_products = [m[1] for m in machines]
    reserved_initial: set[tuple[int, int]] = set()

    # Сначала построим карту (m,d) -> p из schedule для быстрого доступа.
    pd_map: dict[tuple[int, int], int] = {}
    for s in schedule:
        pd_map[(s["machine_idx"], s["day_idx"])] = s["product_idx"]

    for m_idx, p0 in enumerate(initial_products):
        # Пропускаем одностаночные машины: они и так будут исключены из остаточной ёмкости.
        if m_idx in mono_machines:
            continue
        # Префикс дней, где стоит стартовый продукт.
        for d in range(num_days):
            p = pd_map.get((m_idx, d))
            if p != p0:
                break
            reserved_initial.add((m_idx, d))

    print(f"\nКоличество стартовых дней, зарезервированных под начальные продукты (кроме одностаночных машин): {len(reserved_initial)}")

    # 3) Мощность по всем машинам/дням, кроме одностаночных и стартовых дней.
    #    Также вычтем дни, занятые жёсткими продуктами с idx <= 10.
    strict_early = {idx for idx, flag in qty_minus_flags.items() if flag == 0 and idx <= 10}

    total_machine_days_by_type: dict[int, int] = {}
    reserved_initial_by_type: dict[int, int] = {}
    used_strict_early_by_type: dict[int, int] = {}
    free_by_type: dict[int, int] = {}

    for m_idx, m in enumerate(machines):
        if m_idx in mono_machines:
            continue  # полностью исключаем одностаночные машины
        m_type = machine_types[m_idx]
        for d in range(num_days):
            key = (m_idx, d)
            p = pd_map.get(key, None)

            # Общее количество машинных дней (после исключения одностаночных).
            total_machine_days_by_type[m_type] = total_machine_days_by_type.get(m_type, 0) + 1

            if key in reserved_initial:
                reserved_initial_by_type[m_type] = reserved_initial_by_type.get(m_type, 0) + 1
                continue

            if p in strict_early:
                used_strict_early_by_type[m_type] = used_strict_early_by_type.get(m_type, 0) + 1
                continue

            free_by_type[m_type] = free_by_type.get(m_type, 0) + 1

    print("\nСводка по оставшейся мощности (по типам машин, трактуемым как цехи):")
    types = sorted(total_machine_days_by_type.keys())
    for t in types:
        total_days = total_machine_days_by_type.get(t, 0)
        init_days = reserved_initial_by_type.get(t, 0)
        strict_days = used_strict_early_by_type.get(t, 0)
        free_days = free_by_type.get(t, 0)
        print(
            f"  type={t}: total_days={total_days}, reserved_initial={init_days}, "
            f"used_strict_early(idx<=10)={strict_days}, free_after_all={free_days}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
