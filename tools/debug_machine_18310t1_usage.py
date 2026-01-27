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


def find_start_machine_for_product(products, machines, target_idx: int) -> int | None:
    """Возвращает индекс машины, на которой target_idx стоит на начало (если есть)."""
    for m_idx, (_name, p_idx, _id, _t, _remain_day, *_rest) in enumerate(machines):
        if p_idx == target_idx:
            return m_idx
    return None


def run_ls_prop(use_qty_minus: bool, subset: set[int] | None):
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 120
    settings.APPLY_QTY_MINUS = use_qty_minus
    if subset is None:
        settings.SIMPLE_QTY_MINUS_SUBSET = None
    else:
        settings.SIMPLE_QTY_MINUS_SUBSET = set(subset)

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

    return status, status_str, schedule, machines, products, data["count_days"]


def summarize_machine_usage(schedule, machines, products, m0: int, num_days: int):
    """Считает, сколько дней на машине m0 отведено под каждый product_idx."""
    # Инициализируем счётчики по всем продуктам, чтобы порядок был стабилен.
    max_p = max(s["product_idx"] for s in schedule) if schedule else 0
    counts = {p: 0 for p in range(max_p + 1)}

    for s in schedule:
        if s["machine_idx"] != m0:
            continue
        d = s["day_idx"]
        if d < 0 or d >= num_days:
            continue
        p = s["product_idx"]
        counts[p] = counts.get(p, 0) + 1

    # Преобразуем в список только тех продуктов, которые реально используются.
    used = [(p, c) for p, c in counts.items() if c > 0]
    used.sort(key=lambda t: (-t[1], t[0]))  # по убыванию дней, затем по idx

    rows = []
    for p, c in used:
        name = products[p]["name"] if isinstance(products[p], dict) else products[p][0]
        rows.append((p, name, c))

    return rows


def main() -> None:
    # 1) Без qty_minus
    status0, status_str0, schedule0, machines, products, num_days = run_ls_prop(False, None)

    target_idx = 14  # ст18310t1
    m0 = find_start_machine_for_product(products, machines, target_idx)
    if m0 is None:
        print("Не найдена машина с начальным продуктом idx=14 (ст18310t1)")
        return

    print(f"start_machine_for_18310t1 = {m0} ({machines[m0][0]})")
    print(f"LS_PROP без qty_minus: status={status0} ({status_str0})")

    rows0 = summarize_machine_usage(schedule0, machines, products, m0, num_days)
    print("\nРаспределение дней по продуктам на машине m0 без qty_minus:")
    for p, name, c in rows0:
        print(f"  p={p}\t{name}\tdays={c}")

    # 2) С qty_minus для subset до idx=10 (без самого idx=10)
    status1, status_str1, schedule1, machines1, products1, num_days1 = run_ls_prop(True, SUBSET_PRE10)
    print(f"\nLS_PROP с qty_minus для SUBSET_PRE10: status={status1} ({status_str1})")

    rows1 = summarize_machine_usage(schedule1, machines1, products1, m0, num_days1)
    print("\nРаспределение дней по продуктам на машине m0 с qty_minus (SUBSET_PRE10):")
    for p, name, c in rows1:
        print(f"  p={p}\t{name}\tdays={c}")


if __name__ == "__main__":  # pragma: no cover
    main()
