from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Set

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def load_data() -> Tuple[dict, list, list, list, list]:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def get_strict_products_sorted(data: dict) -> List[int]:
    """Вернуть список idx продуктов с qty_minus=0 и qty>0, отсортированный по плану qty (ВОЗРАСТАНИЕ)."""
    strict: List[Tuple[int, int]] = []  # (idx, qty_shifts)
    for p in data["products"]:
        idx = int(p["idx"])
        if idx == 0:
            continue
        qty = int(p.get("qty", 0) or 0)
        if qty <= 0:
            continue
        qm = p.get("qty_minus", 0)
        if qm != 0 and qm is not False:
            continue
        strict.append((idx, qty))
    # сортируем по qty убыв.
    # Сортируем по qty по возрастанию, чтобы сначала включать самые маленькие строгие
    # продукты, а крупные оставлять как более гибкий резерв.
    strict.sort(key=lambda t: t[1])
    return [idx for idx, _ in strict]


def run_with_subset(
    data: dict,
    machines: list,
    products: list,
    cleans: list,
    remains: list,
    subset: Set[int],
):
    """Запуск LONG_SIMPLE с APPLY_QTY_MINUS=True только для заданного подмножества строгих продуктов.

    Возвращает (status, status_str, schedule).
    """
    # Базовые настройки для LONG_SIMPLE.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = set(subset)

    # LS_OVER‑подобный режим: внутренняя цель по пропорциям, без альтернативного overpenalty.
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    # В SIMPLE обычно используем MULT‑вариант для пропорций; для проверки выполнимости это не критично.
    # Оставим текущее значение SIMPLE_USE_PROP_MULT как есть.

    settings.APPLY_INDEX_UP = True

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
        status = int(res.get("status", -1))
        status_str = str(res.get("status_str", ""))
        schedule = res.get("schedule", [])
    else:
        status = int(res.status)
        status_str = str(res.status_str)
        schedule = res.schedule

    return status, status_str, schedule


def summarize_capacity_for_subset(data: dict, subset: Set[int]) -> None:
    """Грубая оценка: по каждому div считаем суммарную нижнюю границу по товарам subset
    и сравниваем с машиноднями этого div в LONG_SIMPLE (simple_days * #machines_div).
    """
    products = data["products"]
    machines = data["machines"]

    orig_days = int(data["count_days"])
    simple_days = (orig_days + 3 - 1) // 3

    # Мощность по цехам (div машин)
    cap_by_div: Dict[int, int] = {}
    for m in machines:
        m_div = int(m.get("div", 1) or 1)
        cap_by_div[m_div] = cap_by_div.get(m_div, 0) + simple_days

    # Суммарные нижние границы по продуктам subset.
    # Для строгих qty_minus=0 берём plan_days; если есть qty_minus_min, учитываем min_days.
    lb_by_div: Dict[int, int] = {}
    flex_lb: int = 0  # продукты с prod_div=0 (могут пойти в любой цех)

    for p in products:
        idx = int(p["idx"])
        if idx not in subset:
            continue
        qty = int(p.get("qty", 0) or 0)
        if qty <= 0:
            continue
        qty_minus = p.get("qty_minus", 0)
        if qty_minus != 0 and qty_minus is not False:
            continue

        qty_shifts = qty
        qmm_shifts = int(p.get("qty_minus_min", 0) or 0)
        plan_days = (qty_shifts + 3 - 1) // 3 if qty_shifts > 0 else 0
        min_days = (qmm_shifts + 3 - 1) // 3 if qmm_shifts > 0 else 0
        lower = max(plan_days, min_days)

        prod_div = int(p.get("div", 0) or 0)
        if prod_div in (1, 2):
            lb_by_div[prod_div] = lb_by_div.get(prod_div, 0) + lower
        else:
            flex_lb += lower

    print("\n=== Capacity summary for strict subset ===")
    print(f"orig_days={orig_days}, simple_days={simple_days}")
    print("Per-div machine-days (capacity) vs strict lower bounds (excluding flex-div products):")
    for d in sorted(cap_by_div.keys()):
        cap = cap_by_div[d]
        lb = lb_by_div.get(d, 0)
        print(f"  div={d}: cap_days={cap}, strict_lb_days={lb}, slack={cap - lb}")
    if flex_lb:
        print(f"  flex-div (prod_div not in (1,2)) lower bound total: {flex_lb} days (распределяется по цехам)")


def main() -> None:
    data, machines, products, cleans, remains = load_data()

    strict_idxs = get_strict_products_sorted(data)
    print("Строгие продукты (qty_minus=0, qty>0) по возрастанию qty:")
    print(strict_idxs)

    # Для контроля H2/H3 сразу выведем статус для всех strict, особенно для idx=13.
    from tools.analyze_simple_subset_capacity import estimate_capacity_for_product

    # Восстановим tuple-структуры products/machines, как в других tools.
    from pandas import DataFrame
    products_df = DataFrame(data["products"])
    machines_df = DataFrame(data["machines"])
    products_tuples = [
        (
            row["name"],
            row["qty"],
            row["id"],
            row["machine_type"],
            row["qty_minus"],
            row["lday"],
            row.get("src_root", -1),
            row.get("qty_minus_min", 0),
            row.get("sr", False),
            row.get("strategy", "--"),
        )
        for _, row in products_df.iterrows()
    ]
    machines_tuples = [
        (row["name"], row["product_idx"], row["id"], row["type"], row["remain_day"], row.get("reserve", 0))
        for _, row in machines_df.iterrows()
    ]
    num_days = data["count_days"]

    print("\nH2/H3 статус для строгих продуктов:")
    for idx in strict_idxs:
        if idx < 0 or idx >= len(products_tuples):
            continue
        cap_total, per_m_cap, extra = estimate_capacity_for_product(idx, products_tuples, machines_tuples, num_days)
        h2 = extra.get("h2_active")
        h3 = extra.get("h3_active")
        name = next(p["name"] for p in data["products"] if int(p["idx"]) == idx)
        print(f"  idx={idx}, name={name}, h2={h2}, h3={h3}, cap_total={cap_total}")

    # Последний выполнимый запуск
    last_feasible_k = 0
    last_feasible_subset: Set[int] = set()

    from ortools.sat.python import cp_model

    for k in range(1, len(strict_idxs) + 1):
        subset = set(strict_idxs[:k])
        print("\n=== Запуск с subset первых", k, "строгих продуктов ===")
        print("subset idxs:", sorted(subset))

        status, status_str, _schedule = run_with_subset(data, machines, products, cleans, remains, subset)
        print(f"status={status} ({status_str})")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            last_feasible_k = k
            last_feasible_subset = subset
            continue

        # Первый невыполнимый k
        print("\n!!! Модель стала невыполнимой при k=", k)
        problem_idx = strict_idxs[k - 1]
        prod = next(p for p in data["products"] if int(p["idx"]) == problem_idx)
        print(
            "Проблемный продукт (k-й в порядке):",
            f"idx={problem_idx}, name={prod['name']}, qty={prod['qty']}, qty_minus={prod['qty_minus']}",
        )

        # Сводка по мощности для подмножества до и включая проблемный продукт.
        summarize_capacity_for_subset(data, subset)

        print("\nПорядок включения строгих продуктов (до k):")
        for i, idx in enumerate(strict_idxs[:k], start=1):
            p = next(pp for pp in data["products"] if int(pp["idx"]) == idx)
            qty = int(p.get("qty", 0) or 0)
            plan_days = (qty + 3 - 1) // 3 if qty > 0 else 0
            print(f"  #{i}: idx={idx}, name={p['name']}, qty={qty}, plan_days≈{plan_days}")

        break

    if last_feasible_k == 0:
        print("\nНе найден ни один выполнимый запуск с APPLY_QTY_MINUS=True даже для одного строгого продукта.")
    else:
        print(
            "\nПоследний выполнимый k=",
            last_feasible_k,
            ", subset idxs=",
            sorted(last_feasible_subset),
        )


if __name__ == "__main__":  # pragma: no cover
    main()
