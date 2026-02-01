from __future__ import annotations

from pathlib import Path
import math
import csv

from src.config import settings
from tools.compare_long_vs_simple import load_input


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def can_run_product_on_machine(prod: dict, mach: dict) -> bool:
    """Упрощённая совместимость по type/div, как в SIMPLE.

    prod["machine_type"]: 1/2 -> только на машинах с таким type; 0 -> на любых.
    prod["div"]: 1/2 -> только на машинах с таким div; 0/отсутствует -> любой div.
    """
    p_type = int(prod.get("machine_type", 0) or 0)
    p_div = int(prod.get("div", 0) or 0)
    m_type = int(mach.get("type", 0) or 0)
    m_div = int(mach.get("div", 1) or 1)

    if p_type > 0 and m_type != p_type:
        return False
    if p_div in (1, 2) and m_div != p_div:
        return False
    return True


def detect_h2_h3(products: list[dict], machines: list[dict], num_days: int):
    """Грубая реконструкция H2/H3 по логике SIMPLE.

    H2: один стартовый станок, plan_days < num_days.
    H3: один стартовый станок, init_p == p, plan_days покрывает ~80% горизонта, стратегия '=' или '+'.
    """
    shifts_per_day = 3

    # Стартовые продукты по машинам (из JSON)
    initial_products: list[int] = [int(m.get("product_idx", 0) or 0) for m in machines]

    product_to_initial_machines: dict[int, list[int]] = {}
    for m_idx, p0 in enumerate(initial_products):
        if p0 <= 0:
            continue
        product_to_initial_machines.setdefault(p0, []).append(m_idx)

    h2_products: set[int] = set()
    h3_products: set[int] = set()

    for p in products:
        idx = int(p["idx"])
        if idx == 0:
            continue
        qty = int(p.get("qty", 0) or 0)
        if qty <= 0:
            continue
        plan_days = (qty + shifts_per_day - 1) // shifts_per_day
        machines_for_p = product_to_initial_machines.get(idx, [])
        if not machines_for_p:
            continue
        m0 = machines_for_p[0]
        capacity_days_m0 = num_days
        strategy = str(p.get("strategy", ""))

        # H2: маленький стартовый продукт на одном станке, план меньше горизонта.
        if len(machines_for_p) == 1 and plan_days < capacity_days_m0:
            h2_products.add(idx)

        # H3: большой стартовый продукт, почти забирающий весь горизонт.
        if (
            len(machines_for_p) == 1
            and initial_products[m0] == idx
            and plan_days * 5 >= capacity_days_m0 * 4  # ~80%
            and strategy in ("=", "+")
        ):
            h3_products.add(idx)

    return h2_products, h3_products, product_to_initial_machines


def build_h123_schedule(data: dict) -> list[list[int]]:
    """Строим матрицу schedule[m][d] только по эффекту H2/H3 в LONG_SIMPLE.

    Остальные дни оставляем -1 (пусто). Горизонт: simple_days = ceil(orig_days/3).
    """
    products_json = data["products"]
    machines_json = data["machines"]

    orig_days = int(data["count_days"])
    shifts_per_day = 3
    simple_days = (orig_days + shifts_per_day - 1) // shifts_per_day

    num_machines = len(machines_json)
    schedule: list[list[int]] = [[-1 for _ in range(simple_days)] for _ in range(num_machines)]

    h2_products, h3_products, product_to_initial_machines = detect_h2_h3(
        products_json, machines_json, simple_days
    )

    # H3 сначала: они забивают машину полностью.
    for p in products_json:
        idx = int(p["idx"])
        if idx not in h3_products:
            continue
        machines_for_p = product_to_initial_machines.get(idx, [])
        if not machines_for_p:
            continue
        m0 = machines_for_p[0]
        for d in range(simple_days):
            schedule[m0][d] = idx

    # H2: короткий стартовый блок на m0, только если H3 его не перезаписал.
    for p in products_json:
        idx = int(p["idx"])
        if idx not in h2_products or idx in h3_products:
            continue
        qty = int(p.get("qty", 0) or 0)
        if qty <= 0:
            continue
        plan_days = (qty + shifts_per_day - 1) // shifts_per_day
        machines_for_p = product_to_initial_machines.get(idx, [])
        if not machines_for_p:
            continue
        m0 = machines_for_p[0]
        for d in range(min(plan_days, simple_days)):
            # Не перезаписываем H3, но можем затирать другие H2 в этой простой визуализации.
            if schedule[m0][d] == -1:
                schedule[m0][d] = idx

    return schedule


def dump_csv(path: Path, data: dict, schedule: list[list[int]]) -> None:
    machines_json = data["machines"]
    simple_days = len(schedule[0]) if schedule else 0

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        # Заголовок: машина, тип, цех, нач продукт, затем d0..d{simple_days-1}
        header = [
            "machine_idx",
            "machine_name",
            "type",
            "div",
            "initial_product_idx",
        ] + [f"d{d}" for d in range(simple_days)]
        writer.writerow(header)

        for m_idx, m in enumerate(machines_json):
            name = m.get("name", f"m{m_idx}")
            m_type = m.get("type", "")
            m_div = m.get("div", "")
            init_p = m.get("product_idx", 0)
            row = [m_idx, name, m_type, m_div, init_p] + schedule[m_idx]
            writer.writerow(row)


def main() -> None:
    data, machines, products, cleans, remains = load_data()
    schedule = build_h123_schedule(data)
    out_path = Path("example/h123_simple_plan.csv")
    dump_csv(out_path, data, schedule)
    print(f"CSV with H1-H3 heuristic schedule saved to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
