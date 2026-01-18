import json
from pathlib import Path
from typing import Any


def compute_capacity(data: dict[str, Any]) -> None:
    machines = data["machines"]
    products = data["products"]
    cleans = data.get("cleans", [])
    count_days = data["count_days"]

    # Собираем список рабочих дней (machine_idx, day_idx), исключая чистки
    cleans_set = {(c["machine_idx"], c["day_idx"]) for c in cleans}

    work_days: list[tuple[int, int]] = []
    for m in machines:
        m_idx = m["idx"]
        for d in range(count_days):
            if (m_idx, d) not in cleans_set:
                work_days.append((m_idx, d))

    # Типы машин по индексу
    machine_type_by_idx: dict[int, int] = {m["idx"]: m.get("type", 0) for m in machines}

    # Для каждого продукта считаем min_required и capacity_machine_days так же, как в модели
    print("ProductIdx;ProductName;qty;qty_minus;qty_minus_min;min_required;capacity_machine_days;lday;machine_type")

    for p in products:
        p_idx = p["idx"]
        qty = int(p.get("qty", 0))
        qty_minus = int(p.get("qty_minus", 0))
        qty_minus_min = int(p.get("qty_minus_min", 0))
        lday = int(p.get("lday", 0))
        machine_type_req = int(p.get("machine_type", 0))
        name = p.get("name", "")

        # min_required по тому же правилу, что и в create_model
        min_required = 0
        if qty > 0:
            if qty_minus == 0:
                min_required = qty
            else:
                if qty_minus_min > 0:
                    min_required = qty_minus_min

        # capacity_machine_days: сколько рабочих дней доступно продукту по типу станка
        capacity = 0
        for (m_idx, d) in work_days:
            m_type = machine_type_by_idx.get(m_idx, 0)
            if machine_type_req == 1 and m_type != 1:
                continue
            capacity += 1

        print(
            f"{p_idx};{name};{qty};{qty_minus};{qty_minus_min};"
            f"{min_required};{capacity};{lday};{machine_type_req}"
        )

    print("\nProducts where min_required > capacity_machine_days:")
    any_bad = False
    for p in products:
        p_idx = p["idx"]
        qty = int(p.get("qty", 0))
        qty_minus = int(p.get("qty_minus", 0))
        qty_minus_min = int(p.get("qty_minus_min", 0))
        machine_type_req = int(p.get("machine_type", 0))
        lday = int(p.get("lday", 0))

        # min_required
        min_required = 0
        if qty > 0:
            if qty_minus == 0:
                min_required = qty
            else:
                if qty_minus_min > 0:
                    min_required = qty_minus_min

        # capacity
        capacity = 0
        for (m_idx, d) in work_days:
            m_type = machine_type_by_idx.get(m_idx, 0)
            if machine_type_req == 1 and m_type != 1:
                continue
            capacity += 1

        if min_required > capacity:
            any_bad = True
            print(
                f"  Product idx={p_idx} name='{p.get('name','')}' has min_required={min_required} > capacity={capacity} "
                f"(lday={lday}, machine_type={machine_type_req})"
            )

    if not any_bad:
        print("  NONE")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check capacity vs min_required for APS loom input file")
    parser.add_argument("input", help="Path to input JSON file (e.g. example/middle_small_in.json)")
    args = parser.parse_args()

    path = Path(args.input)
    with path.open(encoding="utf8") as f:
        data = json.load(f)

    compute_capacity(data)


if __name__ == "__main__":
    main()
