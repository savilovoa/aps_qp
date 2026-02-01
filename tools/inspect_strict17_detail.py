from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input
from tools.analyze_simple_subset_capacity import estimate_capacity_for_product


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def get_strict_sorted(data: dict):
    strict = []  # (idx, qty)
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
    strict.sort(key=lambda t: t[1], reverse=True)
    return strict


def main() -> None:
    data, machines_tuples, products_tuples_orig, cleans, remains = load_data()

    products_json = data["products"]

    strict = get_strict_sorted(data)
    print("Всего строгих продуктов:", len(strict))
    print("Первые 17 по убыванию qty:")
    for i, (idx, qty) in enumerate(strict[:17], start=1):
        p = next(pp for pp in products_json if int(pp["idx"]) == idx)
        name = p["name"]
        qm = p["qty_minus"]
        qmm = p.get("qty_minus_min", 0)
        prod_div = p.get("div", 0)
        mt = p.get("machine_type", 0)
        lday = p.get("lday", 0)
        print(
            f"  #{i}: idx={idx}, name={name}, qty={qty}, qty_minus={qm}, "
            f"qty_minus_min={qmm}, div={prod_div}, machine_type={mt}, lday={lday}"
        )

    # Строим tuple-версии products для estimate_capacity_for_product
    products_tuples = []
    for p in products_json:
        products_tuples.append(
            (
                p["name"],
                p["qty"],
                p["id"],
                p["machine_type"],
                p["qty_minus"],
                p["lday"],
                p["src_root"],
                p["qty_minus_min"],
                p["sr"],
                p["strategy"],
            )
        )

    num_days = data["count_days"]

    print("\nОценка capacity для первых 17 строгих (estimate_capacity_for_product):")
    total_plan_days = 0
    total_cap_days = 0
    for i, (idx, qty) in enumerate(strict[:17], start=1):
        prod_d = next(pp for pp in products_json if int(pp["idx"]) == idx)
        name = prod_d["name"]
        plan_shifts = qty
        plan_days = (plan_shifts + 3 - 1) // 3
        cap_total, per_m_cap, extra = estimate_capacity_for_product(idx, products_tuples, machines_tuples, num_days)
        total_plan_days += plan_days
        total_cap_days += cap_total
        print(
            f"  #{i}: idx={idx}, name={name}, plan_shifts={plan_shifts}, plan_days≈{plan_days}, "
            f"cap_days_total={cap_total}, h2={extra['h2_active']}, h3={extra['h3_active']}, "
            f"init_machines={len(extra['machines_for_p'])}, compat_machines={len(extra['compatible_machines'])}"
        )

    print("\nСуммарно по первым 17 строгим:")
    print(f"  total_plan_days≈{total_plan_days}")
    print(f"  total_cap_days≈{total_cap_days}")


if __name__ == "__main__":
    main()
