from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def compute_hard_products_stats() -> tuple[dict[int, dict], list[dict], dict]:
    """LS_PROP без APPLY_QTY_MINUS: возвращает
      - stats: idx -> {plan, fact}
      - products: список продуктов из JSON
      - data: сырой dict из JSON (для count_days и т.п.)
    """
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 120
    settings.APPLY_QTY_MINUS = False
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True
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
        prod_stats = res.get("products", [])
    else:
        prod_stats = res.products

    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
        }

    return stats, data["products"], data


def estimate_capacity_for_product(
    idx: int,
    products: list[tuple],
    machines: list[tuple],
    num_days: int,
) -> tuple[int, dict[int, int], dict]:
    """Грубая оценка верхней границы по дням для продукта idx в LONG_SIMPLE.

    Возвращает:
      total_cap_days, per_machine_cap[m], extra_info
    где extra_info содержит H2/H3-флаги и стартовые машины.
    """
    prod = products[idx]
    plan_shifts = int(prod[1])
    machine_type_req = int(prod[3])
    strategy = prod[9] if len(prod) > 9 else ""

    # initial_products[m] из массива machines (name, product_idx, id, type, remain_day, reserve)
    initial_products: list[int] = []
    for (_, product_idx, _id, _t, _remain_day, *_rest) in machines:
        initial_products.append(product_idx)

    # Карта продукт -> список машин, на которых он стоит на начало.
    product_to_initial_machines: dict[int, list[int]] = {}
    for m_idx, p0 in enumerate(initial_products):
        if p0 <= 0:
            continue
        product_to_initial_machines.setdefault(p0, []).append(m_idx)

    machines_for_p = product_to_initial_machines.get(idx, [])

    # Совместимые по type машины.
    compatible_machines: list[int] = []
    for m_idx, (_m_name, _p_idx, _id, m_type, _remain_day, *_rest) in enumerate(machines):
        if machine_type_req > 0 and m_type != machine_type_req:
            continue
        compatible_machines.append(m_idx)

    shifts_per_day = 3
    plan_days = (plan_shifts + shifts_per_day - 1) // shifts_per_day

    m0 = machines_for_p[0] if machines_for_p else None
    capacity_days_m0 = num_days if m0 is not None else 0

    ENABLE_SIMPLE_SMALL_START_HEURISTIC = True
    ENABLE_SIMPLE_BIG_START_HEURISTIC = True

    h2_active = False
    h3_active = False

    if (
        ENABLE_SIMPLE_SMALL_START_HEURISTIC
        and len(machines_for_p) == 1
        and plan_days < capacity_days_m0
    ):
        h2_active = True

    if (
        ENABLE_SIMPLE_BIG_START_HEURISTIC
        and len(machines_for_p) == 1
        and m0 is not None
        and initial_products[m0] == idx
        and plan_days * 5 >= capacity_days_m0 * 4
        and strategy in ("=", "+")
    ):
        h3_active = True

    per_machine_capacity: dict[int, int] = {}
    total_cap = 0
    for m_idx in compatible_machines:
        if h3_active:
            cap = num_days if m_idx == m0 else 0
        elif h2_active:
            cap = num_days if m_idx == m0 else 0
        else:
            cap = num_days
        per_machine_capacity[m_idx] = cap
        total_cap += cap

    extra_info = {
        "plan_shifts": plan_shifts,
        "plan_days": plan_days,
        "machines_for_p": machines_for_p,
        "compatible_machines": compatible_machines,
        "h2_active": h2_active,
        "h3_active": h3_active,
    }
    return total_cap, per_machine_capacity, extra_info


def main() -> None:
    # Шаг 1: восстанавливаем hard-продукты и их plan/fact без qty_minus.
    stats, products, data = compute_hard_products_stats()

    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data_json, machines, _products2, cleans, remains = load_input(Path(input_path))

    num_days = data_json["count_days"]

    hard_idxs: list[int] = []
    passed_idxs: list[int] = []
    deficit_entries: list[tuple[int, int, int, int]] = []

    for idx, prod in enumerate(products):
        if idx == 0:
            continue
        qty_minus_flag = prod["qty_minus"]
        if qty_minus_flag != 0 and qty_minus_flag is not False:
            continue
        s = stats.get(idx, {"plan": 0, "fact": 0})
        plan = int(s["plan"])
        fact = int(s["fact"])
        delta = fact - plan
        hard_idxs.append(idx)
        if fact >= plan:
            passed_idxs.append(idx)
        else:
            deficit_entries.append((idx, plan, fact, delta))

    deficit_entries.sort(key=lambda t: (t[1] - t[2]), reverse=True)

    print("Всего жёстких продуктов (qty_minus=0):", len(hard_idxs))
    print("Прошли план (fact>=plan):", len(passed_idxs))
    print("С дефицитом (fact<plan):", len(deficit_entries))

    print("\nЖёсткие продукты, прошедшие план (idx, name, plan, fact):")
    for idx in passed_idxs:
        prod = products[idx]
        name = prod["name"]
        s = stats.get(idx, {"plan": 0, "fact": 0})
        print(f"  {idx}\t{name}\tplan={s['plan']}\tfact={s['fact']}")

    print("\nТоп дефицитных (idx, name, plan, fact, deficit):")
    for idx, plan, fact, delta in deficit_entries[:20]:
        name = products[idx]["name"]
        deficit = plan - fact
        print(f"  {idx}\t{name}\tplan={plan}\tfact={fact}\tdeficit={deficit}")

    # Восстанавливаем subset до и включая проблемный продукт idx=10.
    target_idx = 10
    subset: set[int] = set(passed_idxs)
    order: list[int] = []
    for idx, plan, fact, delta in deficit_entries:
        subset.add(idx)
        order.append(idx)
        if idx == target_idx:
            break

    print("\nSubset до и включая idx=10:")
    print("  passed_idxs:", passed_idxs)
    print("  deficit_order_until_10:", order)
    print("  subset_size=", len(subset))

    # Теперь для каждого продукта из subset считаем plan_days и грубую capacity_upper_bound.
    print("\nДетальный список subset (idx, name, plan_shifts, plan_days, cap_days_total, h2, h3, |machines_for_p|, |compatible_machines|):")
    total_plan_days = 0
    total_cap_days = 0

    # products из JSON (dict), machines из load_input, но estimate_capacity_for_product ждёт tuple-структуру products,
    # поэтому восстановим tuple-список, как в schedule_loom_calc.
    products_tuples = []
    for p in products:
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

    for idx in sorted(subset):
        prod_d = products[idx]
        name = prod_d["name"]
        plan_shifts = int(stats.get(idx, {"plan": prod_d["qty"]})["plan"])
        plan_days = (plan_shifts + 3 - 1) // 3
        cap_total, per_m_cap, extra = estimate_capacity_for_product(idx, products_tuples, machines, num_days)

        total_plan_days += plan_days
        total_cap_days += cap_total

        print(
            f"  {idx}\t{name}\tplan_shifts={plan_shifts}\tplan_days≈{plan_days}\tcap_days_total={cap_total}\t"
            f"h2={extra['h2_active']}\th3={extra['h3_active']}\t"
            f"init_machines={len(extra['machines_for_p'])}\tcompatible_machines={len(extra['compatible_machines'])}"
        )

    print("\nИТОГО по subset:")
    print(f"  total_plan_days≈{total_plan_days}")
    print(f"  total_cap_days≈{total_cap_days} (простая сумма индивидуальных верхних оценок по продуктам)")
    print(f"  Горизонт: num_machines={len(machines)}, num_days={num_days}, machine_days={len(machines)*num_days}")


if __name__ == "__main__":  # pragma: no cover
    main()
