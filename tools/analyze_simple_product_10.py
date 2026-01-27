from pathlib import Path

from src.config import settings
from tools.compare_long_vs_simple import load_input


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    num_days = data["count_days"]
    num_machines = len(machines)

    target_idx = 10  # ст10417RSt по результатам debug_qty_minus_path
    if not (0 <= target_idx < len(products)):
        print(f"target_idx={target_idx} вне диапазона продуктов (0..{len(products)-1})")
        return

    prod = products[target_idx]
    name = prod[0]
    plan_shifts = int(prod[1])
    machine_type_req = int(prod[3])
    qty_minus_flag = prod[4]
    strategy = prod[9] if len(prod) > 9 else ""

    shifts_per_day = 3
    plan_days = (plan_shifts + shifts_per_day - 1) // shifts_per_day

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

    machines_for_p = product_to_initial_machines.get(target_idx, [])

    print(f"Анализ продукта idx={target_idx}, name={name}")
    print(f"  plan_shifts={plan_shifts}, plan_days≈{plan_days}, qty_minus={qty_minus_flag}, strategy='{strategy}', machine_type_req={machine_type_req}")
    print(f"  num_days={num_days}, num_machines={num_machines}")
    print(f"  machines_for_p (стоит на начало): {machines_for_p}")

    # Анализ совместимости по типу машин (div в этом тестовом входе не используем, product_divs/machine_divs по умолчанию 0/1).
    compatible_machines: list[int] = []
    for m_idx, (m_name, _p_idx, _id, m_type, _remain_day, *_rest) in enumerate(machines):
        if machine_type_req > 0 and m_type != machine_type_req:
            comp = False
        else:
            comp = True
        if comp:
            compatible_machines.append(m_idx)
        print(
            f"  machine {m_idx}: name={m_name}, type={m_type}, init_prod={initial_products[m_idx]}, "
            f"compatible={comp}"
        )

    print(f"\nСовместимых по type машин для продукта {target_idx}: {compatible_machines}")

    # Проверяем, срабатывают ли H2/H3 для этого продукта (как в create_model_simple, при use_qty_minus=False).
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
        and initial_products[m0] == target_idx
        and plan_days * 5 >= capacity_days_m0 * 4
        and strategy in ("=", "+")
    ):
        h3_active = True

    print(f"\nH2 active for idx={target_idx}: {h2_active}")
    print(f"H3 active for idx={target_idx}: {h3_active}")

    # Оцениваем грубую верхнюю границу по количеству дней, где этот продукт может стоять,
    # с учётом только совместимости по типу и базовых эффектов H2/H3 (без переходов и qty_minus других продуктов).
    capacity_upper_bound = 0
    per_machine_capacity: dict[int, int] = {}

    for m_idx in compatible_machines:
        # Если H3 активна, допускаем продукт только на m0.
        if h3_active:
            if m_idx == m0:
                cap = num_days
            else:
                cap = 0
        # Если H2 активна, запрещаем продукт на машинах != m0, на m0 — можем ставить в любые дни.
        elif h2_active:
            if m_idx == m0:
                cap = num_days
            else:
                cap = 0
        else:
            # Без H2/H3: по совместимости по типу можем в любом дне.
            cap = num_days

        per_machine_capacity[m_idx] = cap
        capacity_upper_bound += cap

    print("\nОценка верхней границы по дням для продукта idx=10 (по типу + H2/H3):")
    for m_idx in sorted(per_machine_capacity):
        print(f"  machine {m_idx}: cap_days={per_machine_capacity[m_idx]}")
    print(f"Итого capacity_upper_bound={capacity_upper_bound} дней при требуемом plan_days≈{plan_days}")


if __name__ == "__main__":  # pragma: no cover
    main()
