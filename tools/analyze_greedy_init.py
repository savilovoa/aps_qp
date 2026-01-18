import json
from pathlib import Path
from collections import Counter

from src.config import settings
from src.loom.schedule_loom import create_schedule_init


def load_input() -> tuple[list[dict], list[dict], list[dict], int, int]:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    with open(input_path, "r", encoding="utf8") as f:
        data = json.load(f)

    machines: list[dict] = data["machines"]
    products: list[dict] = data["products"]
    cleans: list[dict] = data["cleans"]
    count_days: int = data["count_days"]
    max_daily_prod_zero: int = data["max_daily_prod_zero"]
    return machines, products, cleans, count_days, max_daily_prod_zero


def summarize_by_product(schedule, products: list[dict]) -> Counter[int]:
    """Сводка по продуктам: план qty, факт по жадному плану, отклонение.

    Возвращает fact_counts: idx -> количество смен продукта в жадном плане.
    """
    num_machines = len(schedule)
    count_days = len(schedule[0]) if num_machines > 0 else 0

    fact_counts: Counter[int] = Counter()
    for m in range(num_machines):
        for d in range(count_days):
            val = schedule[m][d]
            # schedule[m][d] может быть либо int, либо списком, где первый элемент — индекс продукта
            if isinstance(val, list):
                if not val:
                    continue
                val = val[0]
            if val is None or val in (-2, 0):
                continue
            fact_counts[int(val)] += 1

    print("\n=== Сводка по продуктам (жадный план) ===")
    print("id\tidx\tname\tplan_qty\tfact_qty\tdelta")

    # products[idx]["idx"] должен совпадать с индексом в расписании
    for prod in products:
        idx = prod["idx"]
        name = prod["name"]
        pid = prod["id"]
        plan_qty = int(prod.get("qty", 0))
        fact_qty = fact_counts.get(idx, 0)
        delta = fact_qty - plan_qty
        print(f"{pid}\t{idx}\t{name}\t{plan_qty}\t{fact_qty}\t{delta}")

    return fact_counts


def summarize_by_machine(schedule, machines: list[dict]):
    """Сводка по машинам: количество переходов (нулей), работа, пусто, % использования."""
    num_machines = len(schedule)
    count_days = len(schedule[0]) if num_machines > 0 else 0

    print("\n=== Сводка по машинам (жадный план) ===")
    print("id\tidx\tname\tzeros(transitions)\twork_days\tidle_days\tcleans\tutil_percent")

    for m in range(num_machines):
        mach = machines[m]
        mid = mach["id"]
        name = mach["name"]
        zeros = 0
        work = 0
        idle = 0
        cleans = 0
        for d in range(count_days):
            val = schedule[m][d]
            if isinstance(val, list):
                if not val:
                    idle += 1
                    continue
                val = val[0]
            if val is None:
                idle += 1
            elif val == -2:
                cleans += 1
            elif val == 0:
                zeros += 1
            else:
                work += 1

        total_slots = count_days - cleans
        util = (work / total_slots * 100.0) if total_slots > 0 else 0.0
        print(
            f"{mid}\t{m}\t{name}\t{zeros}\t{work}\t{idle}\t{cleans}\t{util:.1f}"
        )


def main() -> None:
    machines, products, cleans, count_days, max_daily_prod_zero = load_input()

    schedule, obj, dev_prop, count_zero = create_schedule_init(
        machines=machines,
        products=products,
        cleans=cleans,
        count_days=count_days,
        max_daily_prod_zero=max_daily_prod_zero,
    )

    fact_counts = summarize_by_product(schedule, products)
    summarize_by_machine(schedule, machines)

    # --- Анализ кандидатов на фиксацию по greedy ---
    print("\n=== Анализ кандидатов на фиксацию по greedy ===")

    count_days = len(schedule[0]) if schedule else 0

    # Быстрый доступ к планам по idx
    plan_by_idx: dict[int, int] = {}
    qty_minus_by_idx: dict[int, bool] = {}
    qty_minus_min_by_idx: dict[int, int] = {}
    for prod in products:
        idx = int(prod["idx"])
        if idx == 0:
            continue
        plan_by_idx[idx] = int(prod.get("qty", 0))
        qty_minus_by_idx[idx] = bool(prod.get("qty_minus", False))
        qty_minus_min_by_idx[idx] = int(prod.get("qty_minus_min", 0) or 0)

    # Множество дней чисток
    cleans_set = {(c["machine_idx"], c["day_idx"]) for c in cleans}

    # 1. Машины с тем же начальным продуктом и полной загрузкой им до конца (без нулей и пустот).
    full_initial_machines: list[int] = []
    for m_idx, mach in enumerate(machines):
        init_p = int(mach["product_idx"])
        if init_p == 0:
            continue
        ok = True
        has_work = False
        for d in range(count_days):
            if (m_idx, d) in cleans_set:
                continue
            val = schedule[m_idx][d]
            if isinstance(val, list):
                if not val:
                    continue
                val = val[0]
            if val is None:
                ok = False
                break
            if val == 0:
                ok = False
                break
            if val != init_p:
                ok = False
                break
            has_work = True
        if ok and has_work:
            full_initial_machines.append(m_idx)

    print(f"Полных машин с начальными продуктами (кандидаты на фиксацию): {len(full_initial_machines)}")

    # 2. Из них машины для продуктов с qty_minus_min > 0.
    fixed_qtymin_machines: list[int] = []
    for m_idx in full_initial_machines:
        init_p = int(machines[m_idx]["product_idx"])
        qmm = qty_minus_min_by_idx.get(init_p, 0)
        if qmm > 0:
            fixed_qtymin_machines.append(m_idx)

    print(f"Из них с qty_minus_min>0: {len(fixed_qtymin_machines)} машин")
    if fixed_qtymin_machines:
        print("Машины с полной загрузкой и qty_minus_min>0 (idx -> product_idx):")
        for m_idx in fixed_qtymin_machines:
            print(f"  m={m_idx}\tinit_p={machines[m_idx]['product_idx']}")

    # 3. Дефициты по продуктам относительно плана.
    print("\nДефициты по продуктам (plan - fact, только >0):")
    any_def = False
    for idx, plan in plan_by_idx.items():
        fact = fact_counts.get(idx, 0)
        deficit = plan - fact
        if deficit > 0:
            any_def = True
            print(f"idx={idx}\tplan={plan}\tfact={fact}\tdeficit={deficit}")
    if not any_def:
        print("Дефицитов по продуктам нет (fact_qty >= plan_qty для всех idx>0).")

    # Дополнительно: проверка правил и отклонений пропорций.
    print("\n=== Проверка правил qty_minus и отклонений пропорций (жадный план) ===")

    # Подготовим плановые и фактические суммы по продуктам (idx > 0)
    plan_total = 0
    fact_total = 0
    plan_by_idx: dict[int, int] = {}
    qty_minus_by_idx: dict[int, bool] = {}
    qty_minus_min_by_idx: dict[int, int] = {}

    for prod in products:
        idx = int(prod["idx"])
        if idx == 0:
            continue
        plan = int(prod.get("qty", 0))
        plan_by_idx[idx] = plan
        plan_total += plan
        fact = int(fact_counts.get(idx, 0))
        fact_total += fact
        qty_minus_by_idx[idx] = bool(prod.get("qty_minus", False))
        qty_minus_min_by_idx[idx] = int(prod.get("qty_minus_min", 0) or 0)

    # Проверка жёсткой верхней границы для qty_minus=0: fact <= plan + 6
    print("\nНарушения верхней границы для qty_minus=0 (fact_qty > plan_qty + 6):")
    any_upper = False
    for idx, plan in plan_by_idx.items():
        if not qty_minus_by_idx.get(idx, False):
            fact = fact_counts.get(idx, 0)
            upper = plan + 6
            if fact > upper:
                any_upper = True
                print(f"idx={idx}\tplan={plan}\tfact={fact}\tupper={upper}")
    if not any_upper:
        print("Нет нарушений верхней границы для qty_minus=0.")

    # Проверка потенциальной жёсткой нижней границы для qty_minus=1:
    # fact_qty >= plan_qty - max(qty_minus_min, 10)
    print("\nНарушения нижней границы для qty_minus=1 (fact_qty < plan_qty - max(qty_minus_min, 10)):")
    any_lower = False
    for idx, plan in plan_by_idx.items():
        if not qty_minus_by_idx.get(idx, False):
            continue
        fact = fact_counts.get(idx, 0)
        qmm = qty_minus_min_by_idx.get(idx, 0)
        tol = max(qmm, 10)
        lower = max(0, plan - tol)
        if fact < lower:
            any_lower = True
            print(f"idx={idx}\tplan={plan}\tqty_minus_min={qmm}\tfact={fact}\tlower={lower}")
    if not any_lower:
        print("Нет нарушений нижней границы для qty_minus=1.")

    # Отклонения пропорций: сравниваем доли plan vs fact
    if plan_total > 0 and fact_total > 0:
        sum_abs_share_delta = 0.0
        max_abs_share_delta = 0.0
        for idx, plan in plan_by_idx.items():
            fact = fact_counts.get(idx, 0)
            plan_share = plan / plan_total
            fact_share = fact / fact_total
            delta = fact_share - plan_share
            sum_abs_share_delta += abs(delta)
            max_abs_share_delta = max(max_abs_share_delta, abs(delta))
        print(
            f"\nСуммарное |Δдолей| по всем продуктам: {sum_abs_share_delta:.4f}"
        )
        print(f"Максимальное |Δдоли| для одного продукта: {max_abs_share_delta:.4f}")
    else:
        print("\nОтклонения пропорций не считаем (plan_total или fact_total = 0).")

    # Полный план по сменам для каждой машины в компактном виде
    print("\n=== Полный жадный план по сменам (по машинам) ===")
    count_days = len(schedule[0]) if schedule else 0
    for m_idx, mach in enumerate(machines):
        init_p = int(mach["product_idx"])
        seq = []
        for d in range(count_days):
            val = schedule[m_idx][d]
            if isinstance(val, list):
                if not val:
                    code = "__"
                    seq.append(code)
                    continue
                val = val[0]
            if val is None:
                code = "__"  # пусто
            elif val == -2:
                code = "CL"  # чистка
            else:
                code = f"{int(val):02d}"
            seq.append(code)
        print(f"m={m_idx:02d}\tinit={init_p:02d}\t" + ",".join(seq))


if __name__ == "__main__":
    main()
