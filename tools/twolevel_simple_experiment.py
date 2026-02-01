from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from ortools.sat.python import cp_model

from src.config import settings
from tools.compare_long_vs_simple import load_input


def build_master_model(
    data,
    products_df,
    machines_df,
    count_days: int,
) -> Tuple[cp_model.CpModel, Dict[Tuple[int, int], cp_model.IntVar]]:
    """Master-модель по объёмам x[p,m] в модельных днях.

    Упрощённый вариант:
      - учитывает тип/цех (machine_type/div),
      - лимит мощности по машине,
      - коридор по объёмам для qty_minus!=0 (plan_days±1),
      - нижнюю границу по плану для qty_minus=0.

    H1–H3 и более тонкие капы пока не учитываются — это первый прототип.
    """

    model = cp_model.CpModel()

    # products_df и machines_df ожидаются в формате, близком к исходному JSON
    # (колонки idx, qty, machine_type, qty_minus, qty_minus_min, div и т.п.).
    num_products = len(products_df)

    def _safe_int(val, default: int = 0) -> int:
        """Безопасное преобразование к int с учётом NaN/None/пустых значений."""
        try:
            if val is None:
                return default
            if isinstance(val, float) and pd.isna(val):
                return default
            return int(val)
        except Exception:
            return default
    num_machines = len(machines_df)

    all_p = range(1, num_products)  # p>=1, 0 — служебный
    all_m = range(num_machines)

    # План в сменах и в модельных днях (3 смены = 1 день)
    shifts_per_day = 3
    plan_shifts = {
        p: _safe_int(products_df.iloc[p].get("qty", 0), 0) for p in all_p
    }
    plan_days = {
        p: (plan_shifts[p] + shifts_per_day - 1) // shifts_per_day
        for p in all_p
    }

    # qty_minus и qty_minus_min
    qty_minus = {
        p: _safe_int(products_df.iloc[p].get("qty_minus", 0), 0) for p in all_p
    }
    qty_minus_min_shifts = {
        p: _safe_int(products_df.iloc[p].get("qty_minus_min", 0), 0) for p in all_p
    }

    # div по продуктам/машинам
    product_divs = [
        _safe_int(products_df.iloc[p].get("div", 0), 0) for p in range(num_products)
    ]
    machine_divs = [
        _safe_int(machines_df.iloc[m].get("div", 1), 1) for m in range(num_machines)
    ]

    # machine_type / product machine_type
    product_types = [
        _safe_int(products_df.iloc[p].get("machine_type", 0), 0)
        for p in range(num_products)
    ]
    machine_types = [
        _safe_int(machines_df.iloc[m].get("type", 0), 0)
        for m in range(num_machines)
    ]

    # Переменные x[p,m]
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for p in all_p:
        for m in all_m:
            x[p, m] = model.NewIntVar(0, count_days, f"x_{p}_{m}")

    # Ограничения совместимости по type/div
    for p in all_p:
        prod_type = product_types[p]
        prod_div = product_divs[p]
        for m in all_m:
            m_type = machine_types[m]
            m_div = machine_divs[m]
            type_incompatible = prod_type > 0 and m_type != prod_type
            div_incompatible = prod_div in (1, 2) and m_div != prod_div
            if type_incompatible or div_incompatible:
                model.Add(x[p, m] == 0)

    # Мощность машин: сумма дней по всем продуктам не превышает горизонта
    for m in all_m:
        model.Add(sum(x[p, m] for p in all_p) <= count_days)

    # Объёмные ограничения по продуктам (qty_minus и коридор вокруг плана)
    max_total_days = num_machines * count_days
    deviation_terms: list[cp_model.LinearExpr] = []

    for p in all_p:
        total_p = model.NewIntVar(0, max_total_days, f"tot_{p}")
        model.Add(total_p == sum(x[p, m] for m in all_m))

        plan_d = plan_days[p]
        qm_flag = qty_minus[p]
        qm_min_shifts = qty_minus_min_shifts[p]
        min_days = (
            (qm_min_shifts + shifts_per_day - 1) // shifts_per_day
            if qm_min_shifts > 0
            else 0
        )

        # Жёсткие ограничения по объёму:
        #  - для qty_minus!=0: минимальный объём по qty_minus_min (min_days), если задан;
        #  - для qty_minus==0: пока также только минимальный объём по qty_minus_min.
        # Плановые значения plan_d учитываются ТОЛЬКО через цель |total - plan|,
        # чтобы не делать задачу невыполнимой из-за суммарной ёмкости.
        if min_days > 0:
            model.Add(total_p >= min_days)

        if plan_d <= 0:
            # Продукты без планового объёма участвуют только как мягкое отклонение,
            # но если plan_d=0, штраф за |total_p - 0| будет минимизировать их объём.
            pass

        # Вклад в цель: |total_p - plan_d|
        dev = model.NewIntVar(-max_total_days, max_total_days, f"dev_{p}")
        model.Add(dev == total_p - plan_d)
        abs_dev = model.NewIntVar(0, max_total_days, f"absdev_{p}")
        model.AddAbsEquality(abs_dev, dev)

        # Вес: строже штрафуем строгие продукты.
        w = 2 if qm_flag == 0 else 1
        deviation_terms.append(abs_dev * w)

    # Цель: минимизировать суммарное отклонение по дням
    if deviation_terms:
        model.Minimize(sum(deviation_terms))
    else:
        model.Minimize(0)

    return model, x, plan_days


def solve_master(model: cp_model.CpModel, x, time_limit: int) -> Dict[Tuple[int, int], int]:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = settings.LOOM_NUM_WORKERS
    status = solver.Solve(model)

    print("MASTER status:", solver.StatusName(status))
    print("MASTER objective:", solver.ObjectiveValue())

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Мастер-модель не нашла допустимого решения")

    x_val: Dict[Tuple[int, int], int] = {}
    for (p, m), var in x.items():
        x_val[p, m] = solver.Value(var)
    return x_val


def _safe_int(val, default: int = 0) -> int:
    """Безопасное преобразование к int с учётом NaN/None/пустых значений (глобальная версия)."""
    try:
        if val is None:
            return default
        if isinstance(val, float) and pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def build_machine_sequence(
    m_idx: int,
    x_pm: Dict[int, int],
    products_df,
    machines_df,
    count_days: int,
) -> list[int]:
    """Жадная раскладка по дням для одной машины.

    Не гарантирует строгое совпадение с x_pm, но старается его соблюсти.
    INDEX_UP: продукты по idx не убывают.
    """

    num_products = len(products_df)
    all_p = range(1, num_products)

    # Копия остатков по дням
    remaining = dict(x_pm)

    # Утилита: проверить совместимость по type/div
    prod_type = [
        _safe_int(products_df.iloc[p].get("machine_type", 0), 0)
        for p in range(num_products)
    ]
    prod_div = [
        _safe_int(products_df.iloc[p].get("div", 0), 0)
        for p in range(num_products)
    ]
    m_type = _safe_int(machines_df.iloc[m_idx].get("type", 0), 0)
    m_div = _safe_int(machines_df.iloc[m_idx].get("div", 1), 1)

    def can_run(p: int) -> bool:
        if p <= 0 or p >= num_products:
            return False
        pt = prod_type[p]
        pd = prod_div[p]
        if pt > 0 and m_type != pt:
            return False
        if pd in (1, 2) and m_div != pd:
            return False
        return True

    seq: list[int | None] = [None for _ in range(count_days)]

    # День 0: предпочтительно продукт с max remaining
    best_p = None
    best_rem = -1
    for p in all_p:
        if not can_run(p):
            continue
        if remaining.get(p, 0) > best_rem:
            best_rem = remaining[p]
            best_p = p
    if best_p is not None and best_rem > 0:
        seq[0] = best_p
        remaining[best_p] -= 1
    else:
        seq[0] = next((p for p in all_p if can_run(p)), 1)

    # Остальные дни
    for d in range(1, count_days):
        prev_p = seq[d - 1]
        chosen = None

        # 1) Пытаемся продолжать предыдущий продукт
        if prev_p is not None and prev_p > 0 and can_run(prev_p):
            if remaining.get(prev_p, 0) > 0:
                chosen = prev_p

        # 2) Иначе выбираем продукт с max remaining среди p >= prev_p
        if chosen is None:
            start_idx = prev_p if (prev_p is not None and prev_p > 0) else 1
            best_p = None
            best_rem = -1
            for p in range(start_idx, num_products):
                if not can_run(p):
                    continue
                rem = remaining.get(p, 0)
                if rem > best_rem:
                    best_rem = rem
                    best_p = p
            if best_p is not None and best_rem > 0:
                chosen = best_p

        # 3) Если план исчерпан, продолжаем prev_p или берём первый совместимый
        if chosen is None:
            if prev_p is not None and prev_p > 0 and can_run(prev_p):
                chosen = prev_p
            else:
                chosen = next((p for p in all_p if can_run(p)), 1)

        seq[d] = chosen
        if chosen is not None and chosen > 0:
            if remaining.get(chosen, 0) > 0:
                remaining[chosen] -= 1

    # Заменяем None на первый допустимый продукт (на всякий случай)
    for d in range(count_days):
        if seq[d] is None or seq[d] <= 0:
            seq[d] = next((p for p in all_p if can_run(p)), 1)

    return [int(p) for p in seq]


def compute_external_prop_penalties(data, seq_by_machine: list[list[int]]):
    """Подсчёт внешнего пропорционального штрафа, как в analyze_ls_prop_noIndex_H123_by_product.

    seq_by_machine: список машин, для каждой список продуктов по дням (idx из JSON).
    """

    products = data["products"]
    count_days = len(seq_by_machine[0]) if seq_by_machine else 0
    shifts_per_day = 3

    # План в сменах и факт в сменах
    plan: Dict[int, int] = {}
    fact: Dict[int, int] = {}

    for idx, prod in enumerate(products):
        if idx == 0:
            continue
        plan[idx] = int(prod.get("qty", 0) or 0)
        fact[idx] = 0

    for m_idx, seq in enumerate(seq_by_machine):
        for d, p_internal in enumerate(seq):
            # p_internal — внутренний idx после сортировок мы не строили,
            # поэтому предполагаем, что seq_by_machine использует исходные idx.
            # В этой простой версии будем считать, что внутренний и внешний idx совпадают.
            idx = int(p_internal)
            if idx <= 0 or idx not in fact:
                continue
            fact[idx] += shifts_per_day

    # Внешний штраф
    total_plan = sum(v for v in plan.values() if v > 0)
    total_fact = sum(fact[idx] for idx, v in plan.items() if v > 0)
    prop_penalty = 0
    per_prod_penalty: Dict[int, int] = {}

    if total_plan > 0 and total_fact > 0:
        for idx, p_plan in plan.items():
            if p_plan <= 0:
                continue
            f = fact.get(idx, 0)
            term1 = f * total_plan
            term2 = total_fact * p_plan
            contrib = abs(term1 - term2)
            per_prod_penalty[idx] = contrib
            prop_penalty += contrib

    return prop_penalty, per_prod_penalty, plan, fact


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines_tuples, products_tuples, cleans, remains = load_input(Path(input_path))

    # Для простоты берём DataFrame прямо из JSON
    import pandas as pd

    products_df = pd.DataFrame(data["products"]).copy()
    machines_df = pd.DataFrame(data["machines"]).copy()

    # Горизонт LONG_SIMPLE: агрегируем каждые 3 смены в 1 день
    count_days_raw = int(data["count_days"])
    shifts_per_day = 3
    count_days = (count_days_raw + shifts_per_day - 1) // shifts_per_day

    print(f"TWOLEVEL: count_days_raw={count_days_raw}, count_days_model={count_days}")

    # Строим и решаем master-модель
    model, x, plan_days = build_master_model(data, products_df, machines_df, count_days)
    x_val = solve_master(model, x, time_limit=int(settings.LOOM_MAX_TIME))

    # Собираем жёсткое распределение по машинам
    seq_by_machine: list[list[int]] = []
    num_machines = len(machines_df)
    for m in range(num_machines):
        x_pm = {p: x_val.get((p, m), 0) for p in range(1, len(products_df))}
        seq = build_machine_sequence(m, x_pm, products_df, machines_df, count_days)
        seq_by_machine.append(seq)

    # Оценка внешнего пропорционального штрафа
    total_prop_penalty, per_prod_penalty, plan, fact = compute_external_prop_penalties(
        data, seq_by_machine
    )

    print(f"\nВнешний пропорциональный штраф (TWOLEVEL): {total_prop_penalty}")
    print("idx\tname\tplan_shifts\tfact_shifts\tprop_penalty")
    for idx, prod in enumerate(data["products"]):
        if idx == 0:
            continue
        name = prod["name"]
        p_plan = plan.get(idx, 0)
        p_fact = fact.get(idx, 0)
        pen = per_prod_penalty.get(idx, 0)
        print(f"{idx}\t{name}\t{p_plan}\t{p_fact}\t{pen}")


if __name__ == "__main__":  # pragma: no cover
    main()
