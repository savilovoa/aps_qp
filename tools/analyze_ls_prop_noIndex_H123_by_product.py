from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def run_ls_prop_noindex_h123():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настройки: LS_PROP в LONG_SIMPLE, qty_minus включен, INDEX_UP выключен только в SIMPLE,
    # эвристики H1+H2+H3 включены.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    # settings.LOOM_MAX_TIME не трогаем здесь, чтобы можно было управлять
    # лимитом времени снаружи (через config/env). Это позволяет запускать
    # один и тот же сценарий при разных тайм-лимитах.

    settings.APPLY_QTY_MINUS = True

    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True
    settings.APPLY_STRATEGY_PENALTY = True

    settings.APPLY_INDEX_UP = True
    settings.SIMPLE_DISABLE_INDEX_UP = True

    settings.SIMPLE_DEBUG_H_START = True
    settings.SIMPLE_DEBUG_H_MODE = "H123"

    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    # Сбрасываем debug-флаги
    settings.SIMPLE_DISABLE_INDEX_UP = False
    settings.SIMPLE_DEBUG_H_START = False
    settings.SIMPLE_DEBUG_H_MODE = None

    if not isinstance(res, dict):
        raise RuntimeError("Ожидался dict из schedule_loom_calc в режиме отладки")

    return data, machines, res


def compute_external_prop_penalties(data, prod_stats: list[dict]):
    # prod_stats: элементы из res["products"]
    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
        }

    total_plan = sum(v["plan"] for v in stats.values() if v["plan"] > 0)
    total_fact = sum(v["fact"] for v in stats.values() if v["plan"] > 0)
    prop_penalty = 0
    per_prod_penalty: dict[int, int] = {}
    if total_plan > 0 and total_fact > 0:
        for idx, v in stats.items():
            plan = v["plan"]
            if plan <= 0:
                continue
            fact = v["fact"]
            term1 = fact * total_plan
            term2 = total_fact * plan
            contrib = abs(term1 - term2)
            per_prod_penalty[idx] = contrib
            prop_penalty += contrib
    return prop_penalty, per_prod_penalty


def compute_transitions_per_product_and_day(res: dict, machines: list[tuple]):
    schedule = res["schedule"]
    if not schedule:
        return {}, {}

    max_day = max(s["day_idx"] for s in schedule)
    num_days = max_day + 1

    # (m,d) -> product_idx (в ИСХОДНОМ пространстве idx продуктов)
    md: dict[tuple[int, int], int] = {}
    for rec in schedule:
        md[(rec["machine_idx"], rec["day_idx"])] = rec["product_idx"]

    transitions_per_product: dict[int, int] = {}
    transitions_per_day: dict[int, int] = {d: 0 for d in range(num_days)}

    for m_idx, m_data in enumerate(machines):
        # Переход в день 0: от начального продукта на машине к продукту в день 0.
        # machines[m] = (name, product_idx, id, type, remain_day, reserve)
        # product_id == 0 трактуем как "нет продукта", переходы с/на 0 не штрафуем.
        init_p = m_data[1] if len(m_data) > 1 else None
        p0 = md.get((m_idx, 0), None)
        if (
            init_p is not None
            and p0 is not None
            and init_p > 0
            and p0 > 0
            and p0 != init_p
        ):
            transitions_per_product[p0] = transitions_per_product.get(p0, 0) + 1
            transitions_per_day[0] = transitions_per_day.get(0, 0) + 1

        # Переходы между днями d-1 и d (d >= 1)
        for d in range(1, num_days):
            p_prev = md.get((m_idx, d - 1))
            p_cur = md.get((m_idx, d))
            if p_prev is None or p_cur is None:
                continue
            # product_id == 0 трактуем как "нет продукта" — такие переходы не считаем
            if p_prev <= 0 or p_cur <= 0:
                continue
            if p_cur != p_prev:
                transitions_per_product[p_cur] = transitions_per_product.get(p_cur, 0) + 1
                transitions_per_day[d] = transitions_per_day.get(d, 0) + 1

    return transitions_per_product, transitions_per_day


def main() -> None:
    data, machines, res = run_ls_prop_noindex_h123()

    schedule = res["schedule"]
    max_day = max(s["day_idx"] for s in schedule) if schedule else -1
    num_days = max_day + 1 if max_day >= 0 else 0
    shifts_per_day = 3
    half_machine_period_shifts = (num_days * shifts_per_day) // 2

    # Карта: продукт -> список машин, где он стоит на начало (оригинальные idx из входа).
    machines_for_prod: dict[int, list[int]] = {}
    for m_idx, m in enumerate(machines):
        # machines[m] = (name, product_idx, id, type, remain_day, reserve)
        if len(m) > 1:
            p0 = m[1]
            machines_for_prod.setdefault(p0, []).append(m_idx)

    status = res["status"]
    status_str = res["status_str"]
    print(f"status={status} ({status_str})")

    prod_stats: list[dict] = res["products"]

    # Внутренняя qty теперь уже в сменах (мы умножаем модельные дни на 3 в LONG_SIMPLE).
    # Посчитаем внешний пропорциональный штраф, как в compare_long_vs_simple.
    total_prop_penalty, per_prod_prop = compute_external_prop_penalties(data, prod_stats)
    print(f"\nВнешний пропорциональный штраф (LS_PROP_noIndex_H123): {total_prop_penalty}")

    transitions_per_product, transitions_per_day = compute_transitions_per_product_and_day(res, machines)
    trans_weight = max(1, settings.KFZ_DOWNTIME_PENALTY // 2)

    print(f"\nВес штрафа за переход (per change): {trans_weight}")

    # Сведём по продуктам: план, факт (в сменах), qty_minus, штрафы по типам.
    print("\nПо продуктам: idx\tname\tplan_shifts\tfact_shifts\tqty_minus\tqty_minus_min\tpen_prop_ext\ttransitions\tpen_trans\tpen_strategy\tstrict_small_H2_cap_days")
    stats_by_idx = {ps["product_idx"]: ps for ps in prod_stats}

    rows_summary = []

    for idx, prod in enumerate(data["products"]):
        if idx == 0:
            continue
        name = prod["name"]
        qty_minus_flag = int(prod.get("qty_minus", 0))
        qty_minus_min = int(prod.get("qty_minus_min", 0) or 0)
        ps = stats_by_idx.get(idx)
        if not ps:
            continue
        plan_shifts = int(ps["plan_qty"])
        fact_shifts = int(ps["qty"])
        pen_prop_ext = per_prod_prop.get(idx, 0)
        trans_cnt = transitions_per_product.get(idx, 0)
        pen_trans = trans_cnt * trans_weight
        pen_strategy = int(ps.get("penalty_strategy", 0))

        # Проверка кандидата на новый кап: строгий продукт, есть ровно одна стартовая машина,
        # плановый объём меньше половины машино-периода.
        machines_for_p = machines_for_prod.get(idx, [])
        if (
            qty_minus_flag == 0
            and plan_shifts > 0
            and len(machines_for_p) == 1
            and plan_shifts < half_machine_period_shifts
        ):
            # Приблизительный cap по дням, как в модели: план_days (+1 день, если кратно 3 сменам).
            plan_days = (plan_shifts + shifts_per_day - 1) // shifts_per_day
            extra_day = 1 if (plan_shifts % shifts_per_day == 0) else 0
            cap_days_candidate = plan_days + extra_day
        else:
            cap_days_candidate = 0

        print(
            f"{idx}\t{name}\t{plan_shifts}\t{fact_shifts}\t{qty_minus_flag}\t{qty_minus_min}\t"
            f"{pen_prop_ext}\t{trans_cnt}\t{pen_trans}\t{pen_strategy}\t{cap_days_candidate}"
        )

        total_pen = pen_prop_ext + pen_trans + pen_strategy
        rows_summary.append(
            (
                idx,
                name,
                plan_shifts,
                fact_shifts,
                qty_minus_flag,
                qty_minus_min,
                pen_prop_ext,
                trans_cnt,
                pen_trans,
                pen_strategy,
                total_pen,
                cap_days_candidate,
            )
        )

    # Итоги переходов по дням.
    print("\nИтоги переходов по дням (модельные дни LONG_SIMPLE): day\ttransitions")
    for d in sorted(transitions_per_day.keys()):
        print(f"{d}\t{transitions_per_day[d]}")

    # Топ продуктов по суммарному штрафу.
    rows_summary.sort(key=lambda r: r[-2], reverse=True)
    print("\nTOP-15 продуктов по суммарному штрафу (prop_ext + trans + strategy):")
    print("idx\tname\tplan\tfact\tqty_minus\tqty_minus_min\tpen_prop_ext\tpen_trans\tpen_strategy\ttotal_pen\tstrict_small_H2_cap_days")
    for r in rows_summary[:15]:
        idx, name, plan_shifts, fact_shifts, qm, qm_min, pen_prop_ext, trans_cnt, pen_trans, pen_strat, total_pen, cap_days_candidate = r
        print(
            f"{idx}\t{name}\t{plan_shifts}\t{fact_shifts}\t{qm}\t{qm_min}\t"
            f"{pen_prop_ext}\t{pen_trans}\t{pen_strat}\t{total_pen}\t{cap_days_candidate}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
