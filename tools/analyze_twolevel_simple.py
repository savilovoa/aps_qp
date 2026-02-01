from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input
from tools.analyze_ls_prop_noIndex_H123_by_product import (
    compute_external_prop_penalties,
    compute_transitions_per_product_and_day,
)


def run_twolevel():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Двухуровневый режим по агрегированным дням.
    settings.HORIZON_MODE = "LONG_TWOLEVEL"

    # Остальные флаги оставляем в значениях по умолчанию, чтобы не мешать
    # двухуровневой логике. Внутренние цели LS_PROP/OVER в этом режиме не
    # используются — мы считаем внешний пропорциональный штраф снаружи.
    settings.APPLY_QTY_MINUS = True
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True
    settings.APPLY_STRATEGY_PENALTY = True

    settings.APPLY_INDEX_UP = True
    settings.SIMPLE_DISABLE_INDEX_UP = False

    # Эвристики H1–H3 относятся только к LONG_SIMPLE; здесь они не используются.
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

    if not isinstance(res, dict):
        raise RuntimeError("Ожидался dict из schedule_loom_calc в режиме LONG_TWOLEVEL")

    return data, machines, res


def main() -> None:
    data, machines, res = run_twolevel()

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

    # Посчитаем внешний пропорциональный штраф, как в analyze_ls_prop_noIndex_H123_by_product.
    total_prop_penalty, per_prod_prop = compute_external_prop_penalties(data, prod_stats)
    print(f"\nВнешний пропорциональный штраф (TWOLEVEL): {total_prop_penalty}")

    transitions_per_product, transitions_per_day = compute_transitions_per_product_and_day(res, machines)
    trans_weight = max(1, settings.KFZ_DOWNTIME_PENALTY // 2)

    print(f"\nВес штрафа за переход (per change): {trans_weight}")

    # Сводка по продуктам: план, факт (в сменах), qty_minus, штрафы по типам.
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

        # Кандидат на строгий cap по H2-логике (для справки):
        machines_for_p = machines_for_prod.get(idx, [])
        if (
            qty_minus_flag == 0
            and plan_shifts > 0
            and len(machines_for_p) == 1
            and plan_shifts < half_machine_period_shifts
        ):
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
    print("\nИтоги переходов по дням (модельные дни LONG_TWOLEVEL): day\ttransitions")
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
