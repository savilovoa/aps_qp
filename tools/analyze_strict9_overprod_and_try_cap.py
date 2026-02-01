from __future__ import annotations

from pathlib import Path

from ortools.sat.python import cp_model

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


STRICT9 = [5, 4, 6, 7, 8, 10, 9, 13, 11]
STRICT10 = STRICT9 + [22]


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def run_long_simple_with_subset(
    data,
    machines,
    products,
    cleans,
    remains,
    subset: list[int],
):
    """Запуск LONG_SIMPLE с заданным подмножеством строгих qty_minus продуктов."""
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = True
    # subset задаётся во внешних idx; маппинг во внутренние idx делает schedule_loom_calc.
    settings.SIMPLE_QTY_MINUS_SUBSET = set(subset)

    # Внутренняя пропорциональная цель LS_OVER.
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False
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
        products_stats = res.get("products", [])
        objective_value = res.get("objective_value", 0)
        proportion_diff = res.get("proportion_diff", 0)
    else:
        status = int(res.status)
        status_str = str(res.status_str)
        schedule = res.schedule
        products_stats = res.products
        objective_value = getattr(res, "objective_value", 0)
        proportion_diff = getattr(res, "proportion_diff", 0)

    return status, status_str, schedule, products_stats, objective_value, proportion_diff


def extract_plan_fact(products_stats, idxs: list[int]) -> dict[int, tuple[int, int]]:
    res: dict[int, tuple[int, int]] = {}
    for ps in products_stats:
        p_idx = int(ps.get("product_idx", -1))
        if p_idx in idxs:
            plan = int(ps.get("plan_qty", 0))
            fact = int(ps.get("qty", 0))
            res[p_idx] = (plan, fact)
    return res


def main() -> None:
    data, machines, products, cleans, remains = load_data()

    # 1. Базовый запуск STRICT9 без верхних капов.
    print("=== LONG_SIMPLE STRICT9 (без верхних капов) ===")
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = None

    status9, status_str9, _sched9, stats9, obj9, prop9 = run_long_simple_with_subset(
        data, machines, products, cleans, remains, STRICT9
    )
    print(f"status={status9} ({status_str9}), obj={obj9}, prop={prop9}")

    if status9 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("STRICT9 невыполним даже без верхних капов — дальнейший анализ не имеет смысла.")
        return

    # Собираем план/факт для всех продуктов из входного JSON.
    # В stats9 значения:
    #   plan_qty — в СМЕНАХ (как во входном JSON),
    #   qty      — тоже в СМЕНАХ.
    # Для SIMPLE (LONG_SIMPLE) 1 модельный день = 3 сменам, поэтому для анализа
    # перепроизводства по дням переводим обе величины в ДНИ.
    pf_all = extract_plan_fact(stats9, [int(p["idx"]) for p in data["products"]])

    # 2. Находим продукты с qty_minus=0 (строгие) и положительным (fact_days - plan_days),
    # сортируем по убыванию перепроизводства в ДНЯХ.
    # strict_over: (idx, name, plan_shifts, fact_shifts, plan_days, fact_days, diff_days)
    strict_over = []
    for p in data["products"]:
        idx = int(p["idx"])
        if idx == 0:
            continue
        qm = p.get("qty_minus", 0)
        if qm != 0 and qm is not False:
            continue
        plan_shifts, fact_shifts = pf_all.get(idx, (0, 0))
        # Переводим смены в дни: plan_days по ceil, fact_days — точно (qty всегда кратно 3).
        if plan_shifts > 0:
            plan_days = (plan_shifts + 3 - 1) // 3
        else:
            plan_days = 0
        if fact_shifts > 0:
            fact_days = fact_shifts // 3
        else:
            fact_days = 0
        diff_days = fact_days - plan_days
        if diff_days <= 0:
            continue
        strict_over.append((idx, p["name"], plan_shifts, fact_shifts, plan_days, fact_days, diff_days))

    if not strict_over:
        print("Нет строгих продуктов с положительной разницей fact-plan (в днях).")
    else:
        strict_over.sort(key=lambda t: t[6], reverse=True)
        print("\nСтрогие продукты с fact>plan (по убыванию diff_days):")
        for idx, name, plan_sh, fact_sh, plan_d, fact_d, diff_d in strict_over:
            print(
                f"  idx={idx}, name={name}, plan_shifts={plan_sh}, fact_shifts={fact_sh}, "
                f"plan_days≈{plan_d}, fact_days={fact_d}, diff_days={diff_d}"
            )

    if not strict_over:
        return

    # 3. Последовательно добавляем капы для 1, 2, 3 продуктов с максимальной
    # разницей fact_days - plan_days. На каждом шаге пересобираем STRICT9 и
    # проверяем, что модель остаётся FEASIBLE.
    top_k = min(3, len(strict_over))
    current_caps: dict[int, int] = {}
    last_feasible_caps: dict[int, int] = {}

    print("\n=== Поэтапное добавление верхних капов для строгих продуктов (в днях) ===")
    for step in range(1, top_k + 1):
        idx_i, name_i, plan_sh_i, fact_sh_i, plan_d_i, fact_d_i, diff_d_i = strict_over[step - 1]
        # Кап задаём в ДНЯХ как plan_days + 1.
        cap_days_i = plan_d_i + 1
        if cap_days_i < 0:
            cap_days_i = 0
        current_caps[idx_i] = cap_days_i

        # Логируем текущий набор капов.
        print(f"\nШаг {step}: добавляем/обновляем капы для следующих продуктов:")
        for c_idx, c_cap in current_caps.items():
            print(f"  cap: idx={c_idx} -> cap_days={c_cap}")

        settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = dict(current_caps)

        status_s, status_str_s, _sched_s, _stats_s, obj_s, prop_s = run_long_simple_with_subset(
            data, machines, products, cleans, remains, STRICT9
        )
        print(
            f"Результат STRICT9 с капами (шаг {step}): status={status_s} ({status_str_s}), "
            f"obj={obj_s}, prop={prop_s}"
        )

        if status_s not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(
                "Модель стала невыполнимой после добавления капа на шаге",
                step,
                ". Останавливаемся.",
            )
            break

        last_feasible_caps = dict(current_caps)

    if last_feasible_caps:
        print("\nПоследний выполнимый набор верхних капов для STRICT9 (в днях):")
        for c_idx, c_cap in last_feasible_caps.items():
            print(f"  idx={c_idx} -> cap_days={c_cap}")
    else:
        print("\nНи один из шагов с добавлением верхних капов не дал FEASIBLE для STRICT9.")


if __name__ == "__main__":  # pragma: no cover
    main()
