from __future__ import annotations

from pathlib import Path
from math import ceil
from datetime import datetime

from ortools.sat.python import cp_model

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from src.loom.loom_plan_html import aggregated_schedule_to_html
from tools.compare_long_vs_simple import load_input


QTY_MINUS_SUBSET = {5, 4, 6, 7, 8, 10, 9, 13, 11}


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def run_long_simple_with_cap(data, machines, products, cleans, remains, cap_days: int):
    """Запуск LONG_SIMPLE с включённым qty_minus для заданного подмножества
    продуктов и верхним капом для внешнего idx=26 = cap_days (в ДНЯХ).
    """
    # Базовые настройки LONG_SIMPLE
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300

    # Включаем qty_minus и задаём отладочный поднабор продуктов.
    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = set(QTY_MINUS_SUBSET)

    # Отключаем отладочный контроль H1–H3: используем обычную логику.
    settings.SIMPLE_DEBUG_H_START = False
    settings.SIMPLE_DEBUG_H_MODE = None

    # Индекс только вверх и прочая логика как в боевой конфигурации.
    settings.APPLY_INDEX_UP = True

    # Верхний кап для внешнего idx=26 (перепривяжется к внутреннему в schedule_loom_calc).
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = {26: int(cap_days)}

    # Пропорциональная цель как в LS_OVER.
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = False

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


def extract_plan_fact(products_stats):
    pf: dict[int, tuple[int, int]] = {}
    for ps in products_stats:
        p_idx = int(ps.get("product_idx", -1))
        plan = int(ps.get("plan_qty", 0))
        fact = int(ps.get("qty", 0))
        pf[p_idx] = (plan, fact)
    return pf


def make_html_for_schedule(path: Path, data: dict, schedule: list[dict], title_text: str) -> None:
    counts: dict[tuple[int, int], int] = {}
    for s in schedule:
        p_idx = s["product_idx"]
        d_idx = s["day_idx"]
        if p_idx is None or p_idx <= 0:
            continue
        key = (d_idx, p_idx)
        counts[key] = counts.get(key, 0) + 1

    long_schedule = [
        {"day_idx": d, "product_idx": p, "machine_count": c}
        for (d, p), c in sorted(counts.items())
    ]

    dt_begin = datetime.strptime(data["dt_begin"], "%Y-%m-%dT%H:%M:%S").date()
    html = aggregated_schedule_to_html(
        machines=data["machines"],
        schedule=schedule,
        products=data["products"],
        long_schedule=long_schedule,
        dt_begin=dt_begin,
        title_text=title_text,
    )
    path.write_text(html, encoding="utf-8")


def main() -> None:
    data, machines, products, cleans, remains = load_data()

    # План для внешнего idx=26 в сменах и днях.
    try:
        plan_shifts_26 = int(products[26][1])
    except Exception:
        plan_shifts_26 = 0
    plan_days_26 = ceil(plan_shifts_26 / 3) if plan_shifts_26 > 0 else 0

    print(f"План для idx=26: {plan_shifts_26} смен (~{plan_days_26} дней)")
    print(f"SIMPLE_QTY_MINUS_SUBSET = {sorted(QTY_MINUS_SUBSET)}")

    # Проверяем капы: план+1, план+2, план+3 (в днях).
    base = plan_days_26
    caps = [base + 1, base + 2, base + 3]

    for cap in caps:
        print("\n=== Диагностика: кап idx=26 =", cap,
              "дней, qty_minus subset=", sorted(QTY_MINUS_SUBSET), "===")
        status, status_str, sched, stats, obj, prop = run_long_simple_with_cap(
            data, machines, products, cleans, remains, cap_days=cap
        )
        print(f"status={status} ({status_str}), obj={obj}, prop={prop}")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("Модель НЕ выполнима при таком капе.")
            continue

        pf = extract_plan_fact(stats)
        plan_26, fact_26 = pf.get(26, (0, 0))
        print(f"idx=26: plan={plan_26} смен, fact={fact_26} смен")

        # Профиль по дням: сколько машин в каждый день занято idx=26.
        day_counts: dict[int, int] = {}
        for rec in sched:
            p_idx = rec.get("product_idx")
            d_idx = rec.get("day_idx")
            if p_idx == 26:
                day_counts[d_idx] = day_counts.get(d_idx, 0) + 1

        print("Профиль idx=26 по дням (день -> число машин):")
        for d in sorted(day_counts.keys()):
            print(f"  day={d}: machines={day_counts[d]}")

        out_html = Path(f"example/res_idx26_cap_{cap}_subset.html")
        title = (
            f"LONG_SIMPLE cap idx=26->{cap}, qty_minus subset={sorted(QTY_MINUS_SUBSET)}, "
            f"status={status_str}, obj={obj}, prop={prop}"
        )
        make_html_for_schedule(out_html, data, sched, title)
        print(f"HTML сохранён в {out_html}")


if __name__ == "__main__":  # pragma: no cover
    main()
