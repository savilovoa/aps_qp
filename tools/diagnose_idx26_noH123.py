from __future__ import annotations

from pathlib import Path
from datetime import datetime

from ortools.sat.python import cp_model

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from src.loom.loom_plan_html import aggregated_schedule_to_html
from tools.compare_long_vs_simple import load_input


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def run_long_simple(data, machines, products, cleans, remains):
    """Запуск LONG_SIMPLE с отключёнными H1-H3 и только верхним капом для idx=26."""
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300

    # Включаем qty_minus, но нижние границы ни для кого не задаём.
    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = set()

    # Включаем отладочный дамп ограничений для внешнего idx=26.
    settings.SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX = 26

    # Отключаем H1-H3 через debug-режим.
    settings.SIMPLE_DEBUG_H_START = True
    settings.SIMPLE_DEBUG_H_MODE = "NONE"  # ни H1, ни H2, ни H3

    # Оставляем INDEX_UP включённым, сегменты и прочую бизнес-логику как есть.
    settings.APPLY_INDEX_UP = True

    # Монотонность C[p,d] пока не трогаем (берём текущее значение из config).

    # Единственное продукт-уровневое ограничение сверху — на idx=26 (во внешних idx).
    # Кап задаётся в ДНЯХ: план≈28 дней (82 смены), берём план+1=29.
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = {26: 29}

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

    print("=== Диагностика: только верхний кап idx=26 (29 дней), H1-H3 отключены, SIMPLE_QTY_MINUS_SUBSET = {} ===")
    status, status_str, sched, stats, obj, prop = run_long_simple(
        data, machines, products, cleans, remains
    )
    print(f"status={status} ({status_str}), obj={obj}, prop={prop}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Модель невыполнима даже при отключённых H1-H3 и единственном капе для idx=26.")
        return

    pf = extract_plan_fact(stats)
    plan_26, fact_26 = pf.get(26, (0, 0))
    print(f"\nidx=26: plan={plan_26} смен, fact={fact_26} смен")

    # Профиль по дням: сколько машин в каждый день занято idx=26.
    day_counts: dict[int, int] = {}
    for rec in sched:
        p_idx = rec.get("product_idx")
        d_idx = rec.get("day_idx")
        if p_idx == 26:
            day_counts[d_idx] = day_counts.get(d_idx, 0) + 1

    print("\nПрофиль idx=26 по дням (модельные дни -> число машин):")
    for d in sorted(day_counts.keys()):
        print(f"  day={d}: machines={day_counts[d]}")

    out_html = Path("example/res_idx26_cap_only_noH123.html")
    title = (
        f"LONG_SIMPLE cap idx=26->29, no qty_minus subset, H1-H3 OFF, "
        f"status={status_str}, obj={obj}, prop={prop}"
    )
    make_html_for_schedule(out_html, data, sched, title)
    print(f"HTML сохранён в {out_html}")


if __name__ == "__main__":  # pragma: no cover
    main()
