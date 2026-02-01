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


def run_long_simple(
    data,
    machines,
    products,
    cleans,
    remains,
):
    """Запуск LONG_SIMPLE без qty_minus (SIMPLE_QTY_MINUS_SUBSET = {}),
    с текущими настройками SIMPLE_DEBUG_PRODUCT_UPPER_CAPS."""
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = True
    # Пустой поднабор: qty_minus-блок не накладывает нижних границ ни на один продукт.
    settings.SIMPLE_QTY_MINUS_SUBSET = set()

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


def extract_plan_fact(products_stats):
    pf: dict[int, tuple[int, int]] = {}
    for ps in products_stats:
        p_idx = int(ps.get("product_idx", -1))
        plan = int(ps.get("plan_qty", 0))
        fact = int(ps.get("qty", 0))
        pf[p_idx] = (plan, fact)
    return pf


def make_html_for_schedule(path: Path, data: dict, schedule: list[dict], title_text: str) -> None:
    # Агрегируем schedule в long_schedule (день × продукт → количество машин)
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

    # 1) Проверка: одиночный кап на idx=26 без qty_minus нижних границ,
    # с включённой монотонностью C[p,d] (SIMPLE_USE_MONOTONE_COUNTS=True).
    print("=== Проверка: кап idx=26 (cap_days=29), SIMPLE_QTY_MINUS_SUBSET = {}, MONOTONE_COUNTS=ON ===")
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = {26: 29}
    settings.SIMPLE_USE_MONOTONE_COUNTS = True

    status26, status_str26, _s26, _ps26, obj26, prop26 = run_long_simple(
        data, machines, products, cleans, remains
    )
    print(f"status={status26} ({status_str26}), obj={obj26}, prop={prop26}")

    # 2) Та же конфигурация, но без монотонности C[p,d]
    # (SIMPLE_USE_MONOTONE_COUNTS=False) для проверки её влияния.
    print("\n=== Проверка: тот же кап idx=26 (cap_days=29), SIMPLE_QTY_MINUS_SUBSET = {}, MONOTONE_COUNTS=OFF ===")
    settings.SIMPLE_USE_MONOTONE_COUNTS = False

    status26b, status_str26b, sched26b, stats26b, obj26b, prop26b = run_long_simple(
        data, machines, products, cleans, remains
    )
    print(f"status={status26b} ({status_str26b}), obj={obj26b}, prop={prop26b}")

    if status26b in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Печатаем план/факт и профиль по дням для продукта idx=26.
        pf = extract_plan_fact(stats26b)
        plan_26, fact_26 = pf.get(26, (0, 0))
        print(f"\nidx=26: plan={plan_26} смен, fact={fact_26} смен")

        # Профиль по дням: сколько машин в каждый день занято idx=26.
        day_counts: dict[int, int] = {}
        for rec in sched26b:
            p_idx = rec.get("product_idx")
            d_idx = rec.get("day_idx")
            if p_idx == 26:
                day_counts[d_idx] = day_counts.get(d_idx, 0) + 1

        print("\nПрофиль idx=26 по дням (модельные дни -> число машин):")
        for d in sorted(day_counts.keys()):
            print(f"  day={d}: machines={day_counts[d]}")

        out_html = Path("example/res_cap26_no_monotone.html")
        title = (
            f"LONG_SIMPLE cap idx=26->29, no qty_minus subset, MONOTONE_COUNTS=OFF, "
            f"status={status_str26b}, obj={obj26b}, prop={prop26b}"
        )
        make_html_for_schedule(out_html, data, sched26b, title)
        print(f"HTML сохранён в {out_html}")
    else:
        print("\nДаже при MONOTONE_COUNTS=OFF модель с капом idx=26 невыполнима.")


if __name__ == "__main__":  # pragma: no cover
    main()
