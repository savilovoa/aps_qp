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
TARGET_IDX = 20  # внешний idx продукта


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def run_long_simple_once(
    data,
    machines,
    products,
    cleans,
    remains,
    title: str,
    html_name: str,
):
    """Запуск LONG_SIMPLE и формирование агрегированного HTML."""
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

    print(f"{title}: status={status} ({status_str}), obj={objective_value}, prop={proportion_diff}")

    # короткая сводка по целевому продукту
    p_plan = p_fact = None
    for ps in products_stats:
        p_idx = int(ps.get("product_idx", -1))
        if p_idx != TARGET_IDX:
            continue
        p_plan = int(ps.get("plan_qty", 0))
        p_fact = int(ps.get("qty", 0))
        break
    if p_plan is not None:
        print(f"  idx={TARGET_IDX}: plan={p_plan}, fact={p_fact}")

    # Агрегированный long_schedule по дням/продуктам (кол-во машин)
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
        title_text=title,
    )

    out_html = Path("example") / html_name
    out_html.write_text(html, encoding="utf-8")
    print(f"  HTML сохранён в {out_html}")

    return status, status_str


def main() -> None:
    data, machines, products, cleans, remains = load_data()

    # Базовые настройки LONG_SIMPLE
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = set(QTY_MINUS_SUBSET)
    # Используем div-ограничения по умолчанию (как в модели), hints не включаем.

    # --- Базовый прогон без капа для TARGET_IDX ---
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = None
    settings.SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX = None

    print(f"=== LONG_SIMPLE baseline (без капа для idx={TARGET_IDX}) ===")
    base_status, base_status_str = run_long_simple_once(
        data,
        machines,
        products,
        cleans,
        remains,
        title=f"baseline LONG_SIMPLE (no cap for idx={TARGET_IDX})",
        html_name=f"res_idx{TARGET_IDX}_baseline.html",
    )

    if base_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Базовая модель невыполнима, диагностика капа бессмысленна.")
        return

    # --- Прогон с капом plan_days+2 для TARGET_IDX ---
    # план в сменах берём из products[TARGET_IDX][1]
    try:
        plan_shifts = int(products[TARGET_IDX][1])
    except Exception:
        plan_shifts = 0
    plan_days = ceil(plan_shifts / 3) if plan_shifts > 0 else 0
    cap_days = plan_days + 2
    print(f"\n=== LONG_SIMPLE with cap idx={TARGET_IDX} at plan_days+2 ===")
    print(f"  plan_shifts={plan_shifts}, plan_days={plan_days}, cap_days={cap_days}")

    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = {TARGET_IDX: cap_days}
    settings.SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX = TARGET_IDX

    cap_status, cap_status_str = run_long_simple_once(
        data,
        machines,
        products,
        cleans,
        remains,
        title=(
            f"LONG_SIMPLE cap idx={TARGET_IDX} plan+2 "
            f"(cap_days={cap_days})"
        ),
        html_name=f"res_idx{TARGET_IDX}_cap_plan_plus_2.html",
    )

    if cap_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(
            "Модель с капом plan+2 для idx="
            f"{TARGET_IDX} НЕ выполнима (см. лог simple_constraints_p{TARGET_IDX}.log)."
        )


if __name__ == "__main__":  # pragma: no cover
    main()
