from __future__ import annotations

from pathlib import Path
from datetime import datetime

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from src.loom.loom_plan_html import aggregated_schedule_to_html
from tools.compare_long_vs_simple import load_input

STRICT9 = [5, 4, 6, 7, 8, 10, 9, 13, 11]
STRICT10 = STRICT9 + [22]


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def run_long_simple_with_subset(data, machines, products, cleans, remains, subset: list[int]):
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = True
    # ВАЖНО: даём subset в внешних idx, маппинг на внутренние idx сделает schedule_loom_calc.
    settings.SIMPLE_QTY_MINUS_SUBSET = set(subset)

    # Внутренняя пропорциональная цель LS_OVER включена.
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

    # 1) План с 9 строгими: {5,4,6,7,8,10,9,13,11}
    print("=== LONG_SIMPLE с STRICT9 (qty_minus=0) ===")
    status9, status_str9, sched9, stats9, obj9, prop9 = run_long_simple_with_subset(
        data, machines, products, cleans, remains, STRICT9
    )
    print(f"status={status9} ({status_str9}), obj={obj9}, prop={prop9}")
    pf9 = extract_plan_fact(stats9, STRICT9 + [22])
    for idx in STRICT9:
        plan, fact = pf9.get(idx, (0, 0))
        print(f"  idx={idx}: plan_days≈{plan}, fact_days={fact}")
    if 22 in pf9:
        p22_plan, p22_fact = pf9[22]
        print(f"  idx=22: plan_days≈{p22_plan}, fact_days={p22_fact}")

    if status_str9.upper() not in ("INFEASIBLE",):
        out_html9 = Path("example/res_strict9.html")
        title9 = f"STRICT9 LONG_SIMPLE status={status_str9}, obj={obj9}, prop={prop9}"
        make_html_for_schedule(out_html9, data, sched9, title9)
        print(f"HTML для STRICT9 сохранён в {out_html9}")

    # 2) Диагностика с добавлением idx=22: просто запустим, чтобы удостовериться в INFEASIBLE.
    print("\n=== LONG_SIMPLE с STRICT10 (STRICT9 + 22) ===")
    status10, status_str10, sched10, stats10, obj10, prop10 = run_long_simple_with_subset(
        data, machines, products, cleans, remains, STRICT10
    )
    print(f"status={status10} ({status_str10}), obj={obj10}, prop={prop10}")
    pf10 = extract_plan_fact(stats10, STRICT10)
    for idx in STRICT10:
        plan, fact = pf10.get(idx, (0, 0))
        print(f"  idx={idx}: plan_days≈{plan}, fact_days={fact}")


if __name__ == "__main__":  # pragma: no cover
    main()
