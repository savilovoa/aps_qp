from __future__ import annotations

from pathlib import Path
from datetime import datetime

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from src.loom.loom_plan_html import aggregated_schedule_to_html, schedule_to_html
from tools.compare_long_vs_simple import load_input


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def get_strict_sorted_idxs(data: dict) -> list[int]:
    strict: list[tuple[int, int]] = []  # (idx, qty)
    for p in data["products"]:
        idx = int(p["idx"])
        if idx == 0:
            continue
        qty = int(p.get("qty", 0) or 0)
        if qty <= 0:
            continue
        qm = p.get("qty_minus", 0)
        if qm != 0 and qm is not False:
            continue
        strict.append((idx, qty))
    strict.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in strict]


def main() -> None:
    data, machines, products, cleans, remains = load_data()

    strict_idxs = get_strict_sorted_idxs(data)
    first16 = strict_idxs[:16]
    print("Первые 16 строгих продуктов (idx):", first16)

    # Настройки LONG_SIMPLE + qty_minus только для первых 16 строгих.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = True
    settings.SIMPLE_QTY_MINUS_SUBSET = set(first16)
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
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

    print(f"status={status} ({status_str}), objective={objective_value}, proportion_diff={proportion_diff}")

    if status_str.upper() in ("INFEASIBLE",):
        print("Модель невыполнима для subset из 16 строгих продуктов; HTML не формируем.")
        return

    # Агрегированный long_schedule по дням и продуктам (как в schedule_loom_calc_model / main.calc_test_data_from)
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

    # Дата начала берётся из JSON (как в main.calc_test_data_from)
    dt_begin = datetime.strptime(data["dt_begin"], "%Y-%m-%dT%H:%M:%S").date()
    title_text = f"LONG_SIMPLE strict16, status={status_str}, obj={objective_value}, prop={proportion_diff}"

    html = aggregated_schedule_to_html(
        machines=data["machines"],
        schedule=schedule,
        products=data["products"],
        long_schedule=long_schedule,
        dt_begin=dt_begin,
        title_text=title_text,
    )

    out_path = Path("example/res_strict16.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML сохранён в {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
