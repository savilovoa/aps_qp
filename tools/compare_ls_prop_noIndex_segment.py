from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def collect_stats(mode: str, data, machines, products, cleans, remains):
    settings.HORIZON_MODE = mode
    res = schedule_loom_calc(
        remains=remains,
        products=products,
        machines=machines,
        cleans=cleans,
        max_daily_prod_zero=data["max_daily_prod_zero"],
        count_days=data["count_days"],
        data=data,
    )

    # schedule_loom_calc в этом скрипте всегда возвращает dict
    prod_stats = res["products"]
    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
            "penalty": ps.get("penalty", 0),
            "penalty_strategy": ps.get("penalty_strategy", 0),
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

    return stats, prop_penalty, per_prod_penalty, res


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настройки для LS_PROP с новым правилом:
    # - LONG_SIMPLE
    # - APPLY_QTY_MINUS=True
    # - пропорциональная цель через умножение (как в LS_PROP)
    # - INDEX_UP выключен только в SIMPLE, но включён глобально
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 600

    settings.APPLY_QTY_MINUS = True

    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True  # LS_PROP
    settings.APPLY_STRATEGY_PENALTY = True

    settings.APPLY_INDEX_UP = True
    settings.SIMPLE_DISABLE_INDEX_UP = True

    # Эвристики начального плана: H1+H2+H3
    settings.SIMPLE_DEBUG_H_START = True
    settings.SIMPLE_DEBUG_H_MODE = "H123"

    stats_ls, prop_penalty_ls, per_prod_penalty_ls, res_ls = collect_stats(
        "LONG_SIMPLE", data, machines, products, cleans, remains
    )

    print("status=", res_ls["status"], res_ls["status_str"])
    print(f"Пропорциональный штраф (внешний) для LS_PROP_noIndex: {prop_penalty_ls}")
    print("idx\tname\tplan\tfact\tpenalty_ext")
    for idx, prod in enumerate(data["products"]):
        if idx == 0:
            continue
        name = prod["name"]
        s = stats_ls.get(idx)
        if not s:
            continue
        plan = s["plan"]
        fact = s["fact"]
        pen_ext = per_prod_penalty_ls.get(idx, 0)
        print(f"{idx}\t{name}\t{plan}\t{fact}\t{pen_ext}")


if __name__ == "__main__":  # pragma: no cover
    main()
