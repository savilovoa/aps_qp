import json
from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc

def load_input(path: Path):
    with path.open("r", encoding="utf8") as f:
        data = json.load(f)
    machines = [
        (d["name"], d["product_idx"], d["id"], d["type"], d["remain_day"], d["reserve"])
        for d in data["machines"]
    ]
    products = [
        (
            d["name"],
            d["qty"],
            d["id"],
            d["machine_type"],
            d["qty_minus"],
            d["lday"],
            d["src_root"],
            d["qty_minus_min"],
            d["sr"],
            d["strategy"],
            d["div"]
        )
        for d in data["products"]
    ]
    cleans = [(d["machine_idx"], d["day_idx"]) for d in data["cleans"]]
    remains = data["remains"]
    return data, machines, products, cleans, remains


def collect_stats(mode: str, data, machines, products, cleans, remains):
    """Запуск расчёта для одного режима и вычисление внешнего пропорционального штрафа.

    Возвращает:
      stats: dict[idx] -> {plan, fact, penalty, penalty_strategy}
      prop_penalty: суммарный |fact*sum(plan) - sum(fact)*plan| по всем продуктам с plan>0.
      per_prod_penalty: dict[idx] -> вклад продукта в prop_penalty.
    """
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
    prod_stats = res["products"]  # list of dicts
    # Индекс -> (plan, fact)
    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
            "penalty": ps.get("penalty", 0),
            "penalty_strategy": ps.get("penalty_strategy", 0),
        }

    # Внешний пропорциональный штраф: |fact[p]*sum(plan) - sum(fact)*plan[p]|.
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

    return stats, prop_penalty, per_prod_penalty


def main():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # В этих экспериментах qty_minus явно включаем.
    settings.APPLY_QTY_MINUS = True

    # Два сценария:
    # 1) LONG_SIMPLE + линейный штраф за превышение (LS_OVER)
    # 2) LONG_SIMPLE + пропорциональный алгоритм через умножение (LS_PROP)
    scenarios = [
        # SIMPLE без внутренней пропорциональной цели (жёсткие верхние лимиты + простой + стратегии)
        ("LS_OVER", "LONG_SIMPLE", {"APPLY_OVERPENALTY_INSTEAD_OF_PROP": False, "SIMPLE_USE_PROP_MULT": False}),
        # SIMPLE с полной пропорциональной целью через умножение (для сравнения)
        ("LS_PROP", "LONG_SIMPLE", {"APPLY_OVERPENALTY_INSTEAD_OF_PROP": False, "SIMPLE_USE_PROP_MULT": True}),
    ]

    all_stats: dict[str, dict] = {}

    # --- Первая серия: с текущим лимитом времени ---
    orig_max_time = settings.LOOM_MAX_TIME

    for label, horizon_mode, flags in scenarios:
        print(f"\n=== SCENARIO {label} (HORIZON_MODE={horizon_mode}, flags={flags}) ===")

        # Настраиваем флаги перед запуском.
        settings.HORIZON_MODE = horizon_mode
        settings.APPLY_PROP_OBJECTIVE = True
        settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = flags["APPLY_OVERPENALTY_INSTEAD_OF_PROP"]
        settings.SIMPLE_USE_PROP_MULT = flags["SIMPLE_USE_PROP_MULT"]

        stats, prop_penalty, per_prod_penalty = collect_stats(
            horizon_mode, data, machines, products, cleans, remains
        )
        all_stats[label] = {
            "stats": stats,
            "prop_penalty": prop_penalty,
            "per_prod_penalty": per_prod_penalty,
        }

        print(f"Пропорциональный штраф (внешний) для {label}: {prop_penalty}")
        print("idx\tname\tplan\tfact\tdelta")
        for idx, prod in enumerate(data["products"]):
            name = prod["name"]
            s = stats.get(idx)
            if not s:
                continue
            plan = s["plan"]
            fact = s["fact"]
            delta = fact - plan
            print(f"{idx}\t{name}\t{plan}\t{fact}\t{delta}")

        # Для LS-сценариев дополнительно выводим вклад пропорциональных штрафов по продуктам.
        if label in ("LS_OVER", "LS_PROP"):
            print(f"\n=== Внешний пропорциональный штраф по продуктам для {label} ===")
            print("idx\tname\tplan\tfact\tprop_penalty")
            for idx, prod in enumerate(data["products"]):
                if idx == 0:
                    continue
                name = prod["name"]
                s = stats.get(idx)
                if not s:
                    continue
                plan = s["plan"]
                fact = s["fact"]
                pen = per_prod_penalty.get(idx, 0)
                print(f"{idx}\t{name}\t{plan}\t{fact}\t{pen}")

    # --- Вторая серия: только LS_OVER и LS_PROP с увеличенным лимитом времени ---
    print("\n=== Повторный запуск LS_OVER и LS_PROP при LOOM_MAX_TIME=600 ===")
    settings.LOOM_MAX_TIME = 600

    ls_scenarios = [
        ("LS_OVER_T600", "LONG_SIMPLE", {"APPLY_OVERPENALTY_INSTEAD_OF_PROP": False, "SIMPLE_USE_PROP_MULT": False}),
        ("LS_PROP_T600", "LONG_SIMPLE", {"APPLY_OVERPENALTY_INSTEAD_OF_PROP": False, "SIMPLE_USE_PROP_MULT": True}),
    ]

    for label, horizon_mode, flags in ls_scenarios:
        print(f"\n=== SCENARIO {label} (HORIZON_MODE={horizon_mode}, flags={flags}) ===")

        settings.HORIZON_MODE = horizon_mode
        settings.APPLY_PROP_OBJECTIVE = True
        settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = flags["APPLY_OVERPENALTY_INSTEAD_OF_PROP"]
        settings.SIMPLE_USE_PROP_MULT = flags["SIMPLE_USE_PROP_MULT"]

        stats, prop_penalty, per_prod_penalty = collect_stats(
            horizon_mode, data, machines, products, cleans, remains
        )

        print(f"Пропорциональный штраф (внешний) для {label}: {prop_penalty}")
        print("idx\tname\tplan\tfact\tprop_penalty")
        for idx, prod in enumerate(data["products"]):
            if idx == 0:
                continue
            name = prod["name"]
            s = stats.get(idx)
            if not s:
                continue
            plan = s["plan"]
            fact = s["fact"]
            pen = per_prod_penalty.get(idx, 0)
            print(f"{idx}\t{name}\t{plan}\t{fact}\t{pen}")

    # Восстанавливаем исходный лимит времени
    settings.LOOM_MAX_TIME = orig_max_time


if __name__ == "__main__":
    main()
