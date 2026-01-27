from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def main() -> None:
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настройки для LS_PROP в режиме LONG_SIMPLE без APPLY_QTY_MINUS.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 600

    settings.APPLY_QTY_MINUS = False
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True

    # Остальные флаги оставляем по умолчанию (как в обычных сравнениях),
    # но явно включаем INDEX_UP и стратегии, если они нужны.
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

    # Поддерживаем оба варианта результата: dict или объект.
    if isinstance(res, dict):
        status = res.get("status", -1)
        status_str = res.get("status_str", "")
        prod_stats = res.get("products", [])
    else:
        status = res.status
        status_str = res.status_str
        prod_stats = res.products

    print(f"status={status} ({status_str})")

    # Строим индекс -> (plan, fact) по результату.
    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
        }

    print("\nidx\tname\tqty_minus\tplan\tfact\tdelta")
    for idx, prod in enumerate(data["products"]):
        # В исходном JSON d["qty_minus"] ожидается как 0/1 или False/True.
        qty_minus_flag = prod["qty_minus"]
        if qty_minus_flag != 0 and qty_minus_flag is not False:
            continue  # нас интересуют только qty_minus=false (0)

        name = prod["name"]
        s = stats.get(idx)
        if not s:
            plan = fact = delta = 0
        else:
            plan = s["plan"]
            fact = s["fact"]
            delta = fact - plan

        print(f"{idx}\t{name}\t{qty_minus_flag}\t{plan}\t{fact}\t{delta}")


if __name__ == "__main__":  # pragma: no cover
    main()
