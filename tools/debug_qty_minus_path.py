from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input


def run_ls_prop(use_qty_minus: bool, subset: set[int] | None) -> tuple[int, str]:
    """Запускает LS_PROP (LONG_SIMPLE) с/без APPLY_QTY_MINUS и заданным подмножеством
    продуктов, для которых применяются нижние границы qty_minus.

    Возвращает (status, status_str).
    """
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # Настройки для LS_PROP в режиме LONG_SIMPLE.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 120  # ограничим время, чтобы успеть сделать много прогонов

    settings.APPLY_QTY_MINUS = use_qty_minus
    # Передаём подмножество продуктов в SIMPLE (см. SIMPLE_QTY_MINUS_SUBSET).
    if subset is None:
        settings.SIMPLE_QTY_MINUS_SUBSET = None
    else:
        settings.SIMPLE_QTY_MINUS_SUBSET = set(subset)

    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True  # LS_PROP
    settings.APPLY_INDEX_UP = True

    # Эвристики начального плана используем в обычном (продакшн) режиме.
    settings.SIMPLE_DEBUG_H_START = False
    settings.SIMPLE_DEBUG_H_MODE = None

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
        status = res.get("status", -1)
        status_str = res.get("status_str", "")
    else:
        status = res.status
        status_str = res.status_str

    return status, status_str


def compute_hard_products_stats() -> tuple[dict[int, dict], list[dict]]:
    """Запускает LS_PROP без APPLY_QTY_MINUS и возвращает:
      - stats: idx -> {plan, fact}
      - products: исходный список продуктов из входного JSON.
    """
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))

    # LS_PROP без qty_minus (как в предыдущем эксперименте).
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 120
    settings.APPLY_QTY_MINUS = False
    settings.APPLY_PROP_OBJECTIVE = True
    settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP = False
    settings.SIMPLE_USE_PROP_MULT = True
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
        prod_stats = res.get("products", [])
    else:
        prod_stats = res.products

    stats: dict[int, dict] = {}
    for ps in prod_stats:
        idx = ps["product_idx"]
        stats[idx] = {
            "plan": ps["plan_qty"],
            "fact": ps["qty"],
        }

    return stats, data["products"]


def main() -> None:
    stats, products = compute_hard_products_stats()

    hard_idxs: list[int] = []
    passed_idxs: list[int] = []
    deficit_entries: list[tuple[int, int, int, int]] = []

    for idx, prod in enumerate(products):
        if idx == 0:
            continue  # 0-й индекс обычно служебный
        qty_minus_flag = prod["qty_minus"]
        if qty_minus_flag != 0 and qty_minus_flag is not False:
            continue  # интересуют только жёсткие продукты (qty_minus = 0)

        s = stats.get(idx, {"plan": 0, "fact": 0})
        plan = int(s["plan"])
        fact = int(s["fact"])
        delta = fact - plan

        hard_idxs.append(idx)
        if fact >= plan:
            passed_idxs.append(idx)
        else:
            deficit_entries.append((idx, plan, fact, delta))

    # Сортируем дефицитные продукты по величине недобора (plan - fact) по убыванию.
    deficit_entries.sort(key=lambda t: (t[1] - t[2]), reverse=True)

    print("Всего жёстких продуктов (qty_minus=0):", len(hard_idxs))
    print("Прошли по текущему плану (fact>=plan):", len(passed_idxs))
    print("С дефицитом (fact<plan):", len(deficit_entries))

    print("\nЖёсткие продукты, прошедшие план (idx, name, plan, fact):")
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    # products уже из JSON, не нужно перечитывать

    for idx in passed_idxs:
        prod = products[idx]
        name = prod["name"]
        s = stats.get(idx, {"plan": 0, "fact": 0})
        print(f"  {idx}\t{name}\tplan={s['plan']}\tfact={s['fact']}")

    print("\nТоп дефицитных жёстких продуктов (idx, name, plan, fact, deficit):")
    for idx, plan, fact, delta in deficit_entries[:20]:
        name = products[idx]["name"]
        deficit = plan - fact
        print(f"  {idx}\t{name}\tplan={plan}\tfact={fact}\tdeficit={deficit}")

    # 1) Базовый запуск: только продукты, которые уже прошли по плану.
    subset: set[int] = set(passed_idxs)
    print("\n=== Шаг 1: включаем qty_minus только для прошедших по плану продуктов ===")
    status, status_str = run_ls_prop(use_qty_minus=True, subset=subset)
    print(f"status={status} ({status_str}), subset_size={len(subset)})")

    if status_str and "INFEASIBLE" in status_str.upper():
        print("Модель стала невыполнимой уже при ограничении только прошедших продуктов.")
        return

    # 2) Добавляем дефицитные продукты по одному, начиная с наибольшего недобора.
    print("\n=== Шаг 2: добавляем дефицитные продукты по одному ===")
    for idx, plan, fact, delta in deficit_entries:
        subset.add(idx)
        name = products[idx]["name"]
        deficit = plan - fact
        print(f"\n-- Пробуем добавить продукт idx={idx}, name={name}, plan={plan}, fact={fact}, deficit={deficit}")
        status, status_str = run_ls_prop(use_qty_minus=True, subset=subset)
        print(f"status={status} ({status_str}), subset_size={len(subset)})")

        if status_str and "INFEASIBLE" in status_str.upper():
            print("\n*** ПЕРВАЯ НЕВЫПОЛНИМОСТЬ при добавлении продукта: ***")
            print(f"idx={idx}, name={name}, plan={plan}, fact={fact}, deficit={deficit}")
            break


if __name__ == "__main__":  # pragma: no cover
    main()
