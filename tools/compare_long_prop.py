import json
import re
from pathlib import Path

from src.config import settings, logger


def load_input_products() -> list[tuple]:
    """Загружаем входные продукты из файла TEST_INPUT_FILE (как в run.py).

    Возвращаем тот же массив tuples, что и ProductsModelToArray.
    """
    input_path = settings.TEST_INPUT_FILE or str(Path(settings.BASE_DIR) / "example" / "test_in.json")
    with open(input_path, "r", encoding="utf8") as f:
        data = json.load(f)

    # Преобразуем JSON-продукты в тот же tuple-формат, что и ProductsModelToArray.
    products_tuples = []
    first = True
    for item in data["products"]:
        if first:
            first = False
            if item["qty"] > 0:
                raise Exception("Первый элемент продукции должен быть сменой артикула, т.е. количество плана = 0")
        products_tuples.append((
            item["name"],
            item["qty"],
            item["id"],
            item.get("machine_type", 0),
            item.get("qty_minus", 0),
            item.get("lday", 0),
            item.get("src_root", 0),
            item.get("qty_minus_min", 0),
            item.get("sr", 0),
            item.get("strategy", ""),
        ))
    # products: (name, qty, id, machine_type, qty_minus, lday, src_root, qty_minus_min, sr, strategy)
    return products_tuples


def parse_products_from_log(log_path: Path) -> dict[int, dict]:
    """Парсим строки вида
    "Продукт 21(21): 92 единиц, машины 2-1, штраф пропорций 241, штраф стратегии 730".

    Возвращаем словарь: product_old_idx -> {qty, penalty_prop, penalty_strategy}.
    """
    pattern = re.compile(
        r"Продукт\s+(?P<old_idx>\d+)\((?P<idx>\d+)\):\s+"  # Продукт 21(21):
        r"план=(?P<plan_qty>\d+),\s+факт=(?P<qty>\d+)\s+единиц,\s+"
        r"машины\s+(?P<mstart>-?\d+)-(?P<mend>-?\d+),\s+"
        r"штраф пропорций\s+(?P<penalty_prop>-?\d+),\s+штраф стратегии\s+(?P<penalty_strat>-?\d+)"
    )

    result: dict[int, dict] = {}
    # Лог пишется logging'ом, он может быть в локальной кодировке (cp1251).
    # Используем cp1251 с игнорированием битых символов.
    with log_path.open("r", encoding="cp1251", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            old_idx = int(m.group("old_idx"))
            qty = int(m.group("qty"))
            penalty_prop = int(m.group("penalty_prop"))
            penalty_strat = int(m.group("penalty_strat"))
            result[old_idx] = {
                "qty": qty,
                "penalty_prop": penalty_prop,
                "penalty_strategy": penalty_strat,
            }
    return result


def main() -> None:
    base_dir = Path(settings.BASE_DIR)
    log_path = base_dir / "log" / "aps-loom_full_prop_only.log"
    if not log_path.exists():
        print(f"Лог {log_path} не найден. Сначала запустите tools/run_loom_tests_full_prop_only.ps1")
        return

    products = load_input_products()
    log_stats = parse_products_from_log(log_path)

    print("id\tidx\tname\tplan_qty\tqty_minus\tqty_minus_min\tfact_qty\tpen_prop\tpen_strat")

    total = 0
    zero_penalty = 0
    printed_any = False

    for old_idx, prod in enumerate(products):
        name, plan_qty, pid, machine_type, qty_minus, lday, src_root, qty_minus_min, sr, strategy = prod
        stats = log_stats.get(old_idx, {"qty": 0, "penalty_prop": 0, "penalty_strategy": 0})
        fact_qty = stats["qty"]
        pen_prop = stats["penalty_prop"]
        pen_strat = stats["penalty_strategy"]

        total += 1
        if pen_prop == 0 and pen_strat == 0:
            zero_penalty += 1
            # Не выводим строки без штрафа
            continue

        printed_any = True
        print(
            f"{pid}\t{old_idx}\t{name}\t{plan_qty}\t{qty_minus}\t{qty_minus_min}\t"
            f"{fact_qty}\t{pen_prop}\t{pen_strat}"
        )

    if not printed_any:
        print("Все продукты без штрафов (pen_prop=0 и pen_strat=0)")

    if total > 0:
        share_ok = zero_penalty / total * 100.0
        print(f"\nИтого: без штрафа {zero_penalty} из {total} продуктов ({share_ok:.1f}%)")

    # Дополнительно проверим жёсткие нижние границы для qty_minus=1:
    # fact_qty >= plan_qty - max(qty_minus_min, 10)
    print("\nНарушения жёстких нижних границ для qty_minus=1 (fact_qty < plan_qty - max(qty_minus_min, 10)):")
    any_violation = False
    for old_idx, prod in enumerate(products):
        name, plan_qty, pid, machine_type, qty_minus, lday, src_root, qty_minus_min, sr, strategy = prod
        if not qty_minus:
            continue
        stats = log_stats.get(old_idx, {"qty": 0, "penalty_prop": 0, "penalty_strategy": 0})
        fact_qty = stats["qty"]
        tol = max(int(qty_minus_min), 10)
        lower_hard = max(0, int(plan_qty) - tol)
        if fact_qty < lower_hard:
            any_violation = True
            print(
                f"{pid}\t{old_idx}\t{name}\tplan={plan_qty}\tqty_minus_min={qty_minus_min}\t"
                f"fact={fact_qty}\tlower_hard={lower_hard}"
            )
    if not any_violation:
        print("Нет нарушений жёстких нижних границ.")


if __name__ == "__main__":
    main()
