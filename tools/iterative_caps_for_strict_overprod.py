from __future__ import annotations

from pathlib import Path
from math import ceil
import sys

from ortools.sat.python import cp_model

from src.config import settings
from src.loom.schedule_loom import schedule_loom_calc
from tools.compare_long_vs_simple import load_input

# Гарантируем, что stdout в скрипте работает в UTF-8, чтобы логи,
# перенаправленные в файл (aps-loom_iterative_caps.log), были в UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_data():
    base = Path(settings.BASE_DIR)
    input_path = settings.TEST_INPUT_FILE or base / "example" / "test_in.json"
    data, machines, products, cleans, remains = load_input(Path(input_path))
    return data, machines, products, cleans, remains


def run_long_simple(data, machines, products, cleans, remains):
    """Запуск LONG_SIMPLE с текущими настройками settings и возврат результата."""
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
        products_stats = res.get("products", [])
        objective_value = res.get("objective_value", 0)
        proportion_diff = res.get("proportion_diff", 0)
    else:
        status = int(res.status)
        status_str = str(res.status_str)
        products_stats = res.products
        objective_value = getattr(res, "objective_value", 0)
        proportion_diff = getattr(res, "proportion_diff", 0)

    return status, status_str, products_stats, objective_value, proportion_diff

# Необязательный параметр для удобства логирования/фокуса:
# если задан, просто печатаем позицию этого продукта в over_list, но
# сами капы всегда накладываем последовательно с начала списка (с макс. over).
# Чтобы пройти весь список с самого верха (idx=20 и т.д.), оставьте None.
START_FROM_PRODUCT_IDX: int | None = None


def main() -> None:
    data, machines, products, cleans, remains = load_data()

    # Базовые настройки LONG_SIMPLE как в бою.
    settings.HORIZON_MODE = "LONG_SIMPLE"
    settings.LOOM_MAX_TIME = 300
    settings.APPLY_QTY_MINUS = True
    # Используем тот же поднабор qty_minus, что и в успешном эксперименте.
    settings.SIMPLE_QTY_MINUS_SUBSET = {5, 4, 6, 7, 8, 10, 9, 13, 11}

    # Снимаем все отладочные капы перед первым запуском.
    settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = None

    print("=== Базовый запуск LONG_SIMPLE без доп.кап для strict продуктов ===")
    base_status, base_status_str, base_stats, base_obj, base_prop = run_long_simple(
        data, machines, products, cleans, remains
    )
    print(f"status={base_status} ({base_status_str}), obj={base_obj}, prop={base_prop}")

    if base_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Базовая модель невыполнима, дальнейшая диагностика caps бессмысленна.")
        return

    # Собираем статистику по продуктам: ищем превышения плана у strict (qty_minus=false).
    over_list: list[tuple[int, int, int, int]] = []  # (idx, plan_shifts, fact_shifts, over)

    for ps in base_stats:
        p_idx = int(ps.get("product_idx", -1))
        if p_idx <= 0:
            continue
        plan = int(ps.get("plan_qty", 0))
        fact = int(ps.get("qty", 0))
        if p_idx >= len(products):
            continue
        prod = products[p_idx]
        # products: [ (name, qty, id, machine_type, qty_minus, ...) ]
        try:
            qty_minus_flag = int(prod[4])
        except Exception:
            qty_minus_flag = 0 if not prod[4] else 1

        # Интересуют только строгие продукты (qty_minus=false/0).
        if qty_minus_flag != 0:
            continue

        if fact > plan:
            over = fact - plan
            over_list.append((p_idx, plan, fact, over))

    if not over_list:
        print("Нет strict-продуктов с превышением плана, caps не на что ставить.")
        return

    # Сортируем по величине превышения (по убыванию).
    over_list.sort(key=lambda t: t[3], reverse=True)

    print("\nСписок strict-продуктов с превышением (plan, fact, fact-plan):")
    for p_idx, plan, fact, over in over_list:
        print(f"  idx={p_idx}: plan={plan}, fact={fact}, over={over}")

    # Если задан START_FROM_PRODUCT_IDX, просто сообщаем его позицию для ориентировки,
    # но сами капы всегда накладываем последовательно с начала списка.
    if START_FROM_PRODUCT_IDX is not None:
        focus_pos = None
        for i, (p_idx, plan, fact, over) in enumerate(over_list):
            if p_idx == START_FROM_PRODUCT_IDX:
                focus_pos = i
                break
        if focus_pos is not None:
            print(
                f"\nФокусный продукт idx={START_FROM_PRODUCT_IDX} находится "
                f"на позиции {focus_pos} в over_list"
            )

    # Итеративно добавляем верхний кап plan+2 (в днях) для каждого strict-продукта
    # начиная с максимального перепроизводства. Капы накапливаются: на шаге k
    # в модели активны капы для всех продуктов 0..k. Как только модель становится
    # невыполнимой, останавливаем цикл.
    caps: dict[int, int] = {}

    for p_idx, plan, fact, over in over_list:
        # Пересчёт плана в дни (3 смены в день).
        plan_days = ceil(plan / 3) if plan > 0 else 0
        cap_days = plan_days + 2

        print(
            "\n=== Пробуем кап для strict idx=",
            p_idx,
            "cap_days=",
            cap_days,
            "(plan_days=",
            plan_days,
            ") ===",
        )

        # Добавляем/обновляем кап для очередного продукта и применяем все накопленные капы.
        caps[p_idx] = cap_days
        settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = dict(caps)

        status, status_str, stats, obj, prop = run_long_simple(
            data, machines, products, cleans, remains
        )
        print(f"status={status} ({status_str}), obj={obj}, prop={prop}")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("  -> Модель НЕ выполнима при таком капе для idx=", p_idx)
            print("  -> Цикл остановлен на первом невыполнимом капе.")
            break

        pf = {
            int(ps.get("product_idx", -1)): (
                int(ps.get("plan_qty", 0)),
                int(ps.get("qty", 0)),
            )
            for ps in stats
        }
        new_plan, new_fact = pf.get(p_idx, (0, 0))
        print(f"  -> idx={p_idx}: plan={new_plan}, fact={new_fact} смен после кapa")


if __name__ == "__main__":  # pragma: no cover
    main()
