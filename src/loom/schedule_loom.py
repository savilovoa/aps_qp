from ortools.sat.python import cp_model
from pydantic import BaseModel
from .model_loom import DataLoomIn, LoomPlansOut, Machine, Product, Clean, LoomPlan, LoomPlansViewIn, LoomPlansViewOut
import traceback as tr
from ..config import logger, settings
import pandas as pd
from .loom_plan_html import schedule_to_html
from uuid import uuid4
import time
def MachinesModelToArray(machines: list[Machine]) -> list[(str, int, str, int, int)]:
    result = []
    idx = 0
    for item in machines:
        if item.idx != idx:
            break
        result.append((item.name, item.product_idx, item.id, item.type, item.remain_day))
        idx += 1
    return result

def ProductsModelToArray(products: list[Product]) -> list[tuple]:
    """Преобразование моделей продуктов в кортежи для расчёта/отображения.

    Структура кортежа:
    (name, qty, id, machine_type, qty_minus, lday, src_root, qty_minus_min, sr, strategy)
    """
    result = []
    first = True
    for item in products:
        if first:
            first = False
            if item.qty > 0:
                raise "Первый элемент продукции должен быть сменой артикула, т.е. количество плана = 0"
        result.append((
            item.name,
            item.qty,
            item.id,
            item.machine_type,
            item.qty_minus,
            item.lday,
            item.src_root,
            item.qty_minus_min,
            item.sr,
            item.strategy,
        ))
    return result

def CleansModelToArray(cleans: list[Clean]) -> list[(int, int)]:
    result = []
    for item in cleans:
        result.append((item.machine_idx, item.day_idx))
    return result

def schedule_loom_calc_model(DataIn: DataLoomIn) -> LoomPlansOut:
    try:
        save_model_to_log(DataIn)
        remains = DataIn.remains
        products = ProductsModelToArray(DataIn.products)
        machines = MachinesModelToArray(DataIn.machines)
        cleans = CleansModelToArray(DataIn.cleans)

        max_daily_prod_zero = DataIn.max_daily_prod_zero
        count_days = DataIn.count_days
        days = [i for i in range(count_days)]
        data = DataIn.model_dump()
        if DataIn.apply_index_up:
            settings.APPLY_INDEX_UP = DataIn.apply_index_up
        if DataIn.apply_qty_minus:
            settings.APPLY_QTY_MINUS = DataIn.apply_qty_minus

        result_calc = schedule_loom_calc(remains=remains, products=products, machines=machines, cleans=cleans,
                                    max_daily_prod_zero=max_daily_prod_zero, count_days=count_days, data=data)

        if result_calc["error_str"] == "" and result_calc["status"] != cp_model.INFEASIBLE:
            machines_view = [name for (name, product_idx,  id, type, remain_day) in machines]
            products_view = [name for (name, qty, id, machine_type, qm) in products]
            title_text = f"{result_calc['status_str']} оптимизационное значение {result_calc['objective_value']}"

            res_html = schedule_to_html(machines=machines_view, products=products_view, schedules=result_calc["schedule"],
                                        days=days, dt_begin=DataIn.dt_begin, title_text=title_text)
            id_html = str(uuid4())


            schedule = [LoomPlan(machine_idx=s["machine_idx"], day_idx=s["day_idx"], product_idx=s["product_idx"],
                                 days_in_batch=s["days_in_batch"], prev_lday=s["prev_lday"])
                        for s in result_calc["schedule"]]


            result = LoomPlansOut(status=result_calc["status"], status_str=result_calc["status_str"],
                                  schedule=schedule,objective_value=result_calc["objective_value"],
                                  proportion_diff=result_calc["proportion_diff"], res_html=id_html)
            save_plan_html(id_html, res_html)
            save_model_to_log(result)
        else:
            error_str = result_calc["error_str"]
            if result_calc["status"] == cp_model.INFEASIBLE:
                error_str = error_str + " МОДЕЛЬ НЕ МОЖЕТ БЫТЬ РАССЧИТАНА"
            result = LoomPlansOut(error_str=error_str)
    except Exception as e:
        error = tr.TracebackException(exc_type=type(e), exc_traceback=e.__traceback__, exc_value=e).stack[-1]
        error_str = '{} in file {} in {} row:{} '.format(e, error.filename, error.lineno, error.line)
        logger.error(error_str)
        result = LoomPlansOut(error_str=error_str)
    return result

def loom_plans_view(plan_in: LoomPlansViewIn) -> LoomPlansViewOut:
    try:

        products = ProductsModelToArray(plan_in.products)
        machines = MachinesModelToArray(plan_in.machines)
        count_days = plan_in.count_days
        days = [i for i in range(count_days)]
        schedule =plan_in.schedule.__dict__()

        res_html = schedule_to_html(machines=machines, products=products, schedules=schedule, days=days,
                                    dt_begin=plan_in.dt_begin)
        result = LoomPlansViewOut(res_html=res_html)
    except Exception as e:
        error = tr.TracebackException(exc_type=type(e), exc_traceback=e.__traceback__, exc_value=e).stack[-1]
        error_str = '{} in file {} in {} row:{} '.format(e, error.filename, error.lineno, error.line)
        logger.error(error_str)
        result = LoomPlansViewOut(res_html="", error_str=error_str)
    return result


def create_schedule_init(machines: list[dict], products: list[dict], cleans: list[dict],
                         count_days: int, max_daily_prod_zero: int):
    """Жадный предварительный план (из example/test_df2.py), адаптированный под JSON-формат.

    machines, products, cleans — это списки словарей из входного JSON:
    data["machines"], data["products"], data["cleans"].
    """
    machines_df = pd.DataFrame(machines)
    products_df = pd.DataFrame(products)
    num_machines = len(machines_df)
    num_products = len(products_df)

    # Тип продукта по idx (0 или 1): 1 -> только на машинах с type=1; 0 -> на любых.
    product_type_map: dict[int, int] = {}
    if "idx" in products_df.columns and "machine_type" in products_df.columns:
        product_type_map = (
            products_df.set_index("idx")["machine_type"].astype(int).to_dict()
        )

    def can_run_product_on_machine(p_idx: int, m_idx: int) -> bool:
        """Проверка совместимости типа продукта и типа машины.

        - продукты с типом 1 -> только на машинах с type=1;
        - продукты с типом 0 -> на любых машинах;
        - PRODUCT_ZERO (0) и None не проверяем.
        """
        if p_idx == 0:
            return True
        p_type = int(product_type_map.get(p_idx, 0))
        if p_type == 1:
            try:
                m_type = int(machines_df.at[m_idx, "type"])
            except Exception:
                m_type = 0
            return m_type == 1
        return True

    schedule = [[None for _ in range(count_days)] for _ in range(num_machines)]

    # Предварительно заполняем дни очистки (cleans) - это жесткие ограничения
    for clean in cleans:
        machine_idx = clean["machine_idx"]
        day_idx = clean["day_idx"]
        if 0 <= machine_idx < num_machines and 0 <= day_idx < count_days:
            schedule[machine_idx][day_idx] = -2

    # Подсчет рабочих дней без чисток
    work_days = count_days * len(machines) - sum(1 for _ in cleans)

    # Функция для проверки возможности установки перехода (prod_zero) в день
    def can_place_zero(day: int, zeros_per_day: dict[int, int], max_daily_prod_zero: int) -> bool:
        return zeros_per_day.get(day, 0) < max_daily_prod_zero

    # Счетчик переходов по дням
    zeros_per_day: dict[int, int] = {day: 0 for day in range(count_days)}

    # --- Шаг 3: Добавление колонки с индексом объекта ---
    machines_df.reset_index(inplace=True)
    machines_df.rename(columns={"index": "original_index"}, inplace=True)

    # --- Шаг 4: Добавление колонки 'product_qty' ---
    product_quantities = products_df["qty"]
    machines_df["product_qty"] = machines_df["product_idx"].map(product_quantities)

    # Правим количество пропорционально
    # Считаем мин и мак чисток
    next_count_min = 0
    for _, machine in machines_df.iterrows():
        if machine["product_qty"] < count_days / 2:
            next_count_min += 1
    next_count_max = round(count_days / 2)
    if next_count_max < next_count_min:
        next_count = next_count_max
    else:
        next_count = next_count_min * round((next_count_max - next_count_min) / 2) if (next_count_max - next_count_min) > 0 else 0
    work_days = work_days - next_count * 2

    count_qty = 0
    for qty in products_df[products_df["idx"] > 0]["qty"]:
        count_qty += qty

    # Считаем коэф увеличения/уменьшения
    if count_qty > 0 and work_days > 0:
        kf_count = 0.9 * work_days / count_qty
    else:
        kf_count = 1.0

    for product_idx in range(1, num_products):
        base_qty = int(products_df.at[product_idx, "qty"])
        if base_qty <= 0:
            continue
        qty_minus_val = int(products_df.at[product_idx, "qty_minus"]) if "qty_minus" in products_df.columns else 0
        if qty_minus_val == 0:
            # Строгие продукты (qty_minus=0) не масштабируем, чтобы ближе держаться к плану.
            qty = base_qty
        else:
            qty = round(base_qty * kf_count)
        products_df.at[product_idx, "qty"] = qty

    product_quantities = products_df["qty"]
    machines_df["product_qty"] = machines_df["product_idx"].map(product_quantities)

    # --- Шаг 5: Добавление колонки 'day_remains' ---
    machines_df["day_remains"] = count_days
    for machine_idx in range(num_machines):
        clean_days = sum(1 for clean in cleans if clean["machine_idx"] == machine_idx)
        machines_df.at[machine_idx, "day_remains"] -= clean_days

    # --- Учёт remain_day: дорабатываем начальные партии без переходов ---
    if "remain_day" in machines_df.columns:
        for machine_idx in range(num_machines):
            initial_product_idx = int(machines_df.at[machine_idx, "product_idx"])
            remain_req = int(machines_df.at[machine_idx, "remain_day"])
            # Если стартовый продукт = 0, остатка партии нет (как в основной модели)
            if initial_product_idx == 0 or remain_req <= 0:
                continue
            day_idx = 0
            days_filled = 0
            while day_idx < count_days and days_filled < remain_req:
                if schedule[machine_idx][day_idx] is None:
                    schedule[machine_idx][day_idx] = initial_product_idx
                    machines_df.at[machine_idx, "day_remains"] -= 1
                    if initial_product_idx in products_df.index:
                        products_df.at[initial_product_idx, "qty"] -= 1
                    machines_df.at[machine_idx, "product_qty"] -= 1
                    days_filled += 1
                day_idx += 1

    # --- НОВЫЙ ЖАДНЫЙ АЛГОРИТМ ---
    # Фаза 1. Продукты с начальными машинами без переходов.

    # Словарь: продукт -> список машин, у которых этот продукт начальный
    product_to_initial_machines: dict[int, list[int]] = {}
    for m_idx in range(num_machines):
        p0 = int(machines_df.at[m_idx, "product_idx"])
        product_to_initial_machines.setdefault(p0, []).append(m_idx)

    # Количество оставшихся смен по каждому продукту (после учёта remain_day и масштабирования qty)
    product_qty: dict[int, int] = {}
    for _, row in products_df.iterrows():
        p_idx = int(row["idx"])
        if p_idx == 0:
            continue
        product_qty[p_idx] = int(row["qty"])

    # Разделяем продукты на строгие (qty_minus=0) и гибкие (остальные)
    strict_products: set[int] = set()
    flex_products: set[int] = set()
    if "qty_minus" in products_df.columns:
        for _, row in products_df.iterrows():
            p_idx = int(row["idx"])
            if p_idx == 0:
                continue
            q_minus = int(row.get("qty_minus", 0))
            if q_minus == 0:
                strict_products.add(p_idx)
            else:
                flex_products.add(p_idx)
    else:
        flex_products = set(product_qty.keys())

    # Продукты с qty>0 и начальными машинами
    phase1_all = [
        p for p, q in product_qty.items()
        if q > 0 and p in product_to_initial_machines
    ]
    # Сначала распределяем строго ограниченные продукты (qty_minus=0), затем остальные
    phase1_strict = [p for p in phase1_all if p in strict_products]
    phase1_flex = [p for p in phase1_all if p not in strict_products]
    phase1_strict.sort(key=lambda p: product_qty[p], reverse=True)
    phase1_flex.sort(key=lambda p: product_qty[p], reverse=True)
    phase1_products = phase1_strict + phase1_flex

    def free_days_for_machine(m_idx: int) -> list[int]:
        return [d for d in range(count_days) if schedule[m_idx][d] is None]

    def min_remaining_product_qty() -> int | None:
        vals = [q for q in product_qty.values() if q > 0]
        return min(vals) if vals else None

    # Фаза 1: заполняем начальные машины своими продуктами
    for p in phase1_products:
        machines_for_p = product_to_initial_machines.get(p, [])
        for m_idx in machines_for_p:
            if product_qty[p] <= 0:
                break
            # Проверяем совместимость типа продукта и машины.
            if not can_run_product_on_machine(p, m_idx):
                continue
            free_days = free_days_for_machine(m_idx)
            if not free_days:
                continue

            # Сначала заполняем машину этим продуктом по порядку свободных смен
            for d in free_days:
                if product_qty[p] <= 0:
                    break
                schedule[m_idx][d] = p
                product_qty[p] -= 1

            # После первичного заполнения пересчитываем оставшиеся свободные дни
            free_days = free_days_for_machine(m_idx)
            if not free_days:
                continue

            # Если оставшихся свободных смен на машине меньше, чем
            # минимальный остаток по ещё не распределённым продуктам,
            # добиваем эту машину текущим продуктом (насколько хватит qty).
            min_rem = min_remaining_product_qty()
            if min_rem is not None and len(free_days) < min_rem and product_qty[p] > 0:
                for d in free_days:
                    if product_qty[p] <= 0:
                        break
                    schedule[m_idx][d] = p
                    product_qty[p] -= 1

    # Отладочная иерархическая таблица: продукт -> машины с начальным продуктом и числом смен
    try:
        logger.info("Иерархическая сводка после Фазы 1 (продукт -> (машина, нач_прод, смен продуктом)):")
        for p in phase1_products:
            logger.info(f"Продукт {p} (план после масштабирования = {product_qty.get(p, 0)} остатков)")
            machines_for_p = product_to_initial_machines.get(p, [])
            for m_idx in machines_for_p:
                init_p = int(machines_df.at[m_idx, "product_idx"])
                filled = sum(1 for d in range(count_days) if schedule[m_idx][d] == p)
                logger.info(
                    f"  Машина {m_idx} (init={init_p}): смен с продуктом {p} = {filled}"
                )
    except Exception:
        # Отладочная сводка не должна ломать основной расчёт
        pass

    # Фаза 2. Распределяем оставшиеся продукты (теперь только с переходами).

    def recompute_machine_free_slots() -> list[tuple[int, int]]:
        res: list[tuple[int, int]] = []
        for m_idx in range(num_machines):
            free_cnt = len(free_days_for_machine(m_idx))
            if free_cnt > 0:
                res.append((m_idx, free_cnt))
        # Сортируем по возрастанию числа свободных смен
        res.sort(key=lambda x: x[1])
        return res

    # Оставшиеся продукты распределяем в несколько шагов:
    # 1) тип 1 (machine_type=1): сначала строгие, затем гибкие;
    # 2) тип 0 (machine_type=0): сначала строгие, затем гибкие.
    remaining_all = [p for p, q in product_qty.items() if q > 0]

    def product_type(p_idx: int) -> int:
        # machine_type из исходного products_df по idx
        try:
            return int(products_df.loc[products_df["idx"] == p_idx, "machine_type"].iloc[0])
        except Exception:
            return 0

    remaining_type1 = [p for p in remaining_all if product_type(p) == 1]
    remaining_type0 = [p for p in remaining_all if product_type(p) == 0]

    # Внутри каждого типа сохраняем логику: сначала строгие (qty_minus=0), затем гибкие.
    def split_and_sort(plist: list[int]) -> list[int]:
        strict_part = [p for p in plist if p in strict_products]
        flex_part = [p for p in plist if p not in strict_products]
        strict_part.sort(key=lambda p: product_qty[p], reverse=True)
        flex_part.sort(key=lambda p: product_qty[p], reverse=True)
        return strict_part + flex_part

    remaining_products = split_and_sort(remaining_type1) + split_and_sort(remaining_type0)

    for p in remaining_products:
        while product_qty[p] > 0:
            machines_order = [m for (m, _) in recompute_machine_free_slots()]
            if not machines_order:
                break
            placed_any = False
            for m_idx in machines_order:
                if product_qty[p] <= 0:
                    break
                # Ограничение по типу машины/продукта.
                if not can_run_product_on_machine(p, m_idx):
                    continue
                free_days = free_days_for_machine(m_idx)
                if len(free_days) < 3:
                    continue
                # Ищем позицию для двухдневного перехода 0,0
                d0 = None
                for d in free_days:
                    if d + 1 >= count_days:
                        continue
                    if schedule[m_idx][d] is None and schedule[m_idx][d + 1] is None:
                        if (
                            can_place_zero(d, zeros_per_day, max_daily_prod_zero)
                            and can_place_zero(d + 1, zeros_per_day, max_daily_prod_zero)
                        ):
                            d0 = d
                            break
                if d0 is None:
                    continue

                # Ставим переход 0,0
                schedule[m_idx][d0] = 0
                schedule[m_idx][d0 + 1] = 0
                zeros_per_day[d0] = zeros_per_day.get(d0, 0) + 1
                zeros_per_day[d0 + 1] = zeros_per_day.get(d0 + 1, 0) + 1

                # Заполняем продуктом p дни после перехода
                for d in range(d0 + 2, count_days):
                    if product_qty[p] <= 0:
                        break
                    if schedule[m_idx][d] is None:
                        schedule[m_idx][d] = p
                        product_qty[p] -= 1
                placed_any = True

            if not placed_any:
                # Не получилось ни на одной машине поставить переход и продукт p — выходим,
                # чтобы избежать бесконечного цикла. Остаток qty для этого продукта останется.
                break

    # Фаза 3. Убираем пустоты и хвосты после последней рабочей смены без добавления длинных серий нулей.
    for m_idx, row in enumerate(schedule):
        # Есть ли на машине хоть один рабочий день (ненулевой продукт)?
        work_indices = [d for d, day in enumerate(row) if day not in (None, -2, 0)]
        if not work_indices:
            # Полностью пустая машина будет обрабатываться отдельно в Фазе 4.
            continue

        # 3.0. Сжимаем серии нулей: оставляем максимум две смены 0 подряд.
        d = 0
        while d < count_days:
            val = row[d]
            if isinstance(val, list):
                val = val[0] if val else None
            if val != 0:
                d += 1
                continue
            start = d
            while d < count_days:
                v = row[d]
                if isinstance(v, list):
                    v = v[0] if v else None
                if v != 0:
                    break
                d += 1
            end = d - 1
            length = end - start + 1
            if length <= 2:
                continue
            # Ищем продукт справа от блока нулей.
            right_p = None
            for k in range(end + 1, count_days):
                v = row[k]
                if isinstance(v, list):
                    v = v[0] if v else None
                if v not in (None, -2, 0):
                    right_p = v
                    break
            # Если справа ничего нет, ищем продукт слева.
            left_p = None
            if right_p is None:
                for k in range(start - 1, -1, -1):
                    v = row[k]
                    if isinstance(v, list):
                        v = v[0] if v else None
                    if v not in (None, -2, 0):
                        left_p = v
                        break
            fill_p = right_p if right_p is not None else left_p
            if fill_p is None:
                # Нечем заполнять середину, оставляем длинный блок как есть.
                continue
            # Оставляем переход 0,0 в начале блока, остальное заполняем продуктом fill_p.
            for k in range(start + 2, end + 1):
                if row[k] == -2:
                    continue
                row[k] = fill_p

        # 3.1. Заполняем внутренние None предыдущим продуктом (но не трогаем нули и чистки).
        prev_p = None
        for d in range(count_days):
            val = row[d]
            if isinstance(val, list):
                val = val[0] if val else None
            if val == -2:  # чистка
                # Не сбрасываем prev_p, чтобы после чистки
                # можно было продолжить последний рабочий продукт
                # до первого явного перехода 0,0.
                continue
            if val == 0:
                # Нули считаем переходами, не трогаем и не обновляем prev_p.
                continue
            if val is None:
                if prev_p is not None:
                    row[d] = prev_p
                continue
            # Обычный продукт
            prev_p = val

        # 3.2. Хвост после последнего рабочего дня заполняем продуктом из последней рабочей смены.
        last_idx = max(work_indices)
        last_val = row[last_idx]
        for d in range(last_idx + 1, count_days):
            if row[d] == -2:  # не трогаем чистки
                continue
            # Любые None/нуля/старые коды после последнего продукта превращаем в last_val.
            row[d] = last_val

    # Фаза 4. Заполняем полностью пустые машины продуктом с максимальным плановым qty.
    # Берём только гибкие продукты (qty_minus != 0), их можно превышать как угодно.
    main_product: int | None = None
    if "qty_minus" in products_df.columns:
        flex_df = products_df[(products_df["idx"] > 0) & (products_df["qty_minus"] != 0) & (products_df["qty"] > 0)]
        if not flex_df.empty:
            # Выбираем продукт с максимальным плановым количеством (уже с учётом масштабирования).
            row_max = flex_df.sort_values("qty", ascending=False).iloc[0]
            main_product = int(row_max["idx"])

    if main_product is not None:
        for m_idx, row in enumerate(schedule):
            # Полностью пустая машина: только None/0/-2
            has_work = any((day not in (None, -2, 0)) for day in row)
            if has_work:
                continue

            init_p = int(machines_df.at[m_idx, "product_idx"]) if "product_idx" in machines_df.columns else 0

            # Если машина не может производить main_product по типу,
            # просто заполняем её начальным продуктом (если он не ноль).
            if not can_run_product_on_machine(main_product, m_idx):
                if init_p != 0:
                    for k in range(count_days):
                        if row[k] == -2:
                            continue
                        row[k] = init_p
                continue

            # Если начальный продукт отсутствует или равен 0, просто заливаем main_product.
            if init_p == 0:
                for d in range(count_days):
                    if row[d] == -2:
                        continue
                    row[d] = main_product
                continue

            # Пытаемся сделать один переход init_p -> main_product через 0,0.
            placed_transition = False
            for d in range(0, count_days - 2):
                if row[d] == -2 or row[d + 1] == -2:
                    continue
                # Здесь сейчас только None или 0, но мы хотим ставить новый переход именно на свободные места
                if row[d] is None and row[d + 1] is None:
                    if (
                        can_place_zero(d, zeros_per_day, max_daily_prod_zero)
                        and can_place_zero(d + 1, zeros_per_day, max_daily_prod_zero)
                    ):
                        # До перехода работаем стартовым продуктом
                        for k in range(d):
                            if row[k] == -2:
                                continue
                            row[k] = init_p
                        # Сам переход 0,0
                        row[d] = 0
                        row[d + 1] = 0
                        zeros_per_day[d] = zeros_per_day.get(d, 0) + 1
                        zeros_per_day[d + 1] = zeros_per_day.get(d + 1, 0) + 1
                        # После перехода работаем main_product пока есть остаток qty
                        for k in range(d + 2, count_days):
                            if row[k] == -2:
                                continue
                            row[k] = main_product
                        placed_transition = True
                        break

            if not placed_transition:
                # Не удалось поставить переход (дневной лимит по нулям и т.п.) — заливаем стартовым продуктом.
                for k in range(count_days):
                    if row[k] == -2:
                        continue
                    row[k] = init_p

    # --- Подведение итогов ---

    proportions_input = products_df[products_df["idx"] > 0]["qty"].values
    total_work_days = sum(
        1 for machine in schedule for day in machine if day not in [-2, 0]
    )

    # Грубый коэффициент для штрафов по простоям
    denom = len([d for machine in schedule for d in machine if d != -2])
    if denom > 0 and len(proportions_input) > 0:
        kf_downtime_penalty = round(0.1 * sum(proportions_input) / denom)
    else:
        kf_downtime_penalty = 10
    if kf_downtime_penalty < 10:
        kf_downtime_penalty = 10

    # Подсчет отклонений пропорций (очень грубо)
    proportion_objective_terms = []
    for product_idx in products_df[products_df["idx"] > 0]["idx"]:
        planned_qty = sum(
            1 for machine in schedule for day in machine if day == product_idx
        )
        required_qty = products_df[products_df["idx"] == product_idx]["qty"].iloc[0]
        proportion = planned_qty / total_work_days if total_work_days > 0 else 0
        expected_proportion = required_qty / sum(proportions_input) if sum(proportions_input) > 0 else 0
        proportion_objective_terms.append(abs(round(proportion - expected_proportion)))

    # Подсчет переходов
    count_product_zero = sum(1 for machine in schedule for day in machine if day == 0)

    # Итоговый показатель (как пример)
    objective_value = sum(proportion_objective_terms) + count_product_zero * kf_downtime_penalty
    deviation_proportion = sum(proportion_objective_terms)

    # Допправка: если машина стартует с product_idx=0, не начинаем с перехода в день 0.
    for m_idx, row in enumerate(schedule):
        try:
            initial_p = int(machines[m_idx]["product_idx"])  # исходный список machines (dict)
        except Exception:
            continue
        if initial_p == 0 and row and row[0] == 0:
            # Ищем первый ненулевой продукт в более поздние дни и меняем местами с днём 0.
            for d in range(1, count_days):
                if row[d] not in (None, -2, 0):
                    row[0], row[d] = row[d], row[0]
                    break

    # Выводим получившийся жадный план в отдельный лог-файл (по сменам в строках).
    try:
        from pathlib import Path

        log_dir = Path("log")
        log_dir.mkdir(parents=True, exist_ok=True)
        greedy_log_path = log_dir / "greedy_schedule_init.log"
        with greedy_log_path.open("w", encoding="utf-8") as f:
            for m_idx, row in enumerate(schedule):
                init_p = int(machines_df.at[m_idx, "product_idx"]) if "product_idx" in machines_df.columns else -1
                m_type = int(machines_df.at[m_idx, "type"]) if "type" in machines_df.columns else 0
                codes: list[str] = []
                for val in row:
                    if isinstance(val, list):
                        val = val[0] if val else None
                    if val == -2:
                        codes.append("CL")
                    elif val == 0 or val is None:
                        codes.append("00")
                    else:
                        codes.append(f"{int(val):02d}")
                f.write(f"m={m_idx:02d}\ttype={m_type}\tinit={init_p:02d}\t" + ",".join(codes) + "\n")
    except Exception as e:
        logger.error(f"Не удалось записать жадный план в лог-файл: {e}")

    return schedule, objective_value, deviation_proportion, count_product_zero


def schedule_loom_calc(remains: list, products: list, machines: list, cleans: list, max_daily_prod_zero: int,
                       count_days: int, data: dict) -> LoomPlansOut:

    def MachinesDFToArray(machines_in: pd.DataFrame) -> list[(str, int, str, int, int)]:
        result = []
        idx = 0
        for index, item in machines_in.iterrows():
            if item["idx"] != idx:
                break
            result.append((item["name"], item["product_idx"], item["id"], item["type"], item["remain_day"]))
            idx += 1
        return result

    def ProductsDFToArray(products_in: pd.DataFrame) -> list[tuple]:
        """Преобразуем DataFrame продуктов в кортежи той же структуры, что и ProductsModelToArray.

        (name, qty, id, machine_type, qty_minus, lday, src_root, qty_minus_min, sr, strategy)
        """
        result = []
        first = True
        for index, item in products_in.iterrows():
            if first:
                first = False
                if item["qty"] > 0:
                    raise "Первый элемент продукции должен быть сменой артикула, т.е. количество плана = 0"
            src_root = item.get("src_root", -1)
            qty_minus_min = item.get("qty_minus_min", 0)
            sr = bool(item.get("sr", 0))
            strategy = item.get("strategy", "--")
            result.append((
                item["name"],
                int(item["qty"]),
                item["id"],
                int(item["machine_type"]),
                int(item["qty_minus"]),
                int(item["lday"]),
                int(src_root) if pd.notna(src_root) else -1,
                int(qty_minus_min) if pd.notna(qty_minus_min) else 0,
                bool(sr),
                str(strategy),
            ))
        return result

    def CleansDFToArray(cleans_in: pd.DataFrame) -> list[(int, int)]:
        result = []
        for _, item in cleans_in.iterrows():
            result.append((item["machine_idx"], item["day_idx"]))
        return result

    #machines_full = update_data_for_schedule_init(machines_new, products_new, cleans, count_days, schedule_init)


    class NursesPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
        """Callback для печати промежуточных решений и замера времени до первого решения."""

        def __init__(self, limit: int = -1, start_time: float | None = None):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._solution_count = 0
            self._solution_limit = limit
            self._start_time = start_time
            self._first_solution_time = None

        def on_solution_callback(self):
            self._solution_count += 1
            obj = self.ObjectiveValue()

            # Время до первого найденного допустимого решения
            if self._solution_count == 1 and self._start_time is not None:
                self._first_solution_time = time.time() - self._start_time
                logger.info(
                    f"Первое допустимое решение: objective={obj}, "
                    f"время={self._first_solution_time:.3f} сек"
                )

            print(f"Solution {self._solution_count}: {obj}")
            if self._solution_limit > 0 and self._solution_count >= self._solution_limit:
                self.stop_search()

        def solutionCount(self):
            return self._solution_count

    machines_df = pd.DataFrame(data["machines"])
    products_df = pd.DataFrame(data["products"])
    clean_df = pd.DataFrame(data["cleans"])
    schedule = []
    products_schedule = []
    diff_all = 0

    # Логируем ключевые настройки и размеры данных на момент создания модели.
    logger.info(
        "schedule_loom_calc: SOLVER_ENUMERATE=%s, SOLVER_ENUMERATE_COUNT=%s, "
        "APPLY_QTY_MINUS=%s, APPLY_INDEX_UP=%s, APPLY_DOWNTIME_LIMITS=%s, "
        "APPLY_ZERO_PER_DAY_LIMIT=%s, APPLY_ZERO_PER_MACHINE_LIMIT=%s, "
        "APPLY_THIRD_ZERO_BAN=%s, APPLY_PROP_OBJECTIVE=%s, APPLY_STRATEGY_PENALTY=%s, "
        "USE_GREEDY_HINT=%s; count_days=%s, max_daily_prod_zero=%s, machines=%s, products=%s, cleans=%s",
        settings.SOLVER_ENUMERATE,
        settings.SOLVER_ENUMERATE_COUNT,
        settings.APPLY_QTY_MINUS,
        settings.APPLY_INDEX_UP,
        settings.APPLY_DOWNTIME_LIMITS,
        settings.APPLY_ZERO_PER_DAY_LIMIT,
        settings.APPLY_ZERO_PER_MACHINE_LIMIT,
        settings.APPLY_THIRD_ZERO_BAN,
        settings.APPLY_PROP_OBJECTIVE,
        settings.APPLY_STRATEGY_PENALTY,
        settings.USE_GREEDY_HINT,
        count_days,
        max_daily_prod_zero,
        len(data["machines"]),
        len(data["products"]),
        len(data["cleans"]),
    )

    # Жадный предварительный план для использования как hint (опционально)
    greedy_schedule = None
    dedicated_machines: list[int] = []  # будет заполнен для LONG на основе greedy-анализа
    if settings.USE_GREEDY_HINT:
        try:
            greedy_schedule, _, _, _ = create_schedule_init(
                machines=data["machines"],
                products=data["products"],
                cleans=data["cleans"],
                count_days=count_days,
                max_daily_prod_zero=max_daily_prod_zero,
            )
            logger.info("Greedy initial schedule computed for hinting")

            # Для LONG-режима: ищем машины, которые greedy полностью заполнил
            # их начальными продуктами (без нулей и пустот), и у которых qty_minus_min>0.
            if greedy_schedule is not None and getattr(settings, "HORIZON_MODE", "FULL").upper() == "LONG":
                num_m = len(data["machines"])
                cleans_set = {(c["machine_idx"], c["day_idx"]) for c in data["cleans"]}

                # Быстрый доступ к qty_minus_min по product_idx исходных данных
                products_df_src = pd.DataFrame(data["products"])  # idx, qty_minus_min и т.п.
                qty_minus_min_map = products_df_src.set_index("idx")["qty_minus_min"].to_dict()

                dedicated_machines = []
                for m_idx in range(min(num_m, len(greedy_schedule))):
                    row = greedy_schedule[m_idx]
                    init_p = int(data["machines"][m_idx]["product_idx"])
                    if init_p == 0:
                        continue
                    qmm = int(qty_minus_min_map.get(init_p, 0) or 0)
                    if qmm <= 0:
                        continue
                    ok = True
                    has_work = False
                    for d in range(count_days):
                        if (m_idx, d) in cleans_set:
                            continue
                        if d >= len(row):
                            ok = False
                            break
                        p = row[d]
                        if isinstance(p, list):
                            p = p[0] if p else None
                        if p is None:
                            ok = False
                            break
                        if p == 0:
                            ok = False
                            break
                        if p != init_p:
                            ok = False
                            break
                        has_work = True
                    if ok and has_work:
                        dedicated_machines.append(m_idx)

                logger.info(
                    f"Dedicated machines (full initial product with qty_minus_min>0): {dedicated_machines}"
                )
        except Exception as e:
            logger.error(f"Ошибка при вычислении жадного плана для hint/locking: {e}")
            greedy_schedule = None
            dedicated_machines = []

    # Валидация входных данных: lday для продуктов (кроме нулевого) не должен быть 0 или отрицательным.
    invalid_products = []
    for p in products:
        # products: (name, qty, id, machine_type, qty_minus, lday, src_root, ...)
        name, qty, pid, machine_type, qty_minus, lday_val = p[:6]
        # p_idx нам не нужен для логики, но полезен для сообщения об ошибке — возьмём его из исходного списка.
        p_idx = products.index(p)
        if p_idx > 0 and lday_val <= 0:
            invalid_products.append((p_idx, name, lday_val))

    if invalid_products:
        # Формируем человекочитаемое сообщение об ошибке и логируем его.
        details = ", ".join(
            [f"idx={idx}, name='{name}', lday={lday_val}" for idx, name, lday_val in invalid_products]
        )
        error_msg = f"Некорректные данные: продукт(ы) имеют lday<=0: {details}"
        logger.error(error_msg)
        return {
            "status": int(cp_model.UNKNOWN),
            "status_str": "UNKNOWN",
            "schedule": schedule,
            "products": products_schedule,
            "objective_value": 0,
            "proportion_diff": 0,
            "error_str": error_msg,
        }

    product_id = products_df["id"]
    machines_df["product_id"] = machines_df["product_idx"].map(product_id)
    product_type = products_df["machine_type"]
    mapped_machine_types = machines_df['product_idx'].map(product_type)
    condition = (machines_df['type'] == 0) & (mapped_machine_types == 1)
    machines_df.loc[condition, 'type'] = 1
    product_zero = products_df[products_df["idx"] == 0]

    # Пересортируем продукты один раз на весь период, без разбивки по неделям.
    products_df_new = products_df.tail(-1)
    products_df_new = products_df_new.sort_values(by=['qty'])
    products_df_new = pd.concat([product_zero, products_df_new])

    products_df_zero = products_df[products_df["qty"] == 0]
    for index, p in products_df_zero.iterrows():
        if index > 0 and len(machines_df[machines_df["product_id"] == p["id"]]) == 0:
            products_df_new.drop(index)

    products_df_new['idx'] = range(len(products_df_new))
    id_to_new_idx_map = products_df_new.set_index('id')['idx']
    machines_df['product_idx'] = machines_df['product_id'].map(id_to_new_idx_map)

    products_new = ProductsDFToArray(products_df_new)
    machines_new = MachinesDFToArray(machines_df)
    cleans_new = CleansDFToArray(clean_df)

    solver = cp_model.CpSolver()

    long_mode = getattr(settings, "HORIZON_MODE", "FULL").upper() == "LONG"

    if long_mode:
        (model, jobs, product_counts, proportion_objective_terms, total_products_count, prev_lday, start_batch,
         batch_end_complite, days_in_batch, completed_transition, pred_start_batch, same_as_prev,
         strategy_penalty_terms) = create_model_long(
            remains=remains, products=products_new, machines=machines_new, cleans=cleans_new,
            max_daily_prod_zero=max_daily_prod_zero, count_days=count_days,
            dedicated_machines=dedicated_machines,
        )
    else:
        (model, jobs, product_counts, proportion_objective_terms, total_products_count, prev_lday, start_batch,
         batch_end_complite, days_in_batch, completed_transition, pred_start_batch, same_as_prev,
         strategy_penalty_terms) = create_model(
            remains=remains, products=products_new, machines=machines_new, cleans=cleans_new,
            max_daily_prod_zero=max_daily_prod_zero, count_days=count_days,
            dedicated_machines=dedicated_machines,
        )

    # Если есть жадный план и включён USE_GREEDY_HINT, используем его как hint для jobs[m,d].
    if settings.USE_GREEDY_HINT and greedy_schedule is not None:
        try:
            # Маппинг: исходный idx продукта -> id, и id -> новый idx в products_df_new
            orig_idx_to_id = products_df.set_index("idx")["id"].to_dict()
            id_to_new_idx = products_df_new.set_index("id")["idx"].to_dict()

            for (m, d), var in jobs.items():
                # jobs есть только для рабочих дней (без чисток)
                if m >= len(greedy_schedule) or d >= len(greedy_schedule[m]):
                    continue
                orig_p = greedy_schedule[m][d]
                if orig_p is None or orig_p in (-2,):
                    # None или -2 (чистка) — hint не ставим
                    continue
                # PRODUCT_ZERO остаётся 0
                if orig_p == 0:
                    hint_idx = 0
                else:
                    pid = orig_idx_to_id.get(int(orig_p))
                    if pid is None:
                        continue
                    new_idx = id_to_new_idx.get(pid)
                    if new_idx is None:
                        continue
                    hint_idx = int(new_idx)
                model.AddHint(var, hint_idx)
            logger.info("Greedy hints added to CP-SAT model")
        except Exception as e:
            logger.error(f"Ошибка при установке hints из жадного плана: {e}")

    # solver.parameters.log_search_progress = True
    #solver.parameters.debug_crash_on_bad_hint = True
    #solver.parameters.num_search_workers = 4

    t_start = time.time()
    if settings.SOLVER_ENUMERATE:
        sol_printer = NursesPartialSolutionPrinter(
            settings.SOLVER_ENUMERATE_COUNT,
            start_time=t_start,
        )
        solver.parameters.enumerate_all_solutions = True
        status = solver.solve(model, sol_printer)
    else:
        solver.parameters.max_time_in_seconds = settings.LOOM_MAX_TIME
        solver.parameters.num_search_workers = settings.LOOM_NUM_WORKERS
        status = solver.solve(model)
    t_end = time.time()

    logger.info(f"Статус решения: {solver.StatusName(status)}")
    logger.info(f"Время решения CP-SAT: {t_end - t_start:.3f} сек")

    logger.info(f"Статус решения: {solver.StatusName(status)}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Диагностический лог: сколько PRODUCT_ZERO (простоев) используется по дням.
        # Особенно полезно, когда дневной лимит по нулям отключен (APPLY_ZERO_PER_DAY_LIMIT=False):
        # можно увидеть, в какие дни модель фактически ставит больше нулей, чем max_daily_prod_zero.
        PRODUCT_ZERO = 0
        if not settings.APPLY_ZERO_PER_DAY_LIMIT:
            for d in range(count_days):
                zeros = 0
                for m in range(len(machines_new)):
                    if (m, d) in cleans_new:
                        continue
                    if solver.value(jobs[m, d]) == PRODUCT_ZERO:
                        zeros += 1
                logger.info(
                    f"diag_zero_per_day d={d}: zeros={zeros}, max_daily_prod_zero={max_daily_prod_zero}"
                )

        # Подробная диагностика по lday и батчам имеет смысл только в SHORT/FULL режимах.
        if settings.DEBUG_SCHEDULE and not long_mode:
            for m in range(len(machines_new)):
                s = ""
                s_bc = ""
                for d in range(count_days):
                    if (m, d) not in cleans_new:
                        p = solver.value(days_in_batch[m, d])
                        s = s + str(p) + ","
                        p = solver.value(batch_end_complite[m, d])
                        s_bc = s_bc + str(p) + ","
                    else:
                        s = s + " ,"
                        s_bc = s_bc + " ,"

                logger.info(f"days_in_batch {m}:       [{s}]")
                logger.info(f"batch_end_complite {m}:  [{s_bc}]")

                s = ""
                s_sb = ""
                s_ct = ""
                s_sp = ""
                for d in range(1, count_days):
                    if (m, d) not in cleans_new:
                        p = solver.value(prev_lday[m, d])
                        s = s + str(p) + ","
                        p = solver.value(start_batch[m, d])
                        s_sb = s_sb + str(p) + ","
                        p = solver.value(completed_transition[m, d])
                        s_ct = s_ct + str(p) + ","
                        p = solver.value(same_as_prev[m, d])
                        s_sp = s_sp + str(p) + ","
                    else:
                        s = s + " ,"
                        s_sb = s_sb + " ,"
                        s_ct = s_ct + " ,"
                        s_sp = s_sp + ","

                logger.info(f"same_as_prev {m}:          [{s_sp}]")
                logger.info(f"completed_transition {m}:  [{s_ct}]")
                logger.info(f"prev_lday {m}:             [{s}]")
                logger.info(f"start_batch {m}:           [{s_sb}]")

                s_sb = ""
                for d in range(2, count_days):
                    if (m, d) not in cleans_new:
                        p = solver.value(pred_start_batch[m, d])
                        s_sb = s_sb + str(p) + ","
                    else:
                        s_sb = s_sb + " ,"
                logger.info(f"pred_start_batch {m}:        [{s_sb}]")

        if proportion_objective_terms:
            logger.info(
                "Минимальное значение функции цели (сумма абс. отклонений пропорций): "
                f"{solver.ObjectiveValue()}"
            )

        solver_result(
            solver,
            status,
            machines,
            products,
            machines_new,
            products_new,
            cleans_new,
            count_days,
            proportion_objective_terms,
            product_counts,
            jobs,
            total_products_count,
            days_in_batch,
            prev_lday,
            schedule,
            products_schedule,
            diff_all,
            strategy_penalty_terms,
        )
        logger.info(solver.ResponseStats())  # Основные статистические данные

    result = {
        "status": int(status),
        "status_str": solver.StatusName(status),
        "schedule": schedule,
        "products": products_schedule,
        "objective_value": int(solver.ObjectiveValue()),
        "proportion_diff": int(diff_all),
        "error_str": "",
    }

    return result


def create_model_long(remains: list, products: list, machines: list, cleans: list,
                      max_daily_prod_zero: int, count_days: int,
                      dedicated_machines: list[int] | None = None):
    """Упрощённая модель для длинного горизонта (84 смены).

    - Не учитывает lday и remain_day.
    - Строит только jobs[m,d], product_counts[p], ограничения по нулям, типам машин,
      пропорциональную цель и стратегии.
    - Использует мастер-флаг APPLY_TRANSITION_BUSINESS_LOGIC только для запрета тройного нуля.
    """
    num_days = count_days
    num_machines = len(machines)
    num_products = len(products)

    dedicated_set = set(dedicated_machines or [])

    all_machines = range(num_machines)
    all_days = range(num_days)
    all_products = range(num_products)

    PRODUCT_ZERO = 0

    model = cp_model.CpModel()

    # Рабочие дни и переменные jobs[m,d]
    jobs: dict[tuple[int, int], cp_model.IntVar] = {}
    work_days: list[tuple[int, int]] = []

    for m in all_machines:
        for d in all_days:
            if (m, d) not in cleans:
                work_days.append((m, d))
                jobs[(m, d)] = model.NewIntVar(0, num_products - 1, f"job_{m}_{d}")
                if m in dedicated_set:
                    fixed_product = machines[m][1]  # product_idx из tuples machines: (name, product_idx, id, type, remain_day)
                    model.Add(jobs[(m, d)] == fixed_product)

    logger.debug(
        f"create_model_long: num_machines={num_machines}, num_days={num_days}, "
        f"work_days={len(work_days)}, max_daily_prod_zero={max_daily_prod_zero}"
    )

    # Булевы переменные "производится продукт p на (m,d)".
    product_produced_bools: dict[tuple[int, int, int], cp_model.BoolVar] = {}
    for p in all_products:
        for m, d in work_days:
            b = model.NewBoolVar(f"prod_{p}_{m}_{d}")
            product_produced_bools[p, m, d] = b
            model.Add(jobs[m, d] == p).OnlyEnforceIf(b)
            model.Add(jobs[m, d] != p).OnlyEnforceIf(b.Not())

    # Подсчёт количества смен каждого продукта.
    product_counts: list[cp_model.IntVar] = [
        model.NewIntVar(0, num_machines * num_days, f"count_prod_{p}") for p in all_products
    ]
    for p in all_products:
        model.Add(product_counts[p] == sum(product_produced_bools[p, m, d] for m, d in work_days))

    # Лимиты по нулям: в день и на машину (если включены).
    if settings.APPLY_DOWNTIME_LIMITS and settings.APPLY_ZERO_PER_DAY_LIMIT:
        for d in all_days:
            daily_zero = []
            for m in all_machines:
                if (m, d) in work_days:
                    daily_zero.append(product_produced_bools[PRODUCT_ZERO, m, d])
            model.Add(sum(daily_zero) <= max_daily_prod_zero)

    if settings.APPLY_DOWNTIME_LIMITS and settings.APPLY_ZERO_PER_MACHINE_LIMIT:
        weeks = max(1, count_days // 21)
        max_zero_per_machine = 2 * weeks
        for m in all_machines:
            zeros_m = []
            for d in all_days:
                if (m, d) in work_days:
                    zeros_m.append(product_produced_bools[PRODUCT_ZERO, m, d])
            model.Add(sum(zeros_m) <= max_zero_per_machine)

    # Ограничения по типам машин.
    for p in all_products:
        prod_type = products[p][3]
        if prod_type == 1:
            for m in all_machines:
                if machines[m][3] != 1:
                    for d in all_days:
                        if (m, d) in work_days:
                            model.Add(jobs[m, d] != p)

    # Упрощённая логика переходов: смена ненулевого продукта возможна только
    # через завершённый двухдневный простой (два PRODUCT_ZERO подряд).
    same_as_prev: dict[tuple[int, int], cp_model.BoolVar] = {}
    is_not_zero: dict[tuple[int, int], cp_model.BoolVar] = {}
    prev_is_not_zero: dict[tuple[int, int], cp_model.BoolVar] = {}
    completed_transition: dict[tuple[int, int], cp_model.BoolVar] = {}

    if settings.APPLY_TRANSITION_BUSINESS_LOGIC:
        # Создаём переменные completed_transition для всех рабочих дней.
        for m in all_machines:
            for d in all_days:
                if (m, d) in work_days:
                    completed_transition[m, d] = model.NewBoolVar(f"two_day_zero_long_{m}_{d}")

        for m in all_machines:
            # День 0: переход завершиться не может.
            if (m, 0) in work_days:
                model.Add(completed_transition[m, 0] == 0)

            for d in range(1, num_days):
                if (m, d) not in work_days:
                    continue

                # Определяем предыдущий рабочий день pred_idx (с учётом чисток).
                if (m, d - 1) in work_days:
                    pred_idx = d - 1
                else:
                    pred_idx = d - 2
                    if pred_idx < 0 or (m, pred_idx) not in work_days:
                        # Нет валидного предыдущего рабочего дня -> переход завершиться не мог.
                        model.Add(completed_transition[m, d] == 0)
                        continue

                # Булевы флаги для текущего и предыдущего дня.
                same = model.NewBoolVar(f"same_as_prev_long_{m}_{d}")
                same_as_prev[m, d] = same
                model.Add(jobs[m, d] == jobs[m, pred_idx]).OnlyEnforceIf(same)
                model.Add(jobs[m, d] != jobs[m, pred_idx]).OnlyEnforceIf(same.Not())

                inz = model.NewBoolVar(f"is_not_zero_long_{m}_{d}")
                is_not_zero[m, d] = inz
                model.Add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(inz)
                model.Add(jobs[m, d] == PRODUCT_ZERO).OnlyEnforceIf(inz.Not())

                prev_inz = model.NewBoolVar(f"prev_is_not_zero_long_{m}_{d}")
                prev_is_not_zero[m, d] = prev_inz
                model.Add(jobs[m, pred_idx] != PRODUCT_ZERO).OnlyEnforceIf(prev_inz)
                model.Add(jobs[m, pred_idx] == PRODUCT_ZERO).OnlyEnforceIf(prev_inz.Not())

                # completed_transition[m,d] = 1 <=> и pred_idx, и d — нули.
                model.AddBoolAnd([prev_inz.Not(), inz.Not()]).OnlyEnforceIf(completed_transition[m, d])
                model.AddBoolOr([prev_inz, inz]).OnlyEnforceIf(completed_transition[m, d].Not())

                # Основное правило:
                # - если текущий день ненулевой и продукт отличается от предыдущего,
                #   то к этому моменту должен быть завершён двухдневный простой
                #   (completed_transition на предыдущем рабочем дне).
                model.AddBoolOr([
                    inz.Not(),                # текущий день — ноль, всегда допустимо
                    same,                     # тот же продукт, что и на предыдущем рабочем дне
                    completed_transition[m, pred_idx],  # завершён двухдневный переход перед сменой продукта
                ])

        # Запрет тройного нуля подряд на одной машине.
        for m in all_machines:
            for d in range(2, num_days):
                if (m, d) in work_days and (m, d - 1) in work_days and (m, d - 2) in work_days:
                    model.Add(
                        product_produced_bools[PRODUCT_ZERO, m, d - 2]
                        + product_produced_bools[PRODUCT_ZERO, m, d - 1]
                        + product_produced_bools[PRODUCT_ZERO, m, d]
                        <= 2
                    )

    # Пропорциональная цель и стратегии (по аналогии с create_model, но без lday).
    proportions_input = [p[1] for p in products]

    total_products_count = model.NewIntVar(0, num_machines * num_days, "total_products_count")
    model.Add(total_products_count == sum(product_counts[p] for p in range(1, num_products)))

    proportion_objective_terms: list[cp_model.IntVar] = []
    total_input_quantity = 0
    total_input_max = 0
    if settings.APPLY_PROP_OBJECTIVE:
        # В LONG-режиме усиливаем PROP, но без тяжёлых произведений из FULL-модели.
        # Для qty_minus=0 оставляем только верхний предел (как раньше).
        # Для qty_minus=1 добавляем мягкий штраф за |product_counts[p] - planned_qty|.
        total_input_quantity = sum(proportions_input)
        total_input_max = max(proportions_input) if proportions_input else 0

        max_count = num_machines * num_days

        for p in range(1, num_products):
            planned_qty = int(proportions_input[p])
            if planned_qty <= 0:
                continue

            # products[p]: (name, qty, id, machine_type, qty_minus, lday, src_root, qty_minus_min, sr, strategy)
            qty_minus_flag = products[p][4] if len(products[p]) > 4 else 0

            if qty_minus_flag == 0:
                # qty_minus = 0: "жёсткий" продукт.
                # Нижняя граница: не меньше 0 смен.
                model.Add(product_counts[p] >= 0)
                # Верхняя граница: не более planned_qty + 6 смен (и не больше общего горизонта).
                upper = min(max_count, planned_qty + 6)
                model.Add(product_counts[p] <= upper)
                continue

            # qty_minus = 1: мягкий штраф за отклонение от планового количества.
            target = min(planned_qty, max_count)
            diff_var = model.NewIntVar(-max_count, max_count, f"prop_long_diff_{p}")
            model.Add(diff_var == product_counts[p] - target)
            abs_diff_var = model.NewIntVar(0, max_count, f"prop_long_absdiff_{p}")
            model.AddAbsEquality(abs_diff_var, diff_var)
            proportion_objective_terms.append(abs_diff_var)
    else:
        total_input_max = max(proportions_input) if proportions_input else 0

    # Стратегии (переносим существующую логику без lday).
    strategy_objective_terms: list[cp_model.IntVar] = []
    strategy_penalty_terms: list[cp_model.IntVar] = [
        model.NewIntVar(0, 0, f"strategy_penalty_long_{p}") for p in range(num_products)
    ]

    for p in range(1, num_products):
        if len(products[p]) < 10:
            continue
        strategy = products[p][9]
        if not strategy:
            continue

        count_start = model.NewIntVar(0, num_machines, f"machines_start_long_{p}")
        count_end = model.NewIntVar(0, num_machines, f"machines_end_long_{p}")

        start_bools = [product_produced_bools[p, m, d] for (m, d) in work_days if d == 0]
        end_bools = [product_produced_bools[p, m, d] for (m, d) in work_days if d == num_days - 1]

        if start_bools:
            model.Add(count_start == sum(start_bools))
        else:
            model.Add(count_start == 0)

        if end_bools:
            model.Add(count_end == sum(end_bools))
        else:
            model.Add(count_end == 0)

        weight = total_input_max
        pen: cp_model.IntVar | None = None

        if strategy == "--":
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * count_end)
        elif strategy == "-":
            weight = total_input_max
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_long_{p}")
            model.Add(diff == count_end - count_start)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines, f"strategy_posdiff_long_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * pos_diff)
        elif strategy == "=":
            weight = total_input_max * 2
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_long_{p}")
            model.Add(diff == count_end - count_start)
            abs_diff = model.NewIntVar(0, num_machines, f"strategy_absdiff_long_{p}")
            model.AddAbsEquality(abs_diff, diff)
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * abs_diff)
        elif strategy == "+":
            weight = total_input_max
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_long_{p}")
            model.Add(diff == count_start - count_end)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines, f"strategy_posdiff_long_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * pos_diff)
        elif strategy == "++":
            weight = total_input_max * 2
            diff = model.NewIntVar(-num_machines, num_machines + 1, f"strategy_diff_long_{p}")
            model.Add(diff == (count_start + 1) - count_end)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines + 1, f"strategy_posdiff_long_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = (num_machines + 1) * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * pos_diff)

        if pen is not None:
            strategy_penalty_terms[p] = pen
            if settings.APPLY_STRATEGY_PENALTY:
                strategy_objective_terms.append(pen)

    downtime_penalty = round(total_input_max * settings.KFZ_DOWNTIME_PENALTY)
    if downtime_penalty < 2:
        downtime_penalty = 2

    objective_terms: list[cp_model.LinearExpr] = []
    if settings.APPLY_PROP_OBJECTIVE and proportion_objective_terms:
        objective_terms.append(sum(proportion_objective_terms))

    objective_terms.append(product_counts[PRODUCT_ZERO] * downtime_penalty)

    if settings.APPLY_STRATEGY_PENALTY and strategy_objective_terms:
        objective_terms.append(sum(strategy_objective_terms))

    model.Minimize(sum(objective_terms))

    # Для совместимости с solver_result и вызывающим кодом возвращаем те же позиции,
    # но lday-связанные структуры заполняем пустыми словарями.
    prev_lday: dict = {}
    start_batch: dict = {}
    batch_end_complite: dict = {}
    days_in_batch: dict = {}
    pred_start_batch: dict = {}
    # completed_transition и same_as_prev здесь содержат только упрощённую логику переходов.

    return (
        model,
        jobs,
        product_counts,
        proportion_objective_terms,
        total_products_count,
        prev_lday,
        start_batch,
        batch_end_complite,
        days_in_batch,
        completed_transition,
        pred_start_batch,
        same_as_prev,
        strategy_penalty_terms,
    )


def create_model(remains: list, products: list, machines: list, cleans: list,
                 max_daily_prod_zero: int, count_days: int,
                 dedicated_machines: list[int] | None = None):
    # products: [ # ("idx, "name", "qty", "id", "machine_type", "qty_minus", "lday")
    #     ("", 0, "", 0, 0),
    #     ("ст87017t3", 42, "7ec17dc8-f3bd-4384-9738-7538ab3dc315", 0, 1, 13),
    #     ("ст87416t1", 15, "9559e2e8-6e72-41f8-9dba-08aab5463623", 0, 1, 14),
    #     ("ст2022УИСt4", 4, "cd825c90-aa80-4b95-9f81-2486b871bf94", 0, 0, 20)
    # ]
    # machines = [ # (name, product_idx, id, type, remain_day)
    #   ("ТС Тойота №1", 1, "fbc4c3a0-8087-11ea-80cc-005056aab926", 0, 2),
    #   ("ТС Тойота №2", 1, "fbc4c3a1-8087-11ea-80cc-005056aab926", 0, 5),
    #   ("ТС Тойота №3", 3, "fbc4c372-8087-11ea-80cc-005056aab926", 0, 4),
    # ]
    # cleans: [ # ("machine_idx", "day_idx")
    # (3, 1)
    # max_daily_prod_zero = 3

    num_days = count_days
    num_machines = len(machines)
    num_products = len(products)

    long_mode = getattr(settings, "HORIZON_MODE", "FULL").upper() == "LONG"

    dedicated_set = set(dedicated_machines or [])

    all_machines = range(num_machines)
    all_days = range(num_days)
    all_products = range(num_products)

    proportions_input = [p[1] for p in products]

    # Учёт остатков главного сырья по src_root.
    # Если remains имеет формат списка списков, то для продукта
    # с src_root >= 0 берём длину партии как усреднённое значение по фактическим
    # длинам партий, ИСКЛЮЧАЯ короткие партии (< 6 смен). Если после фильтрации
    # ничего не осталось, используем lday из products.
    #
    # Поддерживаем два формата данных остатков:
    # 1) [[lday1, lday2, ...], ...] — прямой список по src_root.
    # 2) [[[...], [...], ...], ...] — несколько наборов остатков, где
    #    главный ресурс (главное сырьё) сейчас находится с индексом 0.
    ldays: list[int] = []
    remains_batches: list[list[int]] = []

    if not long_mode and remains:
        first = remains[0]
        # Формат 1: прямой список списков длительностей партий.
        if isinstance(first, list) and (not first or isinstance(first[0], (int, float))):
            remains_batches = remains
        # Формат 2: внешний уровень — наборы остатков по разным видам ресурсов,
        # главный ресурс (главное сырьё) с индексом 0.
        elif isinstance(first, list) and first and isinstance(first[0], list):
            main_resource_idx = 0  # главный ресурс сейчас с индексом 0 в таблице остатков
            if 0 <= main_resource_idx < len(remains):
                remains_batches = remains[main_resource_idx]
        # Иные форматы (например, список моделей Remain) пока не используем для расчёта lday.

    for p_idx, p in enumerate(products):
        base_lday = p[5]
        lday_eff = base_lday
        src_root = p[6] if len(p) > 6 else -1
        if (not long_mode
            and isinstance(src_root, int)
            and src_root >= 0
            and src_root < len(remains_batches)):
            batches = remains_batches[src_root]
            if isinstance(batches, list) and len(batches) > 0:
                # Отбрасываем короткие партии (< 6 смен)
                long_batches = [int(b) for b in batches if isinstance(b, (int, float)) and int(b) >= 6]
                if long_batches:
                    avg_lday = round(sum(long_batches) / len(long_batches))
                    if avg_lday > 0:
                        lday_eff = avg_lday
        ldays.append(lday_eff)

    # В LONG-режиме lday не используем для ограничений: ставим всем продуктам одинаковую длину партии 1 (кроме нулевого).
    if long_mode:
        ldays = [0] + [1] * (num_products - 1)

    initial_products = []
    days_to_constrain = []
    for idx, (_, product_idx, m_id, t, remain_day) in enumerate(machines):
        initial_products.append(product_idx)
        days_to_constrain.append(remain_day)

    # В LONG-режиме остатки партий (remain_day) игнорируем
    if long_mode:
        days_to_constrain = [0 for _ in range(num_machines)]

    model = cp_model.CpModel()

    jobs = {}
    prev_lday = {}
    max_lday = max(ldays) if ldays else 1
    work_days = []
    # Значение для отображения чистки в итоговом расписании
    for m in range(num_machines):
        for d in range(num_days):
            if (m, d) not in cleans:
                work_days.append((m, d))
                # Домен переменной: от 0 до num_products - 1
                jobs[(m, d)] = model.new_int_var(0, num_products - 1, f"job_{m}_{d}")
                # Для dedicated-машин жестко фиксируем исходный продукт на всех рабочих днях.
                if m in dedicated_set:
                    fixed_product = initial_products[m]
                    model.Add(jobs[(m, d)] == fixed_product)
                prev_lday[m, d] = model.NewIntVar(0, max_lday, f'prev_lday_m{m}_d{d}')
                model.AddElement(jobs[m, d], ldays, prev_lday[m, d])

    # Диагностический лог по дням партии и ёмкости по машинам
    logger.debug(
        f"Создание модели: num_machines={num_machines}, num_days={num_days}, "
        f"work_days={len(work_days)}, max_daily_prod_zero={max_daily_prod_zero}"
    )
    for p_idx, p in enumerate(products):
        logger.debug(
            f"lday[{p_idx}]={ldays[p_idx]}, qty={p[1]}, qty_minus={p[4]}, "
            f"src_root={p[6] if len(p) > 6 else -1}"
        )

    logger.debug("Сводка по продуктам: минимально требуемое количество и машинные дни ёмкости")
    for p in all_products:
        qty = products[p][1]
        qty_minus_flag = products[p][4]
        qty_minus_min = products[p][7] if len(products[p]) > 7 else 0
        # Минимально требуемое количество по нашему же правилу наложения ограничений
        min_required = 0
        if settings.APPLY_QTY_MINUS and qty > 0:
            if qty_minus_flag == 0:
                min_required = qty
            else:
                if qty_minus_min > 0:
                    min_required = qty_minus_min
        # Оценка доступной ёмкости по машинным дням с учётом типов машин
        product_machine_type_req = products[p][3]
        capacity = 0
        for (m, d) in work_days:
            machine_type = machines[m][3]
            if product_machine_type_req == 1 and machine_type != 1:
                continue
            capacity += 1
        logger.debug(
            f"Product {p}: qty={qty}, qty_minus={qty_minus_flag}, qty_minus_min={qty_minus_min}, "
            f"min_required={min_required}, capacity_machine_days={capacity}, lday={ldays[p]}"
        )

    PRODUCT_ZERO = 0  # Индекс "особенной" продукции

    # ------------ Подсчет общего количества каждого продукта ------------
    # Вспомогательные булевы переменные: product_produced[p, m, d] истинно, если продукт p производится на машине m в день d
    product_produced_bools = {}

    for p in all_products:
        for m, d in work_days:
            product_produced_bools[p, m, d] = model.NewBoolVar(f"product_produced_{p}_{m}_{d}")
            # Связь product_produced_bools с jobs
            model.Add(jobs[m, d] == p).OnlyEnforceIf(product_produced_bools[p, m, d])
            model.Add(jobs[m, d] != p).OnlyEnforceIf(product_produced_bools[p, m, d].Not())

    # Общее количество каждого продукта: product_counts[p]
    product_counts = [model.NewIntVar(0, num_machines * num_days, f"count_prod_{p}") for p in range(num_products)]
    for p in all_products:
        model.Add(product_counts[p] == sum(
            product_produced_bools[p, m, d] for m, d in work_days))
        # Добавляем ограничение для управления минимальными/строгими объёмами.
        # products[p][1] - требуемое количество (qty)
        # products[p][4] - признак qty_minus (0 - строго, 1 - можно недопланировать)
        # products[p][7] - qty_minus_min (минимальное количество планирования при qty_minus=1)
        if settings.APPLY_QTY_MINUS and products[p][1] > 0:
            qty = products[p][1]
            qty_minus_flag = products[p][4]
            qty_minus_min = products[p][7] if len(products[p]) > 7 else 0

            if qty_minus_flag == 0:
                # Строгий продукт: объём должен быть ровно равен входному qty.
                model.Add(product_counts[p] == qty)
            else:
                # Мягкое ограничение: количество не меньше qty_minus_min (если задано)
                if qty_minus_min > 0:
                    model.Add(product_counts[p] >= qty_minus_min)

    # Сумма PRODUCT_ZERO в смену d не более max_daily_prod_zero
    # Количество нулевого продукта по дням
    # И просто количество нулевого продукта
    if settings.APPLY_DOWNTIME_LIMITS and settings.APPLY_ZERO_PER_DAY_LIMIT:
        for d in all_days:
            daily_prod_zero_on_machines = []
            for m in range(num_machines):
                if (m, d) in work_days:
                    # Используем уже созданные product_produced_bools для эффективности
                    # product_produced_bools[PRODUCT_ZERO, m, d] истинно, если на машине m в день d производится PRODUCT_ZERO
                    daily_prod_zero_on_machines.append(product_produced_bools[PRODUCT_ZERO, m, d])

            # Сумма этих булевых переменных даст количество PRODUCT_ZERO в день d
            model.Add(sum(daily_prod_zero_on_machines) <= max_daily_prod_zero)

    # Ограничение по типам машин ###
    # Продукты с типом 1 могут производиться только на машинах с типом 1.
    # Продукты с типом 0 могут производиться на любых машинах.
    for p in all_products:
        # Индекс 3 в кортеже продукта - это 'machine_type'
        product_machine_type_req = products[p][3]
        if product_machine_type_req == 1:
            for m in all_machines:
                # Индекс 3 в кортеже машины - это 'type'
                machine_type = machines[m][3]
                if machine_type != 1:
                    # Эта машина не может производить данный продукт.
                    # Запрещаем назначение этого продукта на эту машину во все дни.
                    for d in all_days:
                        if (m, d) in work_days:
                            model.Add(jobs[m, d] != p)

    # Основная переменная состояния: отслеживает день внутри текущей партии
    days_in_batch = {}
    for m in all_machines:
        for d in all_days:
            days_in_batch[m, d] = model.NewIntVar(0, max_lday, f'days_in_batch_m{m}_d{d}')

    batch_end_complite = {}

    # Ограничения ПЕРЕХОДА
    # Переменные для отслеживания завершения двухдневного перехода
    completed_transition = {}
    is_not_zero = {}
    same_as_prev = {}
    prev_is_not_zero = {}

    start_batch = {}
    for m in range(num_machines):
        for d in range(num_days):
            completed_transition[m, d] = model.NewBoolVar(f"completed_transition_{m}_{d}")

    remain_day = [0 for _ in range(num_machines)]
    # Ограничение для первого дня (d=0)
    for m in range(num_machines):
        initial_product = initial_products[m]
        is_initial_product = model.NewBoolVar(f"is_initial_product_{m}_0")
        is_not_zero[m, 0] = model.NewBoolVar(f"is_not_zero_{m}_0")
        product_lday = ldays[initial_product]
        batch_end_complite[m, 0] = model.NewBoolVar(f"batch_end_complite_m{m}_d0")

        # Особый случай: если стартовый продукт = PRODUCT_ZERO (idx=0),
        # то в первый день ПЕРЕХОД НЕ ТРЕБУЕТСЯ.
        # Можно сразу ставить любой ненулевой продукт без двухдневного перехода.
        if initial_product == PRODUCT_ZERO:
            # Игнорируем days_to_constrain для нулевого продукта: считаем, что остатка партии нет.
            # День 0: либо простой (PRODUCT_ZERO), либо сразу ненулевой продукт.
            model.Add(jobs[m, 0] == PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 0].Not())
            model.Add(jobs[m, 0] != PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 0])

            # Если в первый день поставили ненулевой продукт, это начало новой партии (days_in_batch=1).
            model.Add(days_in_batch[m, 0] == 1).OnlyEnforceIf(is_not_zero[m, 0])
            # Если оставили PRODUCT_ZERO, то длина партии 0.
            model.Add(days_in_batch[m, 0] == 0).OnlyEnforceIf(is_not_zero[m, 0].Not())
        elif days_to_constrain[m] > 0:
            remain_day[m] += 1
            model.Add(jobs[m, 0] == initial_product)
            # выставляем начальное значение остатка партии
            start_val = product_lday - days_to_constrain[m] + 1
            model.Add(days_in_batch[m, 0] == start_val)
        else:
            model.Add(jobs[m, 0] == initial_product).OnlyEnforceIf(is_initial_product)
            model.Add(jobs[m, 0] != initial_product).OnlyEnforceIf(is_initial_product.Not())
            model.Add(jobs[m, 0] == PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 0].Not())
            model.Add(jobs[m, 0] != PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 0])

            # Первый день: либо начальный продукт, либо PRODUCT_ZERO
            model.AddBoolOr([is_initial_product, is_not_zero[m, 0].Not()])
            # Первый день: либо начало партии, либо 0, если переход
            model.Add(days_in_batch[m, 0] == 1).OnlyEnforceIf([is_initial_product, is_not_zero[m, 0]])
            model.Add(days_in_batch[m, 0] == 0).OnlyEnforceIf(is_not_zero[m, 0].Not())

        # Устанавливаем completed_transition для дня 0
        model.Add(completed_transition[m, 0] == 0)  # Нет перехода в день 0

        model.Add(days_in_batch[m, 0] == prev_lday[m, 0]).OnlyEnforceIf(batch_end_complite[m, 0])
        model.Add(days_in_batch[m, 0] != prev_lday[m, 0]).OnlyEnforceIf(batch_end_complite[m, 0].Not())

    # Логика переходов для дней d ≥ 1
    pred_start_batch = {}
    for m in range(num_machines):
        for d in range(1, num_days):
            initial_product = initial_products[m]
            if (m, d) not in cleans and (m, d - 1) not in cleans:
                pred_idx = d - 1
            elif (m, d) in cleans:
                continue
            elif (m, d - 1) in cleans:
                pred_idx = d - 2

            if days_to_constrain[m] > remain_day[m]:
                model.Add(jobs[m, d] == initial_product)
                remain_day[m] += 1

            same_as_prev[m, d] = model.NewBoolVar(f"same_as_prev_{m}_{d}")
            model.Add(jobs[m, d] == jobs[m, pred_idx]).OnlyEnforceIf(same_as_prev[m, d])
            model.Add(jobs[m, d] != jobs[m, pred_idx]).OnlyEnforceIf(same_as_prev[m, d].Not())

            is_not_zero[m, d] = model.NewBoolVar(f"is_not_zero_{m}_{d}")
            model.Add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, d])
            model.Add(jobs[m, d] == PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, d].Not())

            prev_is_not_zero[m, d] = model.NewBoolVar(f"prev_is_not_zero_{m}_{d}")
            model.Add(jobs[m, pred_idx] != PRODUCT_ZERO).OnlyEnforceIf(prev_is_not_zero[m, d])
            model.Add(jobs[m, pred_idx] == PRODUCT_ZERO).OnlyEnforceIf(prev_is_not_zero[m, d].Not())

            # Проверяем, был ли завершен двухдневный переход
            completed_transition[m, d] = model.NewBoolVar(f"two_day_zero_{m}_{d}")
            model.AddBoolAnd(prev_is_not_zero[m, d].Not(), is_not_zero[m, d].Not()).OnlyEnforceIf(
                completed_transition[m, d])
            model.AddBoolOr(prev_is_not_zero[m, d], is_not_zero[m, d]).OnlyEnforceIf(
                completed_transition[m, d].Not())

            # логика расчета количества партии

            model.Add(days_in_batch[m, d] == 0).OnlyEnforceIf(is_not_zero[m, d].Not())
            model.Add(days_in_batch[m, d] == 1).OnlyEnforceIf([batch_end_complite[m, pred_idx], is_not_zero[m, d]])
            model.Add(days_in_batch[m, d] == days_in_batch[m, pred_idx] + 1).OnlyEnforceIf([batch_end_complite[m, pred_idx].Not(), is_not_zero[m, d]])

            batch_end_complite[m, d] = model.NewBoolVar(f"batch_end_complite_m{m}_d{d}")
            model.Add(days_in_batch[m, d] == prev_lday[m ,d]).OnlyEnforceIf(batch_end_complite[m, d])
            model.Add(days_in_batch[m, d] != prev_lday[m, d]).OnlyEnforceIf(batch_end_complite[m, d].Not())

            pred_start_batch[m, d] = model.NewBoolVar(f"pred_start_batch_m{m}_d{d}")
            model.AddBoolAnd([same_as_prev[m, d], batch_end_complite[m, pred_idx]]).OnlyEnforceIf(
                pred_start_batch[m, d])
            model.AddBoolOr([same_as_prev[m, d].Not(), batch_end_complite[m, pred_idx].Not()]).OnlyEnforceIf(
                pred_start_batch[m, d].Not())

            start_batch[m, d] = model.NewBoolVar(f"start_batch_m{m}_d{d}")
            model.AddBoolOr([pred_start_batch[m, d], completed_transition[m, d]]).OnlyEnforceIf(
                start_batch[m, d])
            model.AddBoolAnd([pred_start_batch[m, d].Not(), completed_transition[m, d].Not()]).OnlyEnforceIf(
                start_batch[m, d].Not())

            model.Add(jobs[m, d] == jobs[m, pred_idx]).OnlyEnforceIf([batch_end_complite[m, pred_idx].Not(), prev_is_not_zero[m, d]])


            # ### НАЧАЛО НОВОГО БЛОКА: Ограничение на повышение индекса продукта ###
            # Это ограничение срабатывает только в день `d`, когда завершился двухдневный переход,
            # что определяется переменной completed_transition[m, d].

            # 1. Находим индекс рабочего дня перед началом перехода.
            #    Переход занимал дни `pred_pred_idx` и `pred_idx`. Ищем день до `pred_pred_idx`.
            day_before_transition_start = pred_idx - 2
            while day_before_transition_start >= 0 and (m, day_before_transition_start) in cleans:
                day_before_transition_start -= 1

            # 2. Применяем ограничение, только если такой день существует в расписании.
            if day_before_transition_start >= 0 and settings.APPLY_INDEX_UP:
                # Переменная, указывающая на продукт до начала перехода.
                product_before = jobs[(m, day_before_transition_start)]
                product_before_is_not_zero = model.NewBoolVar(f"prod_before_not_zero_{m}_{d}")
                model.Add(product_before != PRODUCT_ZERO).OnlyEnforceIf(product_before_is_not_zero)
                model.Add(product_before == PRODUCT_ZERO).OnlyEnforceIf(product_before_is_not_zero.Not())

                # model.Add(jobs[m, d] != product_before).OnlyEnforceIf(
                #     [completed_transition[m, pred_idx], product_before_is_not_zero]
                # )
                model.Add(jobs[m, d] > product_before).OnlyEnforceIf(
                    [completed_transition[m, pred_idx], product_before_is_not_zero]
                )
            # ### КОНЕЦ НОВОГО БЛОКА ###

            # Ограничения:
            # Если текущий день - не ноль, то либо:
            # 1) тот же продукт, что и вчера (если вчера не ноль)
            # 2) завершен двухдневный переход

            model.AddBoolOr([
                is_not_zero[m, d].Not(),  # Текущий день - PRODUCT_ZERO
                same_as_prev[m, d],  # Тот же продукт, что вчера
                completed_transition[m, pred_idx]  # Завершен двухдневный переход
            ])
            # Запрет на 3-й ZERO: после двух дней PRODUCT_ZERO (переход)
            # третий день на той же машине уже не может быть нулевым продуктом.
            if settings.APPLY_TRANSITION_BUSINESS_LOGIC:
                model.Add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(completed_transition[m, pred_idx])

    # не более 1-го простоя (двухдневного перехода) за неделю на каждую машину.
    # При планировании на более долгий период масштабируем ограничение по числу недель.
    if settings.APPLY_DOWNTIME_LIMITS and settings.APPLY_ZERO_PER_MACHINE_LIMIT:
        weeks = max(1, count_days // 21)
        max_zero_per_machine = 2 * weeks  # 2 дня простоя на неделю
        for m in range(num_machines):
            prod_zero_on_machine = []
            for d in all_days:
                if not (m, d) in cleans:
                    prod_zero_on_machine.append(product_produced_bools[PRODUCT_ZERO, m, d])
            model.Add(sum(prod_zero_on_machine) <= max_zero_per_machine)

    # ------------ Мягкое ограничение: Пропорции продукции (для продуктов с индексом > 0) ------------
    # Цель: минимизировать отклонение фактических долей от входных qty.
    # Используем облегчённую относительную формулу:
    #   product_counts[p] * total_input_quantity  ≈  total_products_count * qty[p]
    # и штрафуем |term1 - term2|, но с ограниченными доменами и малым весом в целевой функции.

    total_products_count = model.NewIntVar(0, num_machines * num_days, "total_products_count")
    model.Add(total_products_count == sum(product_counts[p] for p in range(1, len(products))))

    proportion_objective_terms: list[cp_model.IntVar] = []
    total_input_quantity = 0
    total_input_max = 0
    if settings.APPLY_PROP_OBJECTIVE:
        total_input_quantity = sum(proportions_input)
        total_input_max = max(proportions_input) if proportions_input else 0
        logger.debug(f"total_input_quantity={total_input_quantity}")

        # Грубая верхняя граница для произведений: максимум дней * машин * суммарное qty.
        max_count = num_machines * num_days
        max_prod = max_count * max(1, total_input_quantity)

        for p in range(1, len(products)):  # Skip p == 0
            qty_minus_flag = products[p][4]
            if qty_minus_flag == 0:
                # Строгие продукты с qty_minus=0 не участвуют в пропорциональной цели:
                # для них объём зафиксирован жёстко через product_counts[p] == qty.
                continue

            logger.debug(f"proportions_input[{p}]={proportions_input[p]}")

            # term1 = product_counts[p] * total_input_quantity
            term1_expr = model.NewIntVar(0, max_prod, f"prop_term1_{p}")
            model.AddMultiplicationEquality(term1_expr, [product_counts[p], total_input_quantity])

            # term2 = total_products_count * qty[p]
            planned_qty = proportions_input[p]
            term2_expr = model.NewIntVar(0, max_prod, f"prop_term2_{p}")
            model.AddMultiplicationEquality(term2_expr, [total_products_count,
                                                         model.NewConstant(planned_qty)])

            # diff = term1_expr - term2_expr
            diff_var = model.NewIntVar(-max_prod, max_prod, f"prop_diff_{p}")
            model.Add(diff_var == (term1_expr - term2_expr))
            abs_diff_var = model.NewIntVar(0, max_prod, f"prop_absdiff_{p}")
            model.AddAbsEquality(abs_diff_var, diff_var)
            proportion_objective_terms.append(abs_diff_var)
    else:
        # При отключённых пропорциях заводим простые константы, чтобы остальной код мог безопасно обращаться к списку.
        total_input_max = max(proportions_input) if proportions_input else 0

    # ------------ Стратегии изменения количества машин по продуктам ------------
    # Для каждого продукта p > 0 учитываем количество машин в первый и последний день.
    strategy_objective_terms: list[cp_model.IntVar] = []
    # Переменные штрафа по стратегии для логирования и анализа.
    strategy_penalty_terms: list[cp_model.IntVar] = [
        model.NewIntVar(0, 0, f"strategy_penalty_{p}") for p in range(num_products)
    ]

    for p in range(1, len(products)):
        if len(products[p]) < 10:
            continue
        strategy = products[p][9]
        if not strategy:
            continue

        # Количество машин с продуктом p в первый и последний день периода
        count_start = model.NewIntVar(0, num_machines, f"machines_start_{p}")
        count_end = model.NewIntVar(0, num_machines, f"machines_end_{p}")

        start_bools = [product_produced_bools[p, m, d] for (m, d) in work_days if d == 0]
        end_bools = [product_produced_bools[p, m, d] for (m, d) in work_days if d == num_days - 1]

        if start_bools:
            model.Add(count_start == sum(start_bools))
        else:
            model.Add(count_start == 0)

        if end_bools:
            model.Add(count_end == sum(end_bools))
        else:
            model.Add(count_end == 0)

        # Переменная штрафа по стратегии для данного продукта
        # По умолчанию штраф 0, далее для каждой стратегии мы задаём pen >= 0
        # и связываем его с нарушением соответствующего неравенства.
        # Общий подход: penalty = weight * max(0, expr), где expr зависит от стратегии.
        weight = total_input_max
        pen = None

        if strategy == "--":
            # В конце периода нежелательно иметь машины с этим продуктом.
            # Штраф пропорционален количеству машин в последний день.
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_{p}")
            model.Add(pen == weight * count_end)

        elif strategy == "-":
            # В конце должно быть НЕ БОЛЬШЕ машин, чем в начале: end <= start.
            # Штрафуем только случаи end > start: penalty ~ (end - start).
            weight = total_input_max
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_{p}")
            model.Add(diff == count_end - count_start)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines, f"strategy_posdiff_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_{p}")
            model.Add(pen == weight * pos_diff)

        elif strategy == "=":
            # Желательно то же количество машин в начале и в конце: end ~= start.
            # Штрафуем |end - start|.
            weight = total_input_max * 2
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_{p}")
            model.Add(diff == count_end - count_start)
            abs_diff = model.NewIntVar(0, num_machines, f"strategy_absdiff_{p}")
            model.AddAbsEquality(abs_diff, diff)
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_{p}")
            model.Add(pen == weight * abs_diff)

        elif strategy == "+":
            # В конце должно быть НЕ МЕНЬШЕ машин, чем в начале: end >= start.
            # Штрафуем только случаи end < start: penalty ~ (start - end).
            weight = total_input_max
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_{p}")
            model.Add(diff == count_start - count_end)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines, f"strategy_posdiff_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_{p}")
            model.Add(pen == weight * pos_diff)

        elif strategy == "++":
            # В конце должно быть ЗАМЕТНО больше машин, чем в начале.
            # Например: end >= start + 1.
            # Штрафуем только случаи, когда end < start + 1: penalty ~ ((start + 1) - end).
            weight = total_input_max * 2
            diff = model.NewIntVar(-num_machines, num_machines + 1, f"strategy_diff_{p}")
            model.Add(diff == (count_start + 1) - count_end)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines + 1, f"strategy_posdiff_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = (num_machines + 1) * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_{p}")
            model.Add(pen == weight * pos_diff)

        else:
            # Неизвестная стратегия — штраф не накладываем.
            continue

        if pen is not None:
            strategy_penalty_terms[p] = pen
            if settings.APPLY_STRATEGY_PENALTY:
                strategy_objective_terms.append(pen)

    downtime_penalty = round(total_input_max * settings.KFZ_DOWNTIME_PENALTY)
    if downtime_penalty < 2:
        downtime_penalty = 2

    objective_terms: list[cp_model.LinearExpr] = []
    if settings.APPLY_PROP_OBJECTIVE and proportion_objective_terms:
        # Пока не нормируем вес пропорций, но домены этих переменных уже сильно ограничены.
        # При необходимости масштаб можно будет подстроить через KFZ_DOWNTIME_PENALTY.
        objective_terms.append(sum(proportion_objective_terms))

    # Штраф за простои (нулевой продукт) всегда учитываем.
    objective_terms.append(product_counts[PRODUCT_ZERO] * downtime_penalty)

    if settings.APPLY_STRATEGY_PENALTY and strategy_objective_terms:
        objective_terms.append(sum(strategy_objective_terms))

    model.Minimize(sum(objective_terms))


    return (
        model,
        jobs,
        product_counts,
        proportion_objective_terms,
        total_products_count,
        prev_lday,
        start_batch,
        batch_end_complite,
        days_in_batch,
        completed_transition,
        pred_start_batch,
        same_as_prev,
        strategy_penalty_terms,
    )


def solver_result(
    solver,
    status,
    machines_old,
    products_old,
    machines,
    products,
    cleans,
    count_days,
    proportion_objective_terms,
    product_counts,
    jobs,
    total_products_count,
    days_in_batch,
    prev_lday,
    schedule,
    products_schedule,
    diff_all,
    strategy_penalty_terms,
):

    long_mode = getattr(settings, "HORIZON_MODE", "FULL").upper() == "LONG"

    def find_machine_id_old(machine_idx: int):
        for i, machine in enumerate(machines_old):
            if machine[2] == machines[machine_idx][2]:
                return i
        raise f"Не нашли id машины {machines[machine_idx][2]}"

    def find_product_id_old(product_idx: int):
        for i, product in enumerate(products_old):
            if product[2] == products[product_idx][2]:
                return i, product[2]
        raise f"Не нашли id продукта {products[product_idx][2]}"

    num_products = len(products)
    num_machines = len(machines)

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return
    for m in range(num_machines):
        m_old = find_machine_id_old(m)
        logger.debug(f"Loom {m_old}")
        for d in range(count_days):
            if not (m, d) in cleans:
                p = solver.value(jobs[m, d])
                p_old, p_id = find_product_id_old(p)
                if not long_mode and (m, d) in days_in_batch and (m, d) in prev_lday:
                    db_v = solver.value(days_in_batch[m, d])
                    plday = solver.value(prev_lday[m, d])
                else:
                    db_v = None
                    plday = None
            else:
                p_old = None
                p_id = ""
                db_v = None
                plday = None
            schedule.append({"machine_idx": m_old, "day_idx": d, "product_idx": p_old,
                             "days_in_batch": db_v, "prev_lday": plday})
            logger.debug(f"  Day {d} works  {p_old}")

    logger.debug("\nОбщее количество произведенной продукции:")
    logger.debug(f"\n{solver.Value(total_products_count)}")
    for p in range(num_products):
        penalty_prop = 0
        if p > 0 and proportion_objective_terms:
            idx = p - 1
            if idx < len(proportion_objective_terms):
                penalty_prop = solver.Value(proportion_objective_terms[idx])
        penalty_strategy = 0
        machines_start = 0
        machines_end = 0

        if p > 0:
            # Подсчитаем количество машин с продуктом p в первый и последний день.
            for m in range(num_machines):
                if (m, 0) not in cleans and solver.Value(jobs[m, 0]) == p:
                    machines_start += 1
                if (m, count_days - 1) not in cleans and solver.Value(jobs[m, count_days - 1]) == p:
                    machines_end += 1
            if strategy_penalty_terms is not None and p < len(strategy_penalty_terms):
                penalty_strategy = solver.Value(strategy_penalty_terms[p])

        diff_all += penalty_prop
        qty = solver.Value(product_counts[p])
        p_old, p_id = find_product_id_old(p)
        plan_qty = products_old[p_old][1]
        products_schedule.append(
            {
                "product_idx": p_old,
                "plan_qty": plan_qty,
                "qty": qty,
                "penalty": penalty_prop,
                "penalty_strategy": penalty_strategy,
                "machines_start": machines_start,
                "machines_end": machines_end,
            }
        )
        logger.debug(
            f"  Продукт {p_old}({p}): план={plan_qty}, факт={qty} единиц, машины {machines_start}-{machines_end}, "
            f"штраф пропорций {penalty_prop}, штраф стратегии {penalty_strategy}"
        )

    # Диагностика: ищем тройные простои (три PRODUCT_ZERO подряд на одной машине).
    PRODUCT_ZERO = 0
    for m in range(num_machines):
        m_old = find_machine_id_old(m)
        for d in range(count_days - 2):
            idx0 = m * count_days + d
            idx1 = m * count_days + d + 1
            idx2 = m * count_days + d + 2
            rec0 = schedule[idx0]
            rec1 = schedule[idx1]
            rec2 = schedule[idx2]
            if (
                rec0["product_idx"] == PRODUCT_ZERO
                and rec1["product_idx"] == PRODUCT_ZERO
                and rec2["product_idx"] == PRODUCT_ZERO
            ):
                logger.info(
                    f"triple_zero: machine={m_old}, days={d}-{d+2} (product_idx=0)"
                )


'''
    Сохраняем модель в файл для отладок 
'''
def save_plan_html(id: str, data: str) -> None:
    try:
        # Запись в файл
        with open(settings.BASE_DIR + f"/data/{id}.html", "w", encoding="utf8") as f:
            f.write(data)
    except Exception as e:
        logger.error("Ошибка записи файла html", exc_info=True)

'''
    Сохраняем модель в файл для отладок 
'''
def save_model_to_log(plan: BaseModel) -> None:
    try:
        plan_json = plan.json()
        # Запись в файл
        with open(settings.BASE_DIR + f"/log/{plan.__class__.__name__}.json", "w", encoding="utf8") as f:
            f.write(plan_json)
    except Exception as e:
        logger.error("Ошибка записи файла плана", exc_info=True)