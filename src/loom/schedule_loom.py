from ortools.sat.python import cp_model
from pydantic import BaseModel
from .model_loom import (
    DataLoomIn,
    LoomPlansOut,
    Machine,
    Product,
    Clean,
    LoomPlan,
    LoomPlansViewIn,
    LoomPlansViewOut,
    LongDayCapacity,
)
import traceback as tr
from ..config import logger, settings
import pandas as pd
from .loom_plan_html import schedule_to_html, aggregated_schedule_to_html
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
        if DataIn.apply_index_up is not None:
            settings.APPLY_INDEX_UP = DataIn.apply_index_up
        if DataIn.apply_qty_minus is not None:
            settings.APPLY_QTY_MINUS = DataIn.apply_qty_minus
        # Режим горизонта/алгоритма (FULL, LONG, LONG_SIMPLE, LONG_TWOLEVEL)
        if getattr(DataIn, "horizon_mode", None):
            settings.HORIZON_MODE = DataIn.horizon_mode.upper()

        result_calc = schedule_loom_calc(remains=remains, products=products, machines=machines, cleans=cleans,
                                    max_daily_prod_zero=max_daily_prod_zero, count_days=count_days, data=data)

        if result_calc["error_str"] == "" and result_calc["status"] != cp_model.INFEASIBLE:
            machines_view = [name for (name, product_idx,  id, type, remain_day) in machines]
            products_view = [name for (name, qty, id, machine_type, qm) in products]
            title_text = f"{result_calc['status_str']} оптимизационное значение {result_calc['objective_value']}"

            # Базовое расписание по машинам (для FULL; для LONG_ отдадим
            # только агрегированное расписание long_schedule).
            base_schedule = [
                LoomPlan(
                    machine_idx=s["machine_idx"],
                    day_idx=s["day_idx"],
                    product_idx=s["product_idx"],
                    days_in_batch=s["days_in_batch"],
                    prev_lday=s["prev_lday"],
                )
                for s in result_calc["schedule"]
            ]

            horizon_mode = getattr(settings, "HORIZON_MODE", "FULL").upper()

            # Для упрощённых режимов (LONG_SIMPLE, LONG_SIMPLE_HINT, LONG_TWOLEVEL) строим
            # агрегированное расписание по дням и продуктам: сколько машин в день под продукт.
            long_schedule: list[LongDayCapacity] | None = None
            if horizon_mode in ("LONG_SIMPLE", "LONG_SIMPLE_HINT", "LONG_TWOLEVEL"):
                counts: dict[tuple[int, int], int] = {}
                for s in result_calc["schedule"]:
                    p = s["product_idx"]
                    d = s["day_idx"]
                    # Пропускаем None/<=0 (чистки, простои и т.п.).
                    if p is None or p <= 0:
                        continue
                    key = (d, p)
                    counts[key] = counts.get(key, 0) + 1

                long_schedule = [
                    LongDayCapacity(day_idx=d, product_idx=p, machine_count=c)
                    for (d, p), c in sorted(counts.items())
                ]

            # Для LONG_SIMPLE_/LONG_TWOLEVEL schedule оставляем пустым, для остальных режимов — детальный план.
            if horizon_mode in ("LONG_SIMPLE", "LONG_SIMPLE_HINT", "LONG_TWOLEVEL"):
                schedule_out: list[LoomPlan] = []
            else:
                schedule_out = base_schedule

            # HTML: для FULL используем детальное расписание по машинам,
            # для LONG_SIMPLE_/LONG_TWOLEVEL – агрегированное представление long_schedule.
            if horizon_mode in ("LONG_SIMPLE", "LONG_SIMPLE_HINT", "LONG_TWOLEVEL"):
                res_html = aggregated_schedule_to_html(
                    machines=data["machines"],
                    schedule=result_calc["schedule"],
                    products=data["products"],
                    long_schedule=long_schedule or [],
                    dt_begin=DataIn.dt_begin,
                    title_text=title_text,
                )
            else:
                res_html = schedule_to_html(
                    machines=machines_view,
                    products=products_view,
                    schedules=result_calc["schedule"],
                    days=days,
                    dt_begin=DataIn.dt_begin,
                    title_text=title_text,
                )
            id_html = str(uuid4())

            result = LoomPlansOut(
                status=result_calc["status"],
                status_str=result_calc["status_str"],
                schedule=schedule_out,
                objective_value=result_calc["objective_value"],
                proportion_diff=result_calc["proportion_diff"],
                res_html=id_html,
                long_schedule=long_schedule,
            )
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
    # Сохраняем исходный план qty до масштабирования, чтобы использовать в метриках.
    if "qty" in products_df.columns and "qty_orig" not in products_df.columns:
        products_df["qty_orig"] = products_df["qty"].astype(int)
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

        Новая договорённость:
        - type машин и machine_type продуктов жёстко 1 или 2 (цех 1 / цех 2);
        - если machine_type в продукте > 0, то продукт можно ставить только на машины с таким же type;
        - если machine_type = 0 (старые данные) — допускаем любой тип машины.
        PRODUCT_ZERO (0) и None не проверяем.
        """
        if p_idx == 0:
            return True
        p_type = int(product_type_map.get(p_idx, 0))
        # machine_type=1 или 2 -> требуем точного совпадения с type машины.
        if p_type > 0:
            try:
                m_type = int(machines_df.at[m_idx, "type"])
            except Exception:
                m_type = 0
            return m_type == p_type
        # machine_type=0: продукт допускается на любых машинах (режим совместимости со старыми данными).
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

    # Выводим получившийся жадный план и диагностические метрики в лог-файлы.
    try:
        from pathlib import Path
        import math

        log_dir = Path("log")
        log_dir.mkdir(parents=True, exist_ok=True)

        # 1) Структура смен по машинам (как раньше).
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

        # 2) Диагностические метрики по машинам и продуктам для анализа LONG.
        metrics_path = log_dir / "greedy_metrics.log"
        cleans_set = {(c["machine_idx"], c["day_idx"]) for c in cleans}
        num_weeks = max(1, math.ceil(count_days / 21))

        # Быстрый доступ к имени/плану продукта по idx.
        name_by_idx: dict[int, str] = {}
        plan_by_idx: dict[int, int] = {}
        for _, row in products_df.iterrows():
            p_idx = int(row.get("idx", 0))
            if p_idx == 0:
                continue
            name_by_idx[p_idx] = str(row.get("name", f"p{p_idx}"))
            # Используем исходный план qty_orig, если есть, иначе текущий qty.
            if "qty_orig" in row.index and not pd.isna(row["qty_orig"]):
                plan_val = int(row["qty_orig"])
            else:
                plan_val = int(row.get("qty", 0))
            plan_by_idx[p_idx] = plan_val

        with metrics_path.open("w", encoding="utf-8") as f:
            # --- MACHINE_STATS ---
            f.write("=== MACHINE_STATS ===\n")
            for m_idx, row in enumerate(schedule):
                m_name = str(machines_df.at[m_idx, "name"]) if "name" in machines_df.columns else f"m{m_idx}"
                m_type = int(machines_df.at[m_idx, "type"]) if "type" in machines_df.columns else 0
                init_p = int(machines_df.at[m_idx, "product_idx"]) if "product_idx" in machines_df.columns else -1
                init_name = name_by_idx.get(init_p, "-") if init_p > 0 else "-"

                # Рабочие дни без чисток.
                work_days_idx: list[int] = [
                    d for d in range(count_days)
                    if (m_idx, d) not in cleans_set
                ]
                work_days_total = len(work_days_idx)

                # Распределение по продуктам на машине.
                counts_by_p: dict[int, int] = {}
                for d in work_days_idx:
                    val = row[d]
                    if isinstance(val, list):
                        val = val[0] if val else None
                    if val is None or val in (-2, 0):
                        continue
                    p_idx = int(val)
                    counts_by_p[p_idx] = counts_by_p.get(p_idx, 0) + 1

                num_products_on_machine = len([p for p, c in counts_by_p.items() if c > 0])
                dominant_p = None
                dominant_cnt = 0
                for p_idx, c in counts_by_p.items():
                    if c > dominant_cnt:
                        dominant_cnt = c
                        dominant_p = p_idx
                if work_days_total > 0 and dominant_p is not None:
                    dominant_share = dominant_cnt / work_days_total
                else:
                    dominant_share = 0.0

                dominant_name = name_by_idx.get(dominant_p, "-") if dominant_p is not None else "-"

                f.write(
                    f"m={m_idx:02d}\tname={m_name}\ttype={m_type}\tinit_idx={init_p}\tinit_name={init_name}"
                    f"\twork_days={work_days_total}\tdominant_idx={dominant_p if dominant_p is not None else -1}"
                    f"\tdominant_name={dominant_name}\tdominant_share={dominant_share:.3f}"
                    f"\tnum_products={num_products_on_machine}\n"
                )

            # --- PRODUCT_STATS ---
            f.write("\n=== PRODUCT_STATS ===\n")
            # Подсчёт фактических дней по продуктам и разбивка по неделям.
            fact_days_by_p: dict[int, int] = {p_idx: 0 for p_idx in plan_by_idx.keys()}
            fact_days_by_p_week: dict[int, list[int]] = {
                p_idx: [0 for _ in range(num_weeks)] for p_idx in plan_by_idx.keys()
            }

            for m_idx, row in enumerate(schedule):
                for d in range(count_days):
                    val = row[d]
                    if isinstance(val, list):
                        val = val[0] if val else None
                    if val is None or val in (-2, 0):
                        continue
                    p_idx = int(val)
                    if p_idx <= 0 or p_idx not in plan_by_idx:
                        continue
                    fact_days_by_p[p_idx] = fact_days_by_p.get(p_idx, 0) + 1
                    w = min(num_weeks - 1, d // 21)  # индекс недели по 21 фактической смене
                    fact_days_by_p_week[p_idx][w] += 1

            for p_idx, plan_qty in plan_by_idx.items():
                name = name_by_idx.get(p_idx, f"p{p_idx}")
                fact_days = fact_days_by_p.get(p_idx, 0)
                weeks = fact_days_by_p_week.get(p_idx, [0 for _ in range(num_weeks)])
                # Кол-во машин, на которых продукт реально стоит в greedy.
                machines_used = 0
                for m_idx, row in enumerate(schedule):
                    used_here = False
                    for d in range(count_days):
                        val = row[d]
                        if isinstance(val, list):
                            val = val[0] if val else None
                        if val == p_idx:
                            used_here = True
                            break
                    if used_here:
                        machines_used += 1

                f.write(
                    f"p_idx={p_idx}\tname={name}\tplan_qty={plan_qty}\tfact_days={fact_days}"
                    f"\tmachines_used={machines_used}\tweeks={weeks}\n"
                )
    except Exception as e:
        logger.error(f"Не удалось записать жадный план или метрики в лог-файлы: {e}")

    return schedule, objective_value, deviation_proportion, count_product_zero


def create_simple_greedy_hint(
    machines: list[tuple],
    products: list[tuple],
    count_days: int,
    product_divs: list[int],
    machine_divs: list[int],
) -> list[list[int]]:
    """Жадный план для LONG_SIMPLE (SIMPLE-модели) без нулей и чисток.

    Формирует для каждой машины m и модельного дня d продукт-idx (>=1),
    соблюдая совместимость по type/div и приблизительно порядок INDEX_UP.
    План строится по дням, без использования PRODUCT_ZERO и clean-дней.
    """
    num_machines = len(machines)
    num_products = len(products)
    schedule: list[list[int | None]] = [
        [None for _ in range(count_days)] for _ in range(num_machines)
    ]

    # Стартовые продукты на машинах
    initial_products: list[int] = []
    for (_, product_idx, _id, _t, _remain_day) in machines:
        initial_products.append(int(product_idx))

    # План по дням для каждого продукта (из qty в сменах)
    shifts_per_day = 3
    plan_days: list[int] = [0 for _ in range(num_products)]
    for p in range(1, num_products):
        try:
            qty_shifts = int(products[p][1])
        except Exception:
            qty_shifts = 0
        if qty_shifts <= 0:
            continue
        plan_days[p] = (qty_shifts + shifts_per_day - 1) // shifts_per_day

    remaining_days: list[int] = plan_days.copy()

    def can_run_product_on_machine(p_idx: int, m_idx: int) -> bool:
        if p_idx <= 0 or p_idx >= num_products:
            return False
        prod_type = products[p_idx][3]
        prod_div = product_divs[p_idx] if 0 <= p_idx < len(product_divs) else 0
        m_type = machines[m_idx][3]
        m_div = machine_divs[m_idx] if 0 <= m_idx < len(machine_divs) else 1

        # Совместимость по type
        if prod_type > 0 and m_type != prod_type:
            return False
        # Совместимость по div
        if prod_div in (1, 2) and m_div != prod_div:
            return False
        return True

    # День 0: пытаемся поставить стартовый продукт, если он планируется и совместим
    for m in range(num_machines):
        p0 = initial_products[m]
        if (
            p0 > 0
            and p0 < num_products
            and remaining_days[p0] > 0
            and can_run_product_on_machine(p0, m)
        ):
            schedule[m][0] = p0
            remaining_days[p0] -= 1

    # Основное заполнение по дням, стараемся поддерживать INDEX_UP (неубывание idx)
    for m in range(num_machines):
        for d in range(count_days):
            if schedule[m][d] is not None:
                continue
            prev_p = schedule[m][d - 1] if d > 0 else None

            chosen: int | None = None
            # 1) Пытаемся продолжать предыдущий продукт, если есть остаток плана.
            if (
                prev_p is not None
                and prev_p > 0
                and prev_p < num_products
                and can_run_product_on_machine(prev_p, m)
            ):
                if remaining_days[prev_p] > 0:
                    chosen = prev_p

            # 2) Иначе выбираем продукт с максимальным remaining_days[p]
            # среди совместимых и с idx >= prev_p (для INDEX_UP).
            if chosen is None:
                start_idx = int(prev_p) if prev_p is not None and prev_p > 0 else 1
                best_p: int | None = None
                best_rem = -1
                for p in range(start_idx, num_products):
                    if remaining_days[p] <= 0:
                        continue
                    if not can_run_product_on_machine(p, m):
                        continue
                    if remaining_days[p] > best_rem:
                        best_rem = remaining_days[p]
                        best_p = p
                if best_p is not None:
                    chosen = best_p

            # 3) Если план по дням исчерпан для всех кандидатов, но надо что-то
            # поставить (для полного заполнения), продолжаем предыдущий продукт,
            # либо берём первый совместимый продукт >= prev_p.
            if chosen is None:
                if (
                    prev_p is not None
                    and prev_p > 0
                    and prev_p < num_products
                    and can_run_product_on_machine(prev_p, m)
                ):
                    chosen = prev_p
                else:
                    start_idx = int(prev_p) if prev_p is not None and prev_p > 0 else 1
                    for p in range(start_idx, num_products):
                        if can_run_product_on_machine(p, m):
                            chosen = p
                            break

            if chosen is None:
                # В крайнем случае ничего не ставим, заполним позже.
                continue

            schedule[m][d] = chosen
            if remaining_days[chosen] > 0:
                remaining_days[chosen] -= 1

    # Заполняем возможные None на каждой машине, протягивая последнее значение.
    for m in range(num_machines):
        # Если день 0 пустой — выберем первый совместимый продукт.
        if schedule[m][0] is None:
            for p in range(1, num_products):
                if can_run_product_on_machine(p, m):
                    schedule[m][0] = p
                    break
        # Протягиваем вперёд последнее заданное значение.
        for d in range(1, count_days):
            if schedule[m][d] is None:
                schedule[m][d] = schedule[m][d - 1]

    # Логируем жадный SIMPLE-план для отладки.
    try:
        from pathlib import Path

        log_dir = Path("log")
        log_dir.mkdir(parents=True, exist_ok=True)
        hint_log_path = log_dir / "greedy_simple_hint.log"
        with hint_log_path.open("w", encoding="utf-8") as f:
            for m in range(num_machines):
                name = machines[m][0]
                init_p = initial_products[m]
                codes: list[str] = []
                for d in range(count_days):
                    p = schedule[m][d]
                    if p is None or p <= 0:
                        codes.append("--")
                    else:
                        codes.append(f"{int(p):02d}")
                f.write(
                    f"m={m:02d}\tname={name}\tinit={init_p:02d}\t" + ",".join(codes) + "\n"
                )
    except Exception:
        pass

    # Преобразуем None -> 0 не будем, hints используют только значения >=1.
    return [[int(p) if p is not None else 0 for p in row] for row in schedule]


def debug_dump_constraints_for_product_idx(
    model: cp_model.CpModel,
    internal_idx: int,
    external_idx: int | None = None,
    log_path: str = "log/simple_constraints_p_debug.log",
) -> None:
    """Отладочный дамп линейных ограничений, в которые входит продукт internal_idx.

    Выбираем все переменные, связанные с этим продуктом (count_prod_simple_*,
    prod_simple_*, C_simple_*, Cdiff_simple_*, start_seg_p*) и выписываем все
    линейные ограничения, в которых они участвуют.
    """
    try:
        from pathlib import Path

        proto = model.Proto()
        var_names: list[str] = [v.name for v in proto.variables]
        target_vars: set[int] = set()

        p = int(internal_idx)
        prefixes = [
            f"prod_simple_{p}_",      # булевы флаги продукта по машинам/дням
            f"C_simple_{p}_",         # дневные мощности C[p,d]
            f"Cdiff_simple_{p}_",     # разности C[p,d+1]-C[p,d]
            f"start_seg_p{p}_",       # начало сегмента продукта p на машине
        ]
        exact = {
            f"count_prod_simple_{p}",   # общий объём продукта p в ДНЯХ
            f"simple_is_up_{p}",        # флаг направления монотонности
        }

        for idx, name in enumerate(var_names):
            if not name:
                continue
            if name in exact:
                target_vars.add(idx)
                continue
            for pref in prefixes:
                if name.startswith(pref):
                    target_vars.add(idx)
                    break

        if not target_vars:
            return

        Path("log").mkdir(parents=True, exist_ok=True)
        if external_idx is None:
            out_path = Path(log_path)
        else:
            out_path = Path(f"log/simple_constraints_p{external_idx}.log")

        with out_path.open("w", encoding="utf-8") as f:
            f.write(
                f"# Debug dump for product internal_idx={internal_idx}, external_idx={external_idx}\n"
            )
            f.write(f"# Target var indices: {sorted(target_vars)}\n\n")

            for ci, ct in enumerate(proto.constraints):
                if not ct.WhichOneof("constraint") == "linear":
                    continue
                lin = ct.linear
                if not lin.vars:
                    continue
                if not any(v in target_vars for v in lin.vars):
                    continue

                dom = list(lin.domain)
                f.write(f"Constraint {ci}: domain={dom}\n")
                for v_idx, coeff in zip(lin.vars, lin.coeffs):
                    name = var_names[v_idx]
                    f.write(f"  {coeff} * {name}\n")
                f.write("\n")
    except Exception as e:
        try:
            logger.error(f"debug_dump_constraints_for_product_idx failed: {e}")
        except Exception:
            pass


def simple_qtyminus_precheck_long_simple(data: dict) -> set[int]:
    """Pre-check для LONG_SIMPLE перед построением модели с APPLY_QTY_MINUS.

    Идея:
      - Рассматриваем строгие продукты (qty_minus=0, qty>0).
      - По каждому div считаем агрегированную нижнюю границу по дням
        (plan_days и/или min_days) и сравниваем с мощностью по div
        (число машин * simple_days).
      - Если по какому-то div нижняя граница превышает мощность, понижаем
        жёсткость ("расслабляем") самые большие по plan_days строгие продукты
        без H2/H3 до тех пор, пока агрегированное неравенство не выполнится.

    Возвращает множество idx строгих продуктов, для которых остаётся
    смысл включать жёсткие нижние границы qty_minus в модели.
    """
    import math

    products_json = data["products"]
    machines_json = data["machines"]

    orig_days = int(data["count_days"])
    shifts_per_day = 3
    simple_days = (orig_days + shifts_per_day - 1) // shifts_per_day

    # Собираем метаданные по продуктам.
    plan_days: dict[int, int] = {}
    min_days: dict[int, int] = {}
    prod_div: dict[int, int] = {}
    qty_minus_flag: dict[int, int] = {}
    qty_shifts: dict[int, int] = {}

    for p in products_json:
        try:
            idx = int(p["idx"])
        except Exception:
            continue
        if idx == 0:
            continue
        qty = int(p.get("qty", 0) or 0)
        qty_shifts[idx] = qty
        qm = int(p.get("qty_minus", 0) or 0)
        qty_minus_flag[idx] = qm
        qmm_shifts = int(p.get("qty_minus_min", 0) or 0)
        pd = (qty + shifts_per_day - 1) // shifts_per_day if qty > 0 else 0
        md = (qmm_shifts + shifts_per_day - 1) // shifts_per_day if qmm_shifts > 0 else 0
        plan_days[idx] = pd
        min_days[idx] = md
        prod_div[idx] = int(p.get("div", 0) or 0)

    # Строгие продукты: qty_minus == 0 и qty>0
    strict_idxs = [
        idx
        for idx, qty in qty_shifts.items()
        if idx != 0 and qty > 0 and qty_minus_flag.get(idx, 0) == 0
    ]

    if not strict_idxs:
        return set()

    # Оценим H2/H3 для разделения на "жёсткие" и "мягкие" строгие.
    # Для pre-check берём тот же подход, что в estimate_capacity_for_product.
    from tools.analyze_simple_subset_capacity import estimate_capacity_for_product
    from tools.compare_long_vs_simple import load_input

    # Построим tuple-версии products/machines, как в schedule_loom_calc/estimate_capacity_for_product.
    # data здесь уже после load_input, поэтому machines/products должны быть кортежами.
    # На всякий случай восстановим tuples из JSON для совместимости.
    import pandas as pd

    machines_df = pd.DataFrame(data["machines"])
    products_df = pd.DataFrame(data["products"])

    machines_tuples = [
        (row["name"], row["product_idx"], row["id"], row["type"], row["remain_day"], row.get("reserve", 0))
        for _, row in machines_df.iterrows()
    ]
    products_tuples = [
        (
            row["name"],
            row["qty"],
            row["id"],
            row["machine_type"],
            row["qty_minus"],
            row["lday"],
            row.get("src_root", -1),
            row.get("qty_minus_min", 0),
            row.get("sr", False),
            row.get("strategy", "--"),
        )
        for _, row in products_df.iterrows()
    ]

    # Классификация H2/H3
    hard_strict: set[int] = set()
    soft_strict: set[int] = set()

    num_days = orig_days
    for idx in strict_idxs:
        # estimate_capacity_for_product ожидает индекс по tuple-списку.
        if idx < 0 or idx >= len(products_tuples):
            soft_strict.add(idx)
            continue
        cap_total, per_m_cap, extra = estimate_capacity_for_product(
            idx, products_tuples, machines_tuples, num_days
        )
        if extra.get("h2_active") or extra.get("h3_active"):
            hard_strict.add(idx)
        else:
            soft_strict.add(idx)

    strict_set = set(strict_idxs)

    # Агрегированная мощность по div.
    cap_by_div: dict[int, int] = {}
    for m in machines_json:
        m_div = int(m.get("div", 1) or 1)
        cap_by_div[m_div] = cap_by_div.get(m_div, 0) + simple_days

    # Нижние границы по дням для текущего strict_set.
    def compute_lb(strict_subset: set[int]) -> dict[int, int]:
        lb: dict[int, int] = {}
        for idx in strict_subset:
            if idx == 0:
                continue
            # Для строгих qty_minus=0 нижняя граница ~ plan_days.
            pd = plan_days.get(idx, 0)
            md = min_days.get(idx, 0)
            lower = max(pd, md)
            d = prod_div.get(idx, 0)
            if d in (1, 2):
                lb[d] = lb.get(d, 0) + lower
        return lb

    lb_by_div = compute_lb(strict_set)

    # Логируем начальную ситуацию.
    from src.config import logger

    logger.info("SIMPLE precheck: strict_idxs=%s", strict_idxs)
    logger.info("SIMPLE precheck: cap_by_div=%s, lb_by_div_initial=%s", cap_by_div, lb_by_div)

    # Функция, возвращающая div, где есть явный дефицит.
    def find_deficit_divs() -> dict[int, int]:
        deficits: dict[int, int] = {}
        for d, cap in cap_by_div.items():
            lb = lb_by_div.get(d, 0)
            if lb > cap:
                deficits[d] = lb - cap
        return deficits

    deficits = find_deficit_divs()
    if not deficits:
        logger.info("SIMPLE precheck: no aggregate deficit by div; keeping all strict products")
        return strict_set

    logger.info("SIMPLE precheck: initial deficits=%s", deficits)

    # Пытаемся устранить дефициты, снимая qty_minus=0 с самых больших soft_strict продуктов по каждому проблемному div.
    # По умолчанию hard_strict (H2/H3) не трогаем.

    # Сгруппируем soft_strict по div.
    soft_by_div: dict[int, list[int]] = {}
    for idx in soft_strict:
        d = prod_div.get(idx, 0)
        if d in (1, 2):
            soft_by_div.setdefault(d, []).append(idx)

    # Для стабильности сортируем по убыванию plan_days (сначала самые большие).
    for d in soft_by_div:
        soft_by_div[d].sort(key=lambda p: plan_days.get(p, 0), reverse=True)

    relaxed: set[int] = set()

    # Итеративно снимаем soft_strict, пока дефицит не исчезнет или кандидаты не кончатся.
    changed = True
    while changed:
        changed = False
        deficits = find_deficit_divs()
        if not deficits:
            break
        logger.info("SIMPLE precheck: current deficits=%s", deficits)
        for d, deficit in list(deficits.items()):
            cand_list = soft_by_div.get(d, [])
            while cand_list and lb_by_div.get(d, 0) > cap_by_div.get(d, 0):
                p = cand_list.pop(0)
                if p not in strict_set:
                    continue
                # Снимаем p из strict_set: больше не считаем его жёстким qty_minus=0.
                strict_set.remove(p)
                relaxed.add(p)
                lower = max(plan_days.get(p, 0), min_days.get(p, 0))
                lb_by_div[d] = lb_by_div.get(d, 0) - lower
                logger.info(
                    "SIMPLE precheck: relax strict qty_minus=0 for p=%d (lower=%d) in div=%d; new lb_div=%d",
                    p,
                    lower,
                    d,
                    lb_by_div[d],
                )
                changed = True
        # цикл продолжится, пока что-то меняется

    final_deficits = find_deficit_divs()
    if final_deficits:
        logger.warning("SIMPLE precheck: residual aggregate deficits after relaxation: %s", final_deficits)
    logger.info(
        "SIMPLE precheck: final strict_set=%s, relaxed_strict=%s",
        sorted(strict_set),
        sorted(relaxed),
    )

    return strict_set


def schedule_loom_calc(remains: list, products: list, machines: list, cleans: list, max_daily_prod_zero: int,
                       count_days: int, data: dict) -> LoomPlansOut:

    def MachinesDFToArray(machines_in: pd.DataFrame) -> list[(str, int, str, int, int)]:
        """Преобразуем DataFrame машин в кортежи для расчёта.

        Сейчас в кортеже храним только поля, которые реально используются в
        CP-моделях: (name, product_idx, id, type, remain_day). Цех (div)
        передаётся отдельно в виде массива machine_divs, чтобы не ломать
        существующие распаковки кортежей.
        """
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

        Базовая структура кортежа (без div):
        (name, qty, id, machine_type, qty_minus, lday, src_root,
         qty_minus_min, sr, strategy)

        Поле div (цех) передаём отдельно массивом product_divs, чтобы не
        менять позиционную индексацию по существующему коду.
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

    # Режим горизонта: запоминаем исходное количество смен и режим LONG.
    long_mode = getattr(settings, "HORIZON_MODE", "FULL").upper() == "LONG"
    orig_count_days = count_days

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
    if settings.USE_GREEDY_HINT and getattr(settings, "HORIZON_MODE", "FULL").upper() != "LONG_SIMPLE":
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

    # Валидация входных данных: запрещаем дубликаты продуктов по id/idx (кроме idx=0)
    # и некорректные значения lday.

    # 1) Проверка на дубликаты id (кроме пустых id).
    dup_id_mask = (
        products_df["id"].notna()
        & (products_df["id"] != "")
        & products_df.duplicated(subset=["id"], keep=False)
    )
    dup_ids_df = products_df[dup_id_mask]

    # 2) Проверка на дубликаты idx (кроме служебного idx=0).
    dup_idx_mask = (
        (products_df["idx"] != 0)
        & products_df.duplicated(subset=["idx"], keep=False)
    )
    dup_idx_df = products_df[dup_idx_mask]

    dup_messages: list[str] = []
    if not dup_ids_df.empty:
        for _, row in dup_ids_df.iterrows():
            dup_messages.append(
                f"id={row.get('id')}, idx={row.get('idx')}, name='{row.get('name')}', qty={row.get('qty')}"
            )
    if not dup_idx_df.empty:
        for _, row in dup_idx_df.iterrows():
            dup_messages.append(
                f"DUP_IDX idx={row.get('idx')}, id={row.get('id')}, name='{row.get('name')}', qty={row.get('qty')}"
            )

    if dup_messages:
        details = "; ".join(dup_messages)
        error_msg = f"Некорректные данные: обнаружены дубликаты продуктов (id/idx): {details}"
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

    # 3) Валидация lday: для продуктов (кроме нулевого) lday не должен быть 0 или отрицательным.
    invalid_products = []
    for p in products:
        # products: (name, qty, id, machine_type, qty_minus, lday, src_root, ...)
        name, qty, pid, machine_type, qty_minus, lday_val = p[:6]
        # p_idx нам не нужен для логики, но полезен для сообщения об ошибке — возьмём его из исходного списка.
        p_idx = products.index(p)
        # Проверяем только "живые" продукты с qty>0; для служебных/нулевых
        # позиций (qty=0) lday может быть 0.
        if p_idx > 0 and qty > 0 and lday_val <= 0:
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

    # В режиме LONG_SIMPLE интерпретируем count_days как количество КАЛЕНДАРНЫХ
    # дней, где каждые 3 смены исходного горизонта образуют один день.
    # Преобразуем day_idx в cleans из смен в дни и сокращаем горизонт.
    horizon_mode_local = getattr(settings, "HORIZON_MODE", "FULL").upper()
    if horizon_mode_local in ("LONG_SIMPLE", "LONG_TWOLEVEL"):
        shifts_per_day = 3  # 84 смены / 3 = 28 календарных дней для SIMPLE/TWOLEVEL
        count_days_days = (count_days + shifts_per_day - 1) // shifts_per_day
        if not clean_df.empty:
            clean_df = clean_df.copy()
            clean_df["day_idx"] = (clean_df["day_idx"] // shifts_per_day).astype(int)
            clean_df = clean_df.drop_duplicates(subset=["machine_idx", "day_idx"]).reset_index(drop=True)
        count_days = count_days_days

    # Для LONG-режима агрегируем горизонт: 2 реальные смены = 1 модельный день.
    if long_mode and False:
        step = 2
        count_days_long = (orig_count_days + step - 1) // step

        cleans_agg_set: set[tuple[int, int]] = set()
        for c in data["cleans"]:
            try:
                m = int(c["machine_idx"])
                d = int(c["day_idx"])
            except (KeyError, TypeError, ValueError):
                continue
            d2 = d // step
            if d2 < 0 or d2 >= count_days_long:
                continue
            cleans_agg_set.add((m, d2))

        clean_df = pd.DataFrame(
            [{"machine_idx": m, "day_idx": d2} for (m, d2) in sorted(cleans_agg_set)]
        )

        # Обновляем количество дней для построения LONG-модели.
        count_days = count_days_long

    product_id = products_df["id"]
    machines_df["product_id"] = machines_df["product_idx"].map(product_id)
    # Приводим типы машин к типам продуктов для старых данных:
    # если type=0, а у стартового продукта machine_type > 0,
    # то считаем, что машина принадлежит этому цеху (1 или 2).
    product_type = products_df["machine_type"]
    mapped_machine_types = machines_df["product_idx"].map(product_type)
    condition = (machines_df["type"] == 0) & (mapped_machine_types > 0)
    machines_df.loc[condition, "type"] = mapped_machine_types[condition]
    product_zero = products_df[products_df["idx"] == 0]

    # Пересортируем продукты один раз на весь период, без разбивки по неделям.
    # Дедубликацию по id не выполняем: при дубликатах входные данные считаем
    # некорректными (см. проверку выше) и сразу возвращаем ошибку, чтобы не
    # "угадывать" правильную запись.
    products_df_new = products_df.tail(-1)

    # Убираем из products_df_new продукты с qty=0, которые нигде не стоят на
    # машинах на начало, по id. Это безопасно: такие строки не могут появиться
    # в расписании и только раздувают модель.
    products_df_zero = products_df[products_df["qty"] == 0]
    for _, p in products_df_zero.iterrows():
        pid = p["id"]
        if len(machines_df[machines_df["product_id"] == pid]) == 0:
            products_df_new = products_df_new[products_df_new["id"] != pid]

    # Теперь сортируем по qty для LONG_SIMPLE-эвристик и добавляем нулевой
    # продукт (idx=0) в начало.
    if not products_df_new.empty:
        products_df_new = products_df_new.sort_values(by=["qty"])
    products_df_new = pd.concat([product_zero, products_df_new]).reset_index(drop=True)

    products_df_new["idx"] = range(len(products_df_new))
    id_to_new_idx_map = products_df_new.set_index("id")["idx"].to_dict()
    machines_df["product_idx"] = machines_df["product_id"].map(id_to_new_idx_map)

    # Для режима LONG_SIMPLE внутренние индексы продуктов (0..num_products_new-1)
    # могут отличаться от исходных idx в JSON. Для отладочных флагов, которые
    # ссылаются на продукты по исходному индексу (SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX,
    # SIMPLE_DEBUG_PRODUCT_UPPER_CAPS), выполним переотображение idx_orig -> idx_internal
    # через поле id продукта.
    horizon_mode_local = getattr(settings, "HORIZON_MODE", "FULL").upper()
    if horizon_mode_local in ("LONG_SIMPLE", "LONG_SIMPLE_HINT"):
        # Строим карту: исходный idx -> id
        orig_idx_to_id: dict[int, str] = {}
        for _, row in products_df.iterrows():
            try:
                orig_idx = int(row["idx"])
                pid = row["id"]
            except Exception:
                continue
            orig_idx_to_id[orig_idx] = pid

        # Обратная карта: id -> внутренний idx (уже есть в id_to_new_idx_map).
        def orig_to_internal_idx(orig_idx: int) -> int | None:
            pid = orig_idx_to_id.get(orig_idx)
            if pid is None:
                return None
            return int(id_to_new_idx_map.get(pid)) if pid in id_to_new_idx_map else None

        # Переотображаем SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX, если задан.
        dbg_max = getattr(settings, "SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX", None)
        if dbg_max is not None:
            try:
                dbg_max_int = int(dbg_max)
            except Exception:
                dbg_max_int = None
            if dbg_max_int is not None:
                internal = orig_to_internal_idx(dbg_max_int)
                if internal is not None:
                    settings.SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX = internal

        # Переотображаем SIMPLE_DEBUG_PRODUCT_UPPER_CAPS: ключи – исходные idx.
        dbg_caps = getattr(settings, "SIMPLE_DEBUG_PRODUCT_UPPER_CAPS", None)
        if dbg_caps is not None:
            new_caps: dict[int, int] = {}
            for orig_idx, cap in dbg_caps.items():
                try:
                    orig_idx_int = int(orig_idx)
                except Exception:
                    continue
                internal = orig_to_internal_idx(orig_idx_int)
                if internal is None:
                    continue
                new_caps[internal] = int(cap)
            settings.SIMPLE_DEBUG_PRODUCT_UPPER_CAPS = new_caps

        # Переотображаем SIMPLE_QTY_MINUS_SUBSET из исходных idx в внутренние idx.
        dbg_subset = getattr(settings, "SIMPLE_QTY_MINUS_SUBSET", None)
        if dbg_subset is not None:
            new_subset: set[int] = set()
            try:
                iterable = list(dbg_subset)
            except TypeError:
                iterable = []
            for orig_idx in iterable:
                try:
                    orig_idx_int = int(orig_idx)
                except Exception:
                    continue
                internal = orig_to_internal_idx(orig_idx_int)
                if internal is None:
                    continue
                new_subset.add(internal)
            settings.SIMPLE_QTY_MINUS_SUBSET = new_subset if new_subset else None

        # Переотображаем SIMPLE_DEBUG_DUMP_CONSTRAINTС_FOR_IDX (если задан)
        # из исходного idx во внутренний idx, чтобы далее можно было сделать
        # дамп ограничений по конкретному продукту.
        dbg_dump_idx = getattr(settings, "SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX", None)
        if dbg_dump_idx is not None:
            try:
                dbg_dump_ext = int(dbg_dump_idx)
            except Exception:
                dbg_dump_ext = None
            if dbg_dump_ext is not None:
                internal = orig_to_internal_idx(dbg_dump_ext)
                if internal is not None:
                    settings.SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX_INTERNAL = internal
                else:
                    settings.SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX_INTERNAL = None

    # Цеха по машинам/продуктам: div=1 или 2 – фиксированный цех, 0 –
    # "плавающий" продукт (можно в любом цехе), None/отсутствие – игнорируем.
    if "div" in products_df_new.columns:
        product_divs = (
            products_df_new["div"].fillna(0).astype(int).tolist()
        )
    else:
        product_divs = [0 for _ in range(len(products_df_new))]

    if "div" in machines_df.columns:
        machine_divs = machines_df["div"].fillna(1).astype(int).tolist()
    else:
        machine_divs = [1 for _ in range(len(machines_df))]

    products_new = ProductsDFToArray(products_df_new)
    machines_new = MachinesDFToArray(machines_df)
    cleans_new = CleansDFToArray(clean_df)

    horizon_mode = getattr(settings, "HORIZON_MODE", "FULL").upper()
    # long_mode используется только для отключения детальной отладки/переходов;
    # считаем "долгими" упрощённые режимы.
    long_mode = horizon_mode in ("LONG_SIMPLE", "LONG_SIMPLE_HINT", "LONG_TWOLEVEL")

    # Для LONG_SIMPLE / LONG_SIMPLE_HINT при включённом USE_GREEDY_HINT или в режиме
    # LONG_SIMPLE_HINT строим жадный SIMPLE-план (без нулей и чисток).
    simple_hint_mode = (
        horizon_mode == "LONG_SIMPLE_HINT"
        or (settings.USE_GREEDY_HINT and horizon_mode == "LONG_SIMPLE")
    )
    if simple_hint_mode:
        try:
            greedy_schedule = create_simple_greedy_hint(
                machines_new,
                products_new,
                count_days,
                product_divs,
                machine_divs,
            )
            logger.info("Greedy SIMPLE hint computed for %s", horizon_mode)
        except Exception as e:
            logger.error(f"Ошибка при вычислении SIMPLE greedy hint: {e}")
            greedy_schedule = None

    if horizon_mode in ("LONG_SIMPLE", "LONG_SIMPLE_HINT"):
        # Pre-check для строгих qty_minus=0 в LONG_SIMPLE: если APPLY_QTY_MINUS включён
        # и внешний код не задал SIMPLE_QTY_MINUS_SUBSET явно, попробуем заранее
        # ослабить часть строгих продуктов, чтобы агрегированные нижние границы
        # по дням не превышали мощность по цехам.
        if settings.APPLY_QTY_MINUS and getattr(settings, "SIMPLE_QTY_MINUS_SUBSET", None) is None:
            strict_final = simple_qtyminus_precheck_long_simple(data)
            if strict_final:
                # Используем strict_final как SIMPLE_QTY_MINUS_SUBSET, чтобы
                # qty_minus-блок применялся только к этому подмножеству.
                settings.SIMPLE_QTY_MINUS_SUBSET = set(strict_final)
        (model, jobs, product_counts, proportion_objective_terms, total_products_count, prev_lday, start_batch,
         batch_end_complite, days_in_batch, completed_transition, pred_start_batch, same_as_prev,
         strategy_penalty_terms) = create_model_simple(
            remains=remains, products=products_new, machines=machines_new, cleans=cleans_new,
            max_daily_prod_zero=max_daily_prod_zero, count_days=count_days,
            dedicated_machines=dedicated_machines,
            product_divs=product_divs,
            machine_divs=machine_divs,
        )

        # Отладочный дамп ограничений для конкретного продукта (если включён).
        dbg_ext = getattr(settings, "SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX", None)
        dbg_int = getattr(settings, "SIMPLE_DEBUG_DUMP_CONSTRAINTS_FOR_IDX_INTERNAL", None)
        if dbg_ext is not None and dbg_int is not None:
            try:
                debug_dump_constraints_for_product_idx(
                    model,
                    internal_idx=int(dbg_int),
                    external_idx=int(dbg_ext),
                )
            except Exception as e:
                logger.error(f"Failed to dump constraints for product idx={dbg_ext}: {e}")
    elif horizon_mode == "LONG_TWOLEVEL":
        from .two_level_simple import build_twolevel_schedule

        schedule = []
        products_schedule = []
        diff_all = 0

        schedule, products_schedule, internal_obj, external_penalty = build_twolevel_schedule(
            remains=remains,
            products_new=products_new,
            machines_new=machines_new,
            cleans_new=cleans_new,
            count_days=count_days,
            products_df_orig=products_df,
            machines_orig=machines,
            data=data,
        )

        result = {
            "status": int(cp_model.OPTIMAL),
            "status_str": "FEASIBLE_TWOLEVEL",
            "schedule": schedule,
            "products": products_schedule,
            "objective_value": int(internal_obj),
            "proportion_diff": int(external_penalty),
            "error_str": "",
        }
        return result
    else:
        (model, jobs, product_counts, proportion_objective_terms, total_products_count, prev_lday, start_batch,
         batch_end_complite, days_in_batch, completed_transition, pred_start_batch, same_as_prev,
         strategy_penalty_terms) = create_model(
            remains=remains, products=products_new, machines=machines_new, cleans=cleans_new,
            max_daily_prod_zero=max_daily_prod_zero, count_days=count_days,
            dedicated_machines=dedicated_machines,
            product_divs=product_divs,
            machine_divs=machine_divs,
        )

    # Если есть жадный план и включён USE_GREEDY_HINT/режим LONG_SIMPLE_HINT,
    # используем его как hint для jobs[m,d]. Для упрощённых режимов (LONG_SIMPLE_)
    # hints безопасны; для FULL/SHORT мы по-прежнему можем использовать shift-based greedy.
    if greedy_schedule is not None and not long_mode:
        try:
            simple_mode = horizon_mode in ("LONG_SIMPLE", "LONG_SIMPLE_HINT")
            if not simple_mode:
                # FULL/SHORT: используем greedy_schedule_init и переотображение id -> новый idx.
                orig_idx_to_id = products_df.set_index("idx")["id"].to_dict()
                id_to_new_idx = products_df_new.set_index("id")["idx"].to_dict()

            for (m, d), var in jobs.items():
                # jobs есть только для рабочих дней (без чисток)
                if m >= len(greedy_schedule) or d >= len(greedy_schedule[m]):
                    continue
                p_val = greedy_schedule[m][d]
                if p_val is None:
                    continue

                if simple_mode:
                    # В SIMPLE hints уже заданы во внутреннем пространстве idx (1..num_products-1).
                    hint_idx = int(p_val)
                    if hint_idx < 1 or hint_idx >= len(products_new):
                        continue
                else:
                    orig_p = p_val
                    if orig_p in (-2,):
                        # чистка — hint не ставим
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

    # Создаём solver после построения модели.
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True

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
                      dedicated_machines: list[int] | None = None,
                      product_divs: list[int] | None = None,
                      machine_divs: list[int] | None = None):
    """Упрощённая модель для длинного горизонта.

    Новая схема: 1 модельная смена соответствует 2 фактическим сменам.
    Переход (две фактические смены простоя 0,0) моделируется одной
    модельной сменой с PRODUCT_ZERO без чисток.

    - lday и remain_day не учитываются.
    - Строятся только jobs[m,d], product_counts[p], ограничения по нулям,
      типам машин, упрощённая пропорциональная цель и стратегии.
    """
    # Реальное количество фактических смен (из входных данных)
    num_days_real = count_days
    # Количество агрегированных модельных смен (по 2 фактические на одну)
    num_days = (num_days_real + 1) // 2

    num_machines = len(machines)
    num_products = len(products)

    # Нормализуем div по продуктам/машинам: product_divs[p] = 1/2 — фиксированный цех,
    # 0 — можно в любом (но только в одном), иначе/отсутствует — игнорируем.
    if product_divs is None:
        product_divs = [0 for _ in range(num_products)]
    if machine_divs is None:
        machine_divs = [1 for _ in range(num_machines)]

    dedicated_set = set(dedicated_machines or [])

    all_machines = range(num_machines)
    all_days = range(num_days)
    all_products = range(num_products)

    PRODUCT_ZERO = 0

    # Множество фактических чисток для построения флага has_clean[m,d]
    cleans_set = {(m_idx, d_idx) for (m_idx, d_idx) in cleans}

    # has_clean[m,d] = True, если в одной из двух фактических смен,
    # соответствующих модельной смене d, есть чистка.
    has_clean: dict[tuple[int, int], bool] = {}
    for m in all_machines:
        for d in all_days:
            day1 = 2 * d
            day2 = 2 * d + 1
            flag = False
            if (m, day1) in cleans_set:
                flag = True
            if day2 < num_days_real and (m, day2) in cleans_set:
                flag = True
            has_clean[m, d] = flag

    model = cp_model.CpModel()

    # Рабочие дни и переменные jobs[m,d] (все модельные дни считаются рабочими).
    jobs: dict[tuple[int, int], cp_model.IntVar] = {}
    work_days: list[tuple[int, int]] = []

    for m in all_machines:
        for d in all_days:
            work_days.append((m, d))
            jobs[(m, d)] = model.NewIntVar(0, num_products - 1, f"job_{m}_{d}")
            if m in dedicated_set:
                # fixed_product = product_idx из tuples machines: (name, product_idx, id, type, remain_day)
                fixed_product = machines[m][1]
                model.Add(jobs[(m, d)] == fixed_product)

    logger.debug(
        f"create_model_long: num_machines={num_machines}, num_days_model={num_days}, "
        f"num_days_real={num_days_real}, work_days={len(work_days)}, max_daily_prod_zero={max_daily_prod_zero}"
    )

    # Булевы переменные "производится продукт p на (m,d)".
    product_produced_bools: dict[tuple[int, int, int], cp_model.BoolVar] = {}
    for p in all_products:
        for m, d in work_days:
            b = model.NewBoolVar(f"prod_{p}_{m}_{d}")
            product_produced_bools[p, m, d] = b
            model.Add(jobs[m, d] == p).OnlyEnforceIf(b)
            model.Add(jobs[m, d] != p).OnlyEnforceIf(b.Not())

    # Подсчёт количества смен каждого продукта (в агрегированных модельных днях).
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
                daily_zero.append(product_produced_bools[PRODUCT_ZERO, m, d])
            model.Add(sum(daily_zero) <= max_daily_prod_zero)

    if settings.APPLY_DOWNTIME_LIMITS and settings.APPLY_ZERO_PER_MACHINE_LIMIT:
        # weeks по-прежнему считаем по фактическим сменам
        weeks = max(1, num_days_real // 21)
        max_zero_per_machine = 2 * weeks
        for m in all_machines:
            zeros_m = []
            for d in all_days:
                zeros_m.append(product_produced_bools[PRODUCT_ZERO, m, d])
            model.Add(sum(zeros_m) <= max_zero_per_machine)

    # Ограничения по типам машин и по цехам (div).
    for p in all_products:
        prod_type = products[p][3]
        prod_div = product_divs[p] if 0 <= p < len(product_divs) else 0

        for m in all_machines:
            m_type = machines[m][3]
            m_div = machine_divs[m] if 0 <= m < len(machine_divs) else 1

            # Тип: если machine_type продукта > 0 (1 или 2), то только на машинах того же type.
            type_incompatible = (prod_type > 0 and m_type != prod_type)
            # Цех: если div продукта = 1/2, то только на машинах этого цеха.
            div_incompatible = (prod_div in (1, 2) and m_div != prod_div)

            if type_incompatible or div_incompatible:
                for d in all_days:
                    model.Add(jobs[m, d] != p)

    # Вариант B для продуктов с div = 0: выбор единственного цеха в модели.
    # Для каждого такого продукта p вводим y[p,1], y[p,2] с y[p,1] + y[p,2] = 1,
    # и требуем: если p стоит на машине m (div=m_div), то y[p,m_div] = 1.
    flex_products = [
        p for p in all_products
        if (0 <= p < len(product_divs) and product_divs[p] == 0 and p != 0)
    ]

    y_div: dict[tuple[int, int], cp_model.BoolVar] = {}
    for p in flex_products:
        y1 = model.NewBoolVar(f"y_div1_long_p{p}")
        y2 = model.NewBoolVar(f"y_div2_long_p{p}")
        model.Add(y1 + y2 == 1)
        y_div[p, 1] = y1
        y_div[p, 2] = y2

    for p in flex_products:
        for m in all_machines:
            m_div = machine_divs[m] if 0 <= m < len(machine_divs) else 1
            if m_div not in (1, 2):
                # Неподдерживаемый div – на всякий случай запрещаем p на этой машине.
                for d in all_days:
                    model.Add(jobs[m, d] != p)
                continue
            y_pm = y_div[p, m_div]
            for d in all_days:
                b = product_produced_bools[p, m, d]
                model.AddImplication(b, y_pm)

    # --------- Переходы и clean-дни в агрегированном LONG-режиме ---------

    # Для удобства: список рабочих модельных дней по каждой машине.
    work_days_by_machine: dict[int, list[int]] = {m: [] for m in all_machines}
    for m, d in work_days:
        work_days_by_machine[m].append(d)

    # Флаги is_zero[m,d] для упрощения логики.
    is_zero: dict[tuple[int, int], cp_model.BoolVar] = {}
    for m in all_machines:
        for d in work_days_by_machine[m]:
            b = model.NewBoolVar(f"is_zero_long_{m}_{d}")
            is_zero[m, d] = b
            model.Add(jobs[m, d] == PRODUCT_ZERO).OnlyEnforceIf(b)
            model.Add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(b.Not())

    # На агрегированных днях с чистками запрещаем чистый простой (PRODUCT_ZERO),
    # чтобы переходный день не содержал clean.
    for m in all_machines:
        for d in work_days_by_machine[m]:
            if has_clean[m, d]:
                model.Add(jobs[m, d] != PRODUCT_ZERO)

    same_as_prev: dict[tuple[int, int], cp_model.BoolVar] = {}

    if settings.APPLY_TRANSITION_BUSINESS_LOGIC:
        for m in all_machines:
            wds = work_days_by_machine[m]
            for idx in range(1, len(wds)):
                d = wds[idx]
                prev_d = wds[idx - 1]

                # Булевы флаги для текущего и предыдущего дня.
                same = model.NewBoolVar(f"same_as_prev_long_{m}_{d}")
                same_as_prev[m, d] = same
                model.Add(jobs[m, d] == jobs[m, prev_d]).OnlyEnforceIf(same)
                model.Add(jobs[m, d] != jobs[m, prev_d]).OnlyEnforceIf(same.Not())

                is_zero_prev = is_zero[m, prev_d]
                is_zero_cur = is_zero[m, d]

                # Основное правило смены продукта:
                # - если текущий день ненулевой и продукт отличается от предыдущего,
                #   то предыдущий день должен быть нулевым БЕЗ clean.
                if has_clean[m, prev_d]:
                    # День с clean не может быть переходным: смена только через тот же продукт или ноль.
                    model.AddBoolOr([
                        is_zero_cur,  # текущий день — ноль
                        same,         # или тот же продукт, что и на предыдущем дне
                    ])
                else:
                    # Разрешаем смену продукта через нулевой день без clean.
                    model.AddBoolOr([
                        is_zero_cur,   # текущий день — ноль
                        same,          # или тот же продукт
                        is_zero_prev,  # или на предыдущем дне был чистый ноль без clean
                    ])

                # Запрет двух нулей подряд в агрегированных днях:
                # один нулевой день уже моделирует (0,0) в фактическом времени.
                model.Add(is_zero_prev + is_zero_cur <= 1)

    # Упрощённое ограничение INDEX_UP в LONG:
    # при включённом APPLY_INDEX_UP требуем неубывающий индекс продукта по агрегированным дням.
    if settings.APPLY_INDEX_UP:
        for m in all_machines:
            wds = work_days_by_machine[m]
            for idx in range(1, len(wds)):
                d = wds[idx]
                prev_d = wds[idx - 1]
                model.Add(jobs[m, d] >= jobs[m, prev_d])

    # Пропорциональная цель и стратегии (по аналогии с create_model, но без lday).
    proportions_input = [p[1] for p in products]

    total_products_count = model.NewIntVar(0, num_machines * num_days, "total_products_count")
    model.Add(total_products_count == sum(product_counts[p] for p in range(1, num_products)))

    # В LONG-режиме используем облегчённый PROP:
    # - для qty_minus=0: верхние ограничения (fact_real <= plan + 6);
    # - для qty_minus=1: линейный штраф за отклонение от агрегированного плана.
    proportion_objective_terms: list[cp_model.IntVar] = []
    total_input_quantity = 0
    total_input_max = 0

    max_count = num_machines * num_days

    # Два варианта цели:
    # 1) APPLY_OVERPENALTY_INSTEAD_OF_PROP=True: линейный штраф за превышение плана
    #    в агрегированных модельных днях (без пропорций).
    # 2) Иначе – исходная облегчённая пропорциональная логика LONG (односторонний shortfall).

    if settings.APPLY_PROP_OBJECTIVE and settings.APPLY_OVERPENALTY_INSTEAD_OF_PROP:
        for p in range(1, num_products):
            planned_qty = int(proportions_input[p])  # план в сменах
            if planned_qty <= 0:
                continue

            qty_minus_flag = products[p][4] if len(products[p]) > 4 else 0

            # Перевод плана в агрегированные модельные дни (2 смены = 1 день).
            plan_model_days = (planned_qty + 1) // 2  # ceil(plan/2)

            if qty_minus_flag == 0:
                # Для строгих продуктов запрещаем перепроизводство.
                model.Add(product_counts[p] <= plan_model_days)
                continue

            # over = max(0, fact - plan_model_days)
            diff = model.NewIntVar(-max_count, max_count, f"over_long_diff_{p}")
            model.Add(diff == product_counts[p] - plan_model_days)
            zero = model.NewConstant(0)
            over = model.NewIntVar(0, max_count, f"over_long_{p}")
            model.AddMaxEquality(over, [diff, zero])

            proportion_objective_terms.append(over)

    elif settings.APPLY_PROP_OBJECTIVE:
        total_input_quantity = sum(proportions_input)
        total_input_max = max(proportions_input) if proportions_input else 0

        for p in range(1, num_products):
            planned_qty = int(proportions_input[p])
            if planned_qty <= 0:
                continue

            qty_minus_flag = products[p][4] if len(products[p]) > 4 else 0
            qty_minus_min = products[p][7] if len(products[p]) > 7 else 0
            product_name = products[p][0] if len(products[p]) > 0 else ""
            product_id = products[p][2] if len(products[p]) > 2 else None

            if qty_minus_flag == 0:
                # qty_minus = 0: "жёсткий" продукт.
                # Нижняя граница: не меньше 0 смен.
                model.Add(product_counts[p] >= 0)
                # Верхняя граница по фактическим сменам: fact_real <= planned_qty + 6.
                # При 2 фактических сменах на одну модельную: 2 * count_model <= planned_qty + 6
                # => count_model <= ceil((planned_qty + 6) / 2).
                upper_model = (planned_qty + 6 + 1) // 2
                upper = min(max_count, upper_model)
                model.Add(product_counts[p] <= upper)
            else:
                # qty_minus = 1: в LONG-режиме по умолчанию не накладываем жёстких
                # нижних границ. Для большинства продуктов нет и верхнего потолка,
                # но для нескольких "жирных" артикулов ограничим перекос сверху.
                plan_model = (planned_qty + 1) // 2

                # Локальные верхние лимиты на явные перекосы только для нескольких
                # проблемных продуктов (по их GUID id из входных данных).
                heavy_overprod_ids = {
                    "83a5f825-bd64-4897-aec0-238c22d4f778",  # ст87026t
                    "f915776a-bbfc-4e36-a826-c7400e5ea542",  # ст11022амt6
                    "49bb002d-cdfa-4476-9768-f80ce0d2be51",  # ст87416t1
                }
                if product_id in heavy_overprod_ids and planned_qty > 0:
                    # Разрешаем до ~2x от плана в реальных сменах.
                    # fact_real ≈ 2 * product_counts[p] <= 2 * planned_qty
                    # => product_counts[p] <= planned_qty.
                    upper_model = planned_qty
                    upper = min(max_count, max(upper_model, 1))
                    model.Add(product_counts[p] <= upper)

                # Целевой агрегированный план: ~ plan_qty / 2 смен.
                # Штрафуем ТОЛЬКО недобор (односторонний shortfall), переизбыток не штрафуем в LONG.
                target_model = min(max_count, max(0, plan_model))
                diff = model.NewIntVar(-max_count, max_count, f"prop_long_diff_{p}")
                model.Add(diff == target_model - product_counts[p])
                zero = model.NewConstant(0)
                shortfall = model.NewIntVar(0, max_count, f"prop_long_shortfall_{p}")
                model.AddMaxEquality(shortfall, [diff, zero])

                # Усиливаем штраф за недобор для ключевых продуктов.
                important_shortfall_ids = {
                    "3c6972f0-4ad8-40b0-9f7d-68b1eebca3c0",  # ст31301амt18
                    "46ec2adb-3979-44ef-abee-bc9a81f04fae",  # ст87001t1
                    "27d53234-2675-4ea5-a07e-de9a8ce465a4",  # ст18310t1
                    "31bbe2cd-ea5c-459e-9cec-90d837dfe54a",  # ст87315t
                }
                prop_weight = 3 if product_id in important_shortfall_ids else 1
                proportion_objective_terms.append(shortfall * prop_weight)

    # Стратегии (переносим существующую логику без lday).
    # В LONG масштабы штрафов по стратегиям держим малыми (1-2 на единицу),
    # чтобы они были слабее, чем пропорции и простои.
    strategy_objective_terms: list[cp_model.IntVar] = []
    strategy_penalty_terms: list[cp_model.IntVar] = [
        model.NewIntVar(0, 0, f"strategy_penalty_long_{p}") for p in range(num_products)
    ]
    strategy_base_weight = 1

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

        weight = strategy_base_weight
        pen: cp_model.IntVar | None = None

        if strategy == "--":
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * count_end)
        elif strategy == "-":
            weight = strategy_base_weight
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_long_{p}")
            model.Add(diff == count_end - count_start)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines, f"strategy_posdiff_long_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * pos_diff)
        elif strategy == "=":
            weight = strategy_base_weight * 2
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_long_{p}")
            model.Add(diff == count_end - count_start)
            abs_diff = model.NewIntVar(0, num_machines, f"strategy_absdiff_long_{p}")
            model.AddAbsEquality(abs_diff, diff)
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * abs_diff)
        elif strategy == "+":
            weight = strategy_base_weight
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_long_{p}")
            model.Add(diff == count_start - count_end)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines, f"strategy_posdiff_long_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_long_{p}")
            model.Add(pen == weight * pos_diff)
        elif strategy == "++":
            weight = strategy_base_weight * 2
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

    # В LONG фиксируем штраф за простой в масштабе настроечного параметра,
    # без умножения на объём плана, чтобы он был понятным: 1 простой ≈ 10.
    downtime_penalty = max(2, settings.KFZ_DOWNTIME_PENALTY)

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
    # В LONG-режиме упрощённые переходы: возвращаем пустые словари.
    completed_transition: dict = {}
    same_as_prev: dict = {}

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


def create_model_simple(remains: list, products: list, machines: list, cleans: list,
                       max_daily_prod_zero: int, count_days: int,
                       dedicated_machines: list[int] | None = None,
                       product_divs: list[int] | None = None,
                       machine_divs: list[int] | None = None):
    """Упрощённая модель по станкам для длинного горизонта.

    Особенности:
    - работаем по реальным дням (без агрегирования 2x1);
    - жёсткое начальное состояние: в день 0 на каждой машине стоит её initial product;
    - жёсткое ограничение на количество переходов (смен продуктов) в день:
      не более ``max_daily_prod_zero`` смен по всем машинам;
    - учитываем разделение по типам машин/продуктов (цех 1 / цех 2);
    - пропорции и стратегии берём в упрощённом виде, как в основной модели.

    Остатки партий, lday и детальная логика двухдневного перехода не учитываются.
    """
    num_days = count_days
    num_machines = len(machines)
    num_products = len(products)

    # Локальные флаги для поэтапной отладки эвристик начального плана H1–H3.
    # В production-режиме (SIMPLE_DEBUG_H_START=False) считаем, что все
    # эвристики H1, H2, H3 включены как бизнес-логика SIMPLE.
    # При SIMPLE_DEBUG_H_START=True используем SIMPLE_DEBUG_H_MODE, чтобы
    # точечно включать/выключать эвристики для поиска конфликтов.
    debug_h_start = getattr(settings, "SIMPLE_DEBUG_H_START", False)
    debug_h_mode = getattr(settings, "SIMPLE_DEBUG_H_MODE", None)

    if not debug_h_start:
        # Обычный режим: все эвристики включены.
        ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT = True   # H1
        ENABLE_SIMPLE_SMALL_START_HEURISTIC = True   # H2
        ENABLE_SIMPLE_BIG_START_HEURISTIC = True     # H3
    else:
        mode = (debug_h_mode or "").upper()
        # Режимы отладки:
        #   "NONE" — все H1–H3 выключены
        #   "H1"   — только H1
        #   "H2"   — только H2
        #   "H3"   — только H3
        #   "H12"  — H1 и H2
        #   "H123" — H1, H2 и H3 (эквивалент обычного режима)
        if mode == "NONE":
            ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT = False
            ENABLE_SIMPLE_SMALL_START_HEURISTIC = False
            ENABLE_SIMPLE_BIG_START_HEURISTIC = False
        elif mode == "H1":
            ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT = True
            ENABLE_SIMPLE_SMALL_START_HEURISTIC = False
            ENABLE_SIMPLE_BIG_START_HEURISTIC = False
        elif mode == "H2":
            ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT = False
            ENABLE_SIMPLE_SMALL_START_HEURISTIC = True
            ENABLE_SIMPLE_BIG_START_HEURISTIC = False
        elif mode == "H3":
            ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT = False
            ENABLE_SIMPLE_SMALL_START_HEURISTIC = False
            ENABLE_SIMPLE_BIG_START_HEURISTIC = True
        elif mode == "H12":
            ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT = True
            ENABLE_SIMPLE_SMALL_START_HEURISTIC = True
            ENABLE_SIMPLE_BIG_START_HEURISTIC = False
        elif mode == "H123":
            ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT = True
            ENABLE_SIMPLE_SMALL_START_HEURISTIC = True
            ENABLE_SIMPLE_BIG_START_HEURISTIC = True
        else:
            # По умолчанию в debug-режиме включаем все эвристики.
            ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT = True
            ENABLE_SIMPLE_SMALL_START_HEURISTIC = True
            ENABLE_SIMPLE_BIG_START_HEURISTIC = True

    dedicated_set = set(dedicated_machines or [])

    # Крупная эвристика для больших продуктов без стартовых машин (п.3):
    # если включена SIMPLE_ENABLE_BIG_NOSTART_HEURISTIC, то для некоторых
    # продуктов без стартовых машин, объём которых сравним с мощностью одной
    # машины за горизонт, мы можем целиком забить одну совместимую машину этим
    # продуктом и запретить его на остальных машинах.
    # Интерпретируем SIMPLE_ENABLE_BIG_NOSTART_HEURISTIC как "включено, если
    # непустая строка". Это обходной путь вокруг строгого bool-parsing pydantic.
    raw_big_nostart = getattr(settings, "SIMPLE_ENABLE_BIG_NOSTART_HEURISTIC", None)
    if isinstance(raw_big_nostart, str):
        ENABLE_SIMPLE_BIG_NOSTART_HEURISTIC = raw_big_nostart.strip() != ""
    else:
        ENABLE_SIMPLE_BIG_NOSTART_HEURISTIC = bool(raw_big_nostart)

    all_machines = range(num_machines)
    all_days = range(num_days)
    all_products = range(num_products)

    proportions_input = [p[1] for p in products]

    # Отслеживаем, какие продукты подпадают под эвристики H1-H3, чтобы
    # при включённом APPLY_QTY_MINUS не накладывать на них дополнительные
    # нижние/верхние границы объёма из блока qty_minus (во избежание
    # дублирования/конфликтов).
    h1_products: set[int] = set()
    # Для H1 дополнительно храним набор станков, на которых продукт
    # стартует с qty=0 и для которых мы реально применили ограничение
    # по дням (has_higher_compatible=True).
    h1_product_machines: dict[int, set[int]] = {}
    h2_products: set[int] = set()
    h3_products: set[int] = set()

    # Нормализуем div по продуктам/машинам.
    # product_divs[p]: 1 или 2 — фиксированный цех;
    #                  0 — можно в любом цехе (но только в одном);
    #                 иначе / отсутствует — не используем div.
    if product_divs is None:
        product_divs = [0 for _ in range(num_products)]
    if machine_divs is None:
        machine_divs = [1 for _ in range(num_machines)]

    # Стартовые продукты на машинах
    initial_products: list[int] = []
    for (_, product_idx, m_id, t, remain_day) in machines:
        initial_products.append(product_idx)

    # Флаг: когда включён APPLY_QTY_MINUS, нижние/верхние границы по объёму
    # задаются через блок qty_minus. В этом случае не накладываем дополнительные
    # product-уровневые ограничения по H2/H3 (фиксированные объёмы и cap'ы),
    # чтобы избежать дублирования и жёстких конфликтов.
    use_qty_minus = bool(getattr(settings, "APPLY_QTY_MINUS", False))

    # Продукты с планом qty=0 (по текущему горизонту) — кандидаты на "технические" стартовые коды.
    zero_plan_products: set[int] = set()
    for p_idx in range(1, num_products):
        try:
            if int(proportions_input[p_idx]) == 0:
                zero_plan_products.add(p_idx)
        except Exception:
            continue

    # Машины, стартующие на таких продуктах.
    zero_plan_machines: set[int] = set(
        m_idx for m_idx, p0 in enumerate(initial_products) if p0 in zero_plan_products
    )

    # Карта: продукт -> список машин, на которых он стоит на начало (день 0).
    product_to_initial_machines: dict[int, list[int]] = {}
    for m_idx, p0 in enumerate(initial_products):
        if p0 <= 0:
            continue
        product_to_initial_machines.setdefault(p0, []).append(m_idx)

    model = cp_model.CpModel()

    jobs: dict[tuple[int, int], cp_model.IntVar] = {}
    work_days: list[tuple[int, int]] = []

    # В SIMPLE все дни считаются рабочими (чистки не исключают день из горизонта).
    # Чистки будем учитывать позже при перерасчёте смен/объёмов, но не в домене переменных.
    for m in all_machines:
        for d in all_days:
            work_days.append((m, d))
            # В SIMPLE не используем PRODUCT_ZERO: домен только 1..num_products-1.
            jobs[m, d] = model.NewIntVar(1, num_products - 1, f"job_simple_{m}_{d}")
            # Для dedicated-машин фиксируем исходный продукт на всех днях,
            # но только если стартовый продукт не равен 0 (PRODUCT_ZERO в LONG).
            if m in dedicated_set:
                fixed_product = initial_products[m]
                if fixed_product > 0:
                    model.Add(jobs[m, d] == fixed_product)

    # Жёсткое начальное состояние: в день 0 машина стоит на своём initial product,
    # если этот продукт > 0. Стартовое PRODUCT_ZERO в SIMPLE не фиксируем
    # (машина свободна выбрать любой реальный продукт с первого дня).
    for m in all_machines:
        init_p = initial_products[m]
        if init_p > 0:
            model.Add(jobs[m, 0] == init_p)

    # 1) Продукты с планом qty=0, которые стоят только как начальные.
    # Разрешаем их только в первые 1-2 дня, далее на этих машинах продукт
    # использовать нельзя ("работаем до ближайшей перезаправки").
    if ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT and zero_plan_machines:
        # Количество машин с продуктами qty=0 на старте.
        total_zero_start_machines = len(zero_plan_machines)
        # Разрешённое число дней для таких продуктов на каждой машине:
        # ceil(N_zero / max_daily_prod_zero). Так мы гарантируем, что
        # суммарного лимита переходов по дням достаточно, чтобы вывести
        # все машины со стартовых qty=0 продуктов.
        if max_daily_prod_zero > 0:
            max_zero_days_per_machine = (total_zero_start_machines + max_daily_prod_zero - 1) // max_daily_prod_zero
        else:
            max_zero_days_per_machine = num_days
        max_zero_days_per_machine = min(num_days, max_zero_days_per_machine)

        # Диагностика для H1: логируем какие именно продукты и машины попадают
        # под ограничение и каков горизонт.
        try:
            logger.info(
                "SIMPLE H1: zero-plan start products: max_zero_days=%d, num_days=%d, N_zero=%d, max_daily_prod_zero=%d, machines=%s",
                max_zero_days_per_machine,
                num_days,
                total_zero_start_machines,
                max_daily_prod_zero,
                list(zero_plan_machines),
            )
        except Exception:
            pass

        for m in sorted(zero_plan_machines):
            p0 = initial_products[m]
            name_p0 = products[p0][0] if 0 <= p0 < len(products) else f"p{p0}"
            plan_shifts_p0 = int(proportions_input[p0]) if p0 < len(proportions_input) else 0

            # Проверяем, существует ли вообще хотя бы один совместимый продукт
            # с индексом > p0 для этой машины. Если нет, то запрещать p0 после
            # 1-2 дней нельзя, иначе домен станет пустым из-за INDEX_UP/типов/div.
            has_higher_compatible = False
            try:
                machine_type = machines[m][3]
                m_div = machine_divs[m] if 0 <= m < len(machine_divs) else 1
                for p in range(p0 + 1, num_products):
                    product_machine_type_req = products[p][3]
                    prod_div = product_divs[p] if 0 <= p < len(product_divs) else 0
                    type_incompatible = (
                        product_machine_type_req > 0 and machine_type != product_machine_type_req
                    )
                    div_incompatible = (
                        prod_div in (1, 2) and m_div != prod_div
                    )
                    if not (type_incompatible or div_incompatible):
                        has_higher_compatible = True
                        break
            except Exception:
                # В случае ошибки безопаснее не применять жёсткое ограничение.
                has_higher_compatible = False

            try:
                logger.info(
                    "  H1: machine=%d, start_product_idx=%d (%s), plan_shifts=%d, has_higher_compatible=%s",
                    m,
                    p0,
                    name_p0,
                    plan_shifts_p0,
                    has_higher_compatible,
                )
            except Exception:
                pass

            if not has_higher_compatible:
                # Оставляем p0 без ограничения по дням на этой машине.
                continue

            # Продукт p0 попадает под эвристику H1 (zero-plan start product).
            h1_products.add(p0)
            if p0 not in h1_product_machines:
                h1_product_machines[p0] = set()
            h1_product_machines[p0].add(m)

            # Запрещаем p0 на машине m, начиная с дня max_zero_days_per_machine.
            for d in range(max_zero_days_per_machine, num_days):
                model.Add(jobs[m, d] != p0)

        # Дополнительно запрещаем H1‑коды на всех машинах, где они не стоят на
        # начало. Таким образом, нулевые стартовые продукты могут встречаться
        # только на своих начальных станках и только в первые max_zero_days_per_machine
        # дней (ограничение по дням задаётся выше).
        if ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT and h1_products:
            for p in h1_products:
                allowed_machines = h1_product_machines.get(p, set())
                for m in all_machines:
                    if m in allowed_machines:
                        continue
                    for d in all_days:
                        model.Add(jobs[m, d] != p)

    # 2) и 3) Продукты с планом qty>0 и особыми условиями на стартовых станках.
    # Используем план в сменах и переводим его в дни.
    small_caps: dict[int, int] = {}
    # Дополнительные капы для "маленьких" строгих продуктов (qty_minus=0,
    # план < 0.5 машино-периода): ограничиваем сверху примерно план + 1 смена.
    # Кап храним в ДНЯХ (product_counts[p]); связь с планом в сменах учитываем
    # через округление по модулю shifts_per_day.
    strict_small_caps_days: dict[int, int] = {}
    # Глобальные капы для всех продуктов с небольшим планом (qty < 0.7 * машино-период):
    # ограничиваем сверху plan_days + 2 дня (с учётом округлений и qty_minus_min).
    global_small_caps_days: dict[int, int] = {}

    shifts_per_day = 3
    half_machine_period_shifts = (num_days * shifts_per_day) // 2
    capacity_shifts_one_machine = num_days * shifts_per_day
    small_plan_threshold_shifts = int(0.7 * capacity_shifts_one_machine)

    for p in range(1, num_products):
        plan_shifts = int(proportions_input[p])
        if plan_shifts <= 0:
            continue
        plan_days = (plan_shifts + shifts_per_day - 1) // shifts_per_day
        machines_for_p = product_to_initial_machines.get(p, [])
        if not machines_for_p:
            continue

        # Берём первую машину, на которой продукт стоит на начало.
        m0 = machines_for_p[0]
        capacity_days_m0 = num_days

        qty_minus_flag = products[p][4]
        strategy = products[p][9] if len(products[p]) > 9 else ""

        # Глобальный кап для всех продуктов с небольшим планом: qty < 0.7 * машино-период.
        # Кап задаётся по дням как plan_days + 2, но не ниже, чем минимальный объём
        # по qty_minus_min (если он есть), чтобы не создать противоречий с нижней границей.
        if plan_shifts > 0 and plan_shifts < small_plan_threshold_shifts:
            qty_minus_min_shifts = int(products[p][7]) if len(products[p]) > 7 else 0
            if qty_minus_min_shifts > 0:
                min_days_global = (qty_minus_min_shifts + shifts_per_day - 1) // shifts_per_day
            else:
                min_days_global = 0
            cap_days_global = plan_days + 2
            if min_days_global > 0 and cap_days_global < min_days_global:
                cap_days_global = min_days_global
            prev_cap_g = global_small_caps_days.get(p)
            if prev_cap_g is None or cap_days_global < prev_cap_g:
                global_small_caps_days[p] = cap_days_global

        # 2) Если продукт стартует ровно на одном станке и его план меньше
        # длины горизонта этого станка, то планируем его только в начале на
        # этом станке. Для строгих продуктов (qty_minus=0) фиксируем первые
        # plan_days дней и запрещаем продукт на других станках.
        #
        # При включённом APPLY_QTY_MINUS (use_qty_minus=True) дополнительные
        # product-уровневые нижние/верхние границы по объёмам задаёт блок
        # qty_minus, поэтому эвристику H2 отключаем, чтобы не дублировать
        # ограничения и не создавать конфликтов с INDEX_UP и другими лимитами.
        if (
            ENABLE_SIMPLE_SMALL_START_HEURISTIC
            and len(machines_for_p) == 1
            and plan_days < capacity_days_m0
        ):
            # Продукт p подпадает под эвристику H2.
            h2_products.add(p)

            # Для строгих продуктов с маленьким планом (qty_minus=0 и объём меньше
            # половины машино-периода) закладываем общий верхний кап в сменах:
            # fact_shifts <= plan_shifts + 1. Важно, что это сработает и тогда,
            # когда H2 выключает ограничения qty_minus для этого продукта.
            if qty_minus_flag == 0 and plan_shifts < half_machine_period_shifts:
                # Переводим условие "план + 1 смена" в дни. При агрегировании 3 смен
                # в 1 день точное ограничение по сменам может конфликтовать с
                # округлением, поэтому берём верхнюю границу по дням:
                #   max_days = plan_days (+1 день, если план ровно кратен 3 сменам).
                extra_day = 1 if (plan_shifts % shifts_per_day == 0) else 0
                cap_days = plan_days + extra_day
                prev_cap = strict_small_caps_days.get(p)
                if prev_cap is None or cap_days < prev_cap:
                    strict_small_caps_days[p] = cap_days
            # Диагностика для H2: логируем параметры продукта и машины.
            try:
                name_p = products[p][0]
                # Список всех машин, где p стоит на начало
                m_list = list(machines_for_p)
                # Подбор всех продуктов, совместимых с m0 по type/div
                compat_indices: list[int] = []
                machine_type = machines[m0][3]
                m_div = machine_divs[m0] if 0 <= m0 < len(machine_divs) else 1
                for pp in range(1, num_products):
                    product_machine_type_req = products[pp][3]
                    prod_div = product_divs[pp] if 0 <= pp < len(product_divs) else 0
                    type_incompatible = (
                        product_machine_type_req > 0 and machine_type != product_machine_type_req
                    )
                    div_incompatible = (
                        prod_div in (1, 2) and m_div != prod_div
                    )
                    if not (type_incompatible or div_incompatible):
                        compat_indices.append(pp)

                init_p_m0 = initial_products[m0]
                compat_lt = [pp for pp in compat_indices if pp < p]
                compat_eq = [pp for pp in compat_indices if pp == p]
                compat_gt = [pp for pp in compat_indices if pp > p]

                logger.info(
                    "SIMPLE H2: small-start candidate p=%d (%s), plan_shifts=%d, plan_days=%d, "
                    "m0=%d, init_p_m0=%d, machines_for_p=%s, capacity_days_m0=%d, qty_minus_flag=%d, strategy=%s, "
                    "compat_lt=%s, compat_eq=%s, compat_gt=%s",
                    p,
                    name_p,
                    plan_shifts,
                    plan_days,
                    m0,
                    init_p_m0,
                    m_list,
                    capacity_days_m0,
                    qty_minus_flag,
                    strategy,
                    compat_lt,
                    compat_eq,
                    compat_gt,
                )
            except Exception:
                pass

            # Строгий продукт: фиксируем первые plan_days дней на m0 ровно p
            # и запрещаем p на остальных машинах. При этом НЕ запрещаем p на
            # m0 после plan_days, ограничивая его только через верхний cap
            # по product_counts[p]. Это уменьшает риск конфликтов с INDEX_UP
            # и другими ограничениями.
            if qty_minus_flag == 0:
                for d in range(min(plan_days, num_days)):
                    model.Add(jobs[m0, d] == p)
                for m in all_machines:
                    if m == m0:
                        continue
                    for d in all_days:
                        model.Add(jobs[m, d] != p)

            # Если план по дням существенно меньше половины горизонта станка,
            # ограничиваем сверху общее количество дней для p (чтобы избежать
            # чрезмерной перепланировки). Верхняя граница ~2*plan_days.
            # Сам cap по product_counts[p] накладывается ниже, после того как
            # будут созданы переменные product_counts.
            if plan_days * 2 <= capacity_days_m0:
                cap_small = max(1, plan_days * 2)
                try:
                    logger.info(
                        "  H2: apply cap for p=%d (%s): cap_small=%d (plan_days=%d, capacity_days_m0=%d)",
                        p,
                        products[p][0],
                        cap_small,
                        plan_days,
                        capacity_days_m0,
                    )
                except Exception:
                    pass
                # Сохраняем кап в словарь, чтобы применить после создания product_counts.
                small_caps[p] = max(small_caps.get(p, 0), cap_small)

        # 3) Крупные продукты: если есть РОВНО ОДНА машина, на которой этот
        # продукт стоит на начало (init_p == p), и его план в сменах велик
        # (порядка >= 70 из 83, т.е. план по дням покрывает ~80% и более
        # горизонта), и стратегия "=" или "+", то забиваем эту машину
        # продуктом p на весь горизонт и исключаем p с других машин.
        #
        # При включённом APPLY_QTY_MINUS (use_qty_minus=True) объёмные
        # ограничения реализованы через qty_minus, поэтому H3 (жёсткая
        # фиксация всего горизонта под p и запрет p на других машинах)
        # отключается, чтобы не дублировать/усиливать эти ограничения.
        if (
            ENABLE_SIMPLE_BIG_START_HEURISTIC
            and len(machines_for_p) == 1
            and initial_products[m0] == p
            and plan_days * 5 >= capacity_days_m0 * 4  # ~80% горизонта
            and strategy in ("=", "+")
        ):
            # Продукт p подпадает под эвристику H3.
            h3_products.add(p)
            # Диагностика для H3: логируем параметры продукта и машины и
            # возможные индексы на m0 с учётом type/div.
            try:
                name_p = products[p][0]
                m_list = list(machines_for_p)
                compat_indices: list[int] = []
                machine_type = machines[m0][3]
                m_div = machine_divs[m0] if 0 <= m0 < len(machine_divs) else 1
                for pp in range(1, num_products):
                    product_machine_type_req = products[pp][3]
                    prod_div = product_divs[pp] if 0 <= pp < len(product_divs) else 0
                    type_incompatible = (
                        product_machine_type_req > 0 and machine_type != product_machine_type_req
                    )
                    div_incompatible = (
                        prod_div in (1, 2) and m_div != prod_div
                    )
                    if not (type_incompatible or div_incompatible):
                        compat_indices.append(pp)

                init_p_m0 = initial_products[m0]
                compat_lt = [pp for pp in compat_indices if pp < p]
                compat_eq = [pp for pp in compat_indices if pp == p]
                compat_gt = [pp for pp in compat_indices if pp > p]

                logger.info(
                    "SIMPLE H3: big-start candidate p=%d (%s), plan_shifts=%d, plan_days=%d, "
                    "m0=%d, init_p_m0=%d, machines_for_p=%s, capacity_days_m0=%d, qty_minus_flag=%d, strategy=%s, "
                    "compat_lt=%s, compat_eq=%s, compat_gt=%s",
                    p,
                    name_p,
                    plan_shifts,
                    plan_days,
                    m0,
                    init_p_m0,
                    m_list,
                    capacity_days_m0,
                    qty_minus_flag,
                    strategy,
                    compat_lt,
                    compat_eq,
                    compat_gt,
                )
            except Exception:
                pass

            # Полное заполнение m0 продуктом p на весь горизонт.
            for d in all_days:
                model.Add(jobs[m0, d] == p)
            # Запрещаем p на других машинах.
            for m in all_machines:
                if m == m0:
                    continue
                for d in all_days:
                    model.Add(jobs[m, d] != p)

        # --- Дополнительная крупная эвристика для больших продуктов без стартовых машин ---
        # Если включён ENABLE_SIMPLE_BIG_NOSTART_HEURISTIC, рассматриваем продукты,
        # которые не стоят ни на одной машине в день 0 (len(machines_for_p) == 0),
        # имеют ненулевой плановый объём и могут по объёму примерно занять одну
        # машину на весь горизонт: план по дням в диапазоне [0.7 * num_days, 1.3 * num_days].
        # Для такого продукта p выбираем одну подходящую по type/div машину m0 и
        # забиваем её этим продуктом на весь горизонт, одновременно запрещая p
        # на остальных машинах. Продукт добавляем в h3_products, чтобы блок
        # APPLY_QTY_MINUS не накладывал на него дополнительные объёмные пределы.
        if ENABLE_SIMPLE_BIG_NOSTART_HEURISTIC and not machines_for_p and plan_days > 0:
            # Оцениваем, сопоставим ли объём мощности одной машины.
            if (
                plan_days * 10 >= num_days * 7  # план >= ~0.7 горизонта
                and plan_days * 10 <= num_days * 13  # план <= ~1.3 горизонта
            ):
                # Ищем совместимые по type/div машины.
                compatible_machines: list[int] = []
                for m_idx in all_machines:
                    machine_type = machines[m_idx][3]
                    m_div = machine_divs[m_idx] if 0 <= m_idx < len(machine_divs) else 1
                    product_machine_type_req = products[p][3]
                    prod_div = product_divs[p] if 0 <= p < len(product_divs) else 0
                    type_incompatible = (
                        product_machine_type_req > 0
                        and machine_type != product_machine_type_req
                    )
                    div_incompatible = (
                        prod_div in (1, 2) and m_div != prod_div
                    )
                    if not (type_incompatible or div_incompatible):
                        compatible_machines.append(m_idx)

                # Выбираем машину: предпочтительно без стартового продукта (initial_products[m]==0).
                m0_big: int | None = None
                for m_idx in compatible_machines:
                    if initial_products[m_idx] == 0:
                        m0_big = m_idx
                        break
                if m0_big is None and compatible_machines:
                    m0_big = compatible_machines[0]

                if m0_big is not None:
                    try:
                        name_p = products[p][0]
                        logger.info(
                            "SIMPLE BIG_NOSTART: p=%d (%s), plan_days=%d, num_days=%d, "
                            "m0=%d, compatible_machines=%s",
                            p,
                            name_p,
                            plan_days,
                            num_days,
                            m0_big,
                            compatible_machines,
                        )
                    except Exception:
                        pass

                    # Помечаем продукт как попавший под крупную эвристику (аналог H3).
                    h3_products.add(p)

                    # Фиксируем m0_big целиком под продукт p.
                    for d in all_days:
                        model.Add(jobs[m0_big, d] == p)
                    # Запрещаем p на остальных машинах.
                    for m_idx in all_machines:
                        if m_idx == m0_big:
                            continue
                        for d in all_days:
                            model.Add(jobs[m_idx, d] != p)

    # ------------ Подсчёт общего количества каждого продукта ------------
    product_produced_bools: dict[tuple[int, int, int], cp_model.BoolVar] = {}
    # В SIMPLE не используем продукт 0, поэтому считаем только p >= 1.
    for p in range(1, num_products):
        for m, d in work_days:
            b = model.NewBoolVar(f"prod_simple_{p}_{m}_{d}")
            product_produced_bools[p, m, d] = b
            model.Add(jobs[m, d] == p).OnlyEnforceIf(b)
            model.Add(jobs[m, d] != p).OnlyEnforceIf(b.Not())

    # Общее количество дней по продукту (все машины × все дни).
    product_counts: list[cp_model.IntVar] = [
        model.NewIntVar(0, num_machines * num_days, f"count_prod_simple_{p}") for p in all_products
    ]

    debug_upper_caps = getattr(settings, "SIMPLE_DEBUG_PRODUCT_UPPER_CAPS", None)

    for p in range(1, num_products):
        model.Add(product_counts[p] == sum(
            product_produced_bools[p, m, d] for m, d in work_days
        ))

    # Глобальный кап по H1‑продуктам временно отключён: ограничение
    # действует только по дням на стартовых станках (см. выше) и по
    # запрету на других машинах. Это нужно, чтобы проверить влияние
    # жёсткого "2 дня на продукт" на выполнимость STRICT9.
    # if ENABLE_SIMPLE_ZERO_PLAN_START_LIMIT and h1_products:
    #     for p in h1_products:
    #         cap_days = max_zero_days_per_machine
    #         cap_days = min(cap_days, num_machines * num_days)
    #         model.Add(product_counts[p] <= cap_days)

    # Отладочные верхние капы по продуктам: если заданы, просто добавляем
    # ограничение вида product_counts[p] <= cap_dbg. Оно сочетается с H1‑капами
    # (берётся более жёсткая граница).
    if debug_upper_caps:
        for p, cap_dbg in debug_upper_caps.items():
            if 0 < p < num_products:
                model.Add(product_counts[p] <= int(cap_dbg))

    # ------------ Монотонность по дням для каждого продукта ------------
    # Для каждого продукта p считаем дневные мощности C[p,d] = кол-во машин,
    # занятых продуктом p в день d, и вводим бинарную переменную направления
    # is_up[p]: 1 — профиль неубывающий (выводим/наращиваем), 0 — невозрастающий
    # (снимаем). Волнистые паттерны 3-1-0-1-2-3 запрещены.
    C_pd: dict[tuple[int, int], cp_model.IntVar] = {}
    is_up: dict[int, cp_model.BoolVar] = {}

    if getattr(settings, "SIMPLE_USE_MONOTONE_COUNTS", True):
        # Для верхней границы берём максимум машин, на которых продукт в принципе
        # может стоять по type/div. Это уменьшает Big-M в ограничениях.
        max_cap_per_product: dict[int, int] = {}
        for p in range(1, num_products):
            product_machine_type_req = products[p][3]
            prod_div = product_divs[p] if 0 <= p < len(product_divs) else 0
            cap = 0
            for m in all_machines:
                machine_type = machines[m][3]
                m_div = machine_divs[m] if 0 <= m < len(machine_divs) else 1
                type_incompatible = (
                    product_machine_type_req > 0 and machine_type != product_machine_type_req
                )
                div_incompatible = (
                    prod_div in (1, 2) and m_div != prod_div
                )
                if not (type_incompatible or div_incompatible):
                    cap += 1
            if cap <= 0:
                cap = num_machines
            max_cap_per_product[p] = cap

        for p in range(1, num_products):
            is_up[p] = model.NewBoolVar(f"simple_is_up_{p}")
            M_p = max_cap_per_product[p]
            for d in all_days:
                C = model.NewIntVar(0, M_p, f"C_simple_{p}_{d}")
                C_pd[p, d] = C
                # C[p,d] = sum_m is_prod[p,m,d]
                model.Add(C == sum(
                    product_produced_bools[p, m, d] for m in all_machines
                ))
            # Монотонность по дням: либо неубывающий, либо невозрастающий профиль.
            for d in range(num_days - 1):
                diff = model.NewIntVar(-M_p, M_p, f"Cdiff_simple_{p}_{d}")
                model.Add(diff == C_pd[p, d + 1] - C_pd[p, d])
                # Если is_up[p] = 1 → diff >= 0; если 0 → ограничение ослабляем до diff >= -M_p.
                model.Add(diff >= 0 - M_p * (1 - is_up[p]))
                # Если is_up[p] = 0 → diff <= 0; если 1 → ограничение ослабляем до diff <= M_p.
                model.Add(diff <= 0 + M_p * is_up[p])
    else:
        # Если монотонность отключена, всё равно инициализируем C_pd для совместимости,
        # но без ограничений на разности. Верхняя граница берётся как num_machines.
        for p in range(1, num_products):
            for d in all_days:
                C = model.NewIntVar(0, num_machines, f"C_simple_{p}_{d}")
                C_pd[p, d] = C
                model.Add(C == sum(
                    product_produced_bools[p, m, d] for m in all_machines
                ))
        # Раньше здесь дополняли модель верхними капами по product_counts[p]
        # из эвристики H2/H3 (small_caps, strict_small_caps_days,
        # global_small_caps_days). В текущем варианте для увеличения
        # выполнимости в LONG_SIMPLE эти капы отключаем и оставляем только
        # отладочные верхние границы при необходимости (см. debug_upper_caps
        # выше).
    # Минимальные объёмы по qty_minus / qty_minus_min (как в create_model),
    # с учётом того, что в SIMPLE product_counts[p] считаются в ДНЯХ, а qty и
    # qty_minus_min заданы в СМЕНАХ. Предполагаем 3 смены в день.
    #
    # Для отладки взаимодействия с H1–H3 поддерживаем необязательный фильтр
    # SIMPLE_QTY_MINUS_SUBSET в settings: если он задан (итерируемое множество
    # индексов продуктов), то нижние границы по qty_minus применяются только
    # к этим продуктам.
    if settings.APPLY_QTY_MINUS:
        shifts_per_day = 3
        debug_subset = getattr(settings, "SIMPLE_QTY_MINUS_SUBSET", None)
        debug_max_idx = getattr(settings, "SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX", None)

        # --- Выбор балансирующих продуктов по цехам (div) среди qty_minus != 0 ---
        # Для каждого div отбираем до 2 продуктов с наибольшим планом в днях,
        # которые не попали под H1–H3 и не вырезаны отладочными фильтрами.
        balance_products: set[int] = set()
        per_div_candidates: dict[int, list[tuple[int, int]]] = {}

        for p in range(1, num_products):
            # Те же фильтры, что и для основного блока qty_minus.
            if p in h1_products or p in h2_products or p in h3_products:
                continue
            if debug_subset is not None and p not in debug_subset:
                continue
            if debug_max_idx is not None and p == debug_max_idx:
                continue

            qty_shifts = int(products[p][1])
            if qty_shifts <= 0:
                continue
            qty_minus_flag = products[p][4]
            if qty_minus_flag == 0:
                continue

            plan_days = (qty_shifts + shifts_per_day - 1) // shifts_per_day
            if plan_days <= 0:
                continue

            prod_div = product_divs[p] if 0 <= p < len(product_divs) else 0
            per_div_candidates.setdefault(prod_div, []).append((p, plan_days))

        # Разрешаем не более SIMPLE_QTY_MINUS_MAX_BALANCE_PER_DIV балансирующих
        # продуктов на цех (div). Значение по умолчанию = 1 и настраивается
        # через settings.SIMPLE_QTY_MINUS_MAX_BALANCE_PER_DIV.
        try:
            max_balance_per_div = int(getattr(settings, "SIMPLE_QTY_MINUS_MAX_BALANCE_PER_DIV", 1))
        except Exception:
            max_balance_per_div = 1
        if max_balance_per_div < 1:
            max_balance_per_div = 1
        for prod_div, cand_list in per_div_candidates.items():
            cand_list.sort(key=lambda t: t[1], reverse=True)
            for p, _plan_days in cand_list[:max_balance_per_div]:
                balance_products.add(p)

        # Диагностический вывод: классификация продуктов по режимам qty_minus.
        # mode:
        #   STRICT_QM0 — строгий qty_minus=0, только нижняя граница >= плану;
        #   BALANCE    — продукт из balance_products (только нижняя граница min_days);
        #   CORRIDOR   — обычный продукт с коридором plan_days±1;
        #   SKIP_DEBUG — продукт вырезан отладочными фильтрами (subset/max_idx).
        try:
            for p in range(1, num_products):
                try:
                    qty_shifts = int(products[p][1])
                except Exception:
                    qty_shifts = 0
                qty_minus_flag = products[p][4]
                try:
                    qty_minus_min_shifts = int(products[p][7]) if len(products[p]) > 7 else 0
                except Exception:
                    qty_minus_min_shifts = 0
                prod_div = product_divs[p] if 0 <= p < len(product_divs) else 0

                if debug_subset is not None and p not in debug_subset:
                    mode = "SKIP_DEBUG"
                elif debug_max_idx is not None and p == debug_max_idx:
                    mode = "SKIP_DEBUG"
                elif qty_shifts <= 0:
                    mode = "ZERO_PLAN"
                elif qty_minus_flag == 0:
                    mode = "STRICT_QM0"
                elif p in balance_products:
                    mode = "BALANCE"
                else:
                    mode = "CORRIDOR"

                name_p = products[p][0] if 0 <= p < len(products) else f"p{p}"
                logger.info(
                    "SIMPLE qty_minus: p=%d name=%s div=%d qty=%d qm=%d qm_min=%d mode=%s",
                    p,
                    name_p,
                    prod_div,
                    qty_shifts,
                    qty_minus_flag,
                    qty_minus_min_shifts,
                    mode,
                )
        except Exception:
            pass

        for p in range(1, num_products):

            # Если задан отладочный поднабор продуктов, применяем qty_minus
            # только к нему.
            if debug_subset is not None and p not in debug_subset:
                continue

            # Если мы в режиме отладки максимально допустимого объёма для
            # конкретного продукта (SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX), не
            # накладываем на него нижнюю границу qty_minus, чтобы модель
            # могла свободно выбирать объём и мы могли его максимизировать.
            if debug_max_idx is not None and p == debug_max_idx:
                continue

            qty_shifts = int(products[p][1])
            if qty_shifts <= 0:
                continue
            qty_minus_flag = products[p][4]
            qty_minus_min_shifts = int(products[p][7]) if len(products[p]) > 7 else 0

            # Переводим смены в минимальное количество дней: ceil(x / shifts_per_day).
            plan_days = (qty_shifts + shifts_per_day - 1) // shifts_per_day
            min_days = (qty_minus_min_shifts + shifts_per_day - 1) // shifts_per_day if qty_minus_min_shifts > 0 else 0

            if qty_minus_flag == 0:
                # "Строгий" продукт: требуем не меньше планового объёма в днях.
                # Не навязываем верхнюю границу здесь, чтобы не получать невыполнимость
                # из-за округлений и ограничений по мощностям.
                model.Add(product_counts[p] >= plan_days)
            else:
                # qty_minus != 0: продукты, которые могут отклоняться от плана.
                # Для продуктов под эвристиками H1–H3 оставляем только нижнюю
                # границу по объёму (min_days/plan_days), без жёсткого верха,
                # чтобы не усиливать уже заданные структурные ограничения.
                if p in h1_products or p in h2_products or p in h3_products:
                    if plan_days > 0 or min_days > 0:
                        lower = max(plan_days, min_days)
                        model.Add(product_counts[p] >= lower)
                # Для небольшой подгруппы balance_products (по div) оставляем
                # только нижнюю границу (min_days), чтобы они могли балансировать
                # общий перекос.
                elif p in balance_products:
                    if min_days > 0:
                        model.Add(product_counts[p] >= min_days)
                else:
                    # Для всех остальных вводим жёсткий коридор вокруг плана:
                    # примерно plan_days ± 1 день с учётом минимальной границы min_days.
                    if plan_days > 0:
                        lower_bound = max(min_days, max(plan_days - 1, 0))
                        upper_bound = plan_days + 1
                        model.Add(product_counts[p] >= lower_bound)
                        model.Add(product_counts[p] <= upper_bound)
                    elif min_days > 0:
                        # На всякий случай, если план по дням 0, но есть min_days
                        # из qty_minus_min, ограничиваем только снизу.
                        model.Add(product_counts[p] >= min_days)

    # Ограничение по типам машин (machine_type) и по цехам (div)
    for p in all_products:
        product_machine_type_req = products[p][3]
        prod_div = product_divs[p] if 0 <= p < len(product_divs) else 0

        for m in all_machines:
            machine_type = machines[m][3]
            m_div = machine_divs[m] if 0 <= m < len(machine_divs) else 1

            # Ограничение по типу машины
            type_incompatible = (product_machine_type_req > 0 and machine_type != product_machine_type_req)
            # Ограничение по цеху для фиксированных продуктов
            div_incompatible = (prod_div in (1, 2) and m_div != prod_div)

            if type_incompatible or div_incompatible:
                for d in all_days:
                    if (m, d) in work_days:
                        model.Add(jobs[m, d] != p)

    # Вариант B для продуктов с div = 0: выбор единственного цеха в модели.
    # Для каждого такого продукта p вводим y[p,1], y[p,2] с y[p,1] + y[p,2] = 1,
    # и требуем: если p стоит на машине m (div=m_div), то y[p,m_div] = 1.
    flex_products = [p for p in all_products if (0 <= p < len(product_divs) and product_divs[p] == 0 and p != 0)]

    y_div: dict[tuple[int, int], cp_model.BoolVar] = {}
    for p in flex_products:
        y1 = model.NewBoolVar(f"y_div1_p{p}")
        y2 = model.NewBoolVar(f"y_div2_p{p}")
        model.Add(y1 + y2 == 1)
        y_div[p, 1] = y1
        y_div[p, 2] = y2

    for p in flex_products:
        for m in all_machines:
            m_div = machine_divs[m] if 0 <= m < len(machine_divs) else 1
            if m_div not in (1, 2):
                # Неподдерживаемый div – на всякий случай запрещаем p на этой машине.
                for d in all_days:
                    if (m, d) in work_days:
                        model.Add(jobs[m, d] != p)
                continue
            y_pm = y_div[p, m_div]
            for d in all_days:
                if (m, d) not in work_days:
                    continue
                b = product_produced_bools[p, m, d]
                # Если p стоит на (m,d), соответствующий y[p,div] обязан быть 1.
                model.AddImplication(b, y_pm)

    # Жёсткое ограничение на количество переходов (смен продуктов) в день.
    # Переход считаем, если на машине m в день d продукт отличается от дня d-1.
    change_bools: dict[tuple[int, int], cp_model.BoolVar] = {}
    simple_disable_index_up = getattr(settings, "SIMPLE_DISABLE_INDEX_UP", False)

    for m in all_machines:
        for d in range(1, num_days):
            ch = model.NewBoolVar(f"change_simple_{m}_{d}")
            change_bools[m, d] = ch
            # Определяем факт смены продукта между днями d-1 и d.
            model.Add(jobs[m, d] != jobs[m, d - 1]).OnlyEnforceIf(ch)
            model.Add(jobs[m, d] == jobs[m, d - 1]).OnlyEnforceIf(ch.Not())
            # Бизнес-логика "индекс только вверх": при переходе индекс продукта
            # должен увеличиваться (как в LONG-модели).
            if settings.APPLY_INDEX_UP and not simple_disable_index_up:
                model.Add(jobs[m, d] > jobs[m, d - 1]).OnlyEnforceIf(ch)

    for d in all_days:
        day_changes = [ch for (m, dd), ch in change_bools.items() if dd == d]
        if not day_changes:
            continue
        model.Add(sum(day_changes) <= max_daily_prod_zero)

    # --- Ограничение: на каждой машине каждый продукт может образовывать не
    # более одного непрерывного блока (сегмента). Это запрещает паттерны
    # p -> q -> p на одной машине, но не навязывает глобальный порядок индексов
    # как INDEX_UP.
    start_seg: dict[tuple[int, int, int], cp_model.BoolVar] = {}

    for p in range(1, num_products):
        for m in all_machines:
            for d in all_days:
                if d == 0:
                    # start_seg[p,m,0] = (jobs[m,0] == p)
                    b0 = product_produced_bools[p, m, 0]
                    start_seg[p, m, 0] = b0
                else:
                    b_cur = product_produced_bools[p, m, d]
                    b_prev = product_produced_bools[p, m, d - 1]
                    s = model.NewBoolVar(f"start_seg_p{p}_m{m}_d{d}")
                    start_seg[p, m, d] = s
                    # s >= b_cur - b_prev
                    model.Add(s >= b_cur - b_prev)
                    # s <= b_cur
                    model.Add(s <= b_cur)
                    # s <= 1 - b_prev
                    model.Add(s <= 1 - b_prev)

            # На машине m для продукта p суммарное количество стартов сегментов не
            # превышает 1.
            seg_starts = [start_seg[p, m, d] for d in all_days]
            if seg_starts:
                model.Add(sum(seg_starts) <= 1)

    # Считаем общее количество переходов по всем машинам и дням, чтобы
    # при необходимости штрафовать их в целевой функции.
    if change_bools:
        max_changes = num_machines * max(0, num_days - 1)
        total_changes = model.NewIntVar(0, max_changes, "total_changes_simple")
        model.Add(total_changes == sum(change_bools[m_d] for m_d in change_bools))
    else:
        total_changes = None

    # Для совместимости с интерфейсом возвращаем total_products_count,
    # но в упрощённой модели SIMPLE он не участвует в целевой функции.
    total_products_count = model.NewIntVar(0, num_machines * num_days, "total_products_count_simple")
    model.Add(total_products_count == sum(product_counts[p] for p in range(1, len(products))))

    # Внутренняя цель по пропорциям для LONG_SIMPLE (SIMPLE) в LS_OVER:
    # линейный штраф за отклонение факта от плана по каждому продукту в ДНЯХ,
    # |fact_days - plan_days|. План в сменах (qty) переводим в дни через ceil(qty/3).
    #
    # Цель включается, только если APPLY_PROP_OBJECTIVE=True и SIMPLE_USE_PROP_MULT=False
    # (LS_OVER-сценарий). При SIMPLE_USE_PROP_MULT=True (LS_PROP) внутренняя
    # пропорциональная цель в этой ветке не задаётся: пропорции оцениваются снаружи.
    proportion_objective_terms: list[cp_model.IntVar] = []
    total_input_max = max(proportions_input) if proportions_input else 0

    use_prop_objective = bool(getattr(settings, "APPLY_PROP_OBJECTIVE", False))
    use_prop_mult = bool(getattr(settings, "SIMPLE_USE_PROP_MULT", False))

    if use_prop_objective and not use_prop_mult:
        shifts_per_day = 3
        max_days = num_machines * num_days
        # Создаём по одной переменной штрафа на каждый продукт p>=1;
        # индекс в списке proportion_objective_terms совпадает с p-1, чтобы
        # solver_result мог корректно восстановить penalty по p.
        for p in range(1, num_products):
            pen = model.NewIntVar(0, max_days, f"prop_simple_pen_{p}")
            proportion_objective_terms.append(pen)

            plan_shifts = int(proportions_input[p]) if p < len(proportions_input) else 0
            if plan_shifts <= 0:
                # Для продуктов без плана внутренний пропорциональный штраф = 0.
                model.Add(pen == 0)
                continue

            plan_days = (plan_shifts + shifts_per_day - 1) // shifts_per_day
            # diff = fact_days - plan_days
            diff = model.NewIntVar(-max_days, max_days, f"prop_simple_diff_{p}")
            model.Add(diff == product_counts[p] - plan_days)
            # abs_diff = |diff| = |fact_days - plan_days|
            abs_diff = model.NewIntVar(0, max_days, f"prop_simple_abs_{p}")
            model.AddAbsEquality(abs_diff, diff)

            # Пока используем единичный вес для всех продуктов; масштаб можно
            # настраивать через KFZ_DOWNTIME_PENALTY (см. ниже при сборке цели).
            model.Add(pen == abs_diff)
    else:
        # В режиме SIMPLE_USE_PROP_MULT (LS_PROP) внутренняя пропорциональная цель
        # здесь не задаётся: proportion_objective_terms остаётся пустым, а
        # внешний пропорциональный штраф считается в анализирующих скриптах.
        proportion_objective_terms = []

    # ------------ Стратегии по продуктам (по аналогии с create_model_long) ------------
    strategy_objective_terms: list[cp_model.IntVar] = []
    strategy_penalty_terms: list[cp_model.IntVar] = [
        model.NewIntVar(0, 0, f"strategy_penalty_simple_{p}") for p in range(num_products)
    ]
    # В SIMPLE держим штрафы по стратегиям малыми (1–2 на единицу),
    # чтобы они не доминировали над пропорциями и простоями.
    strategy_base_weight = 1

    for p in range(1, len(products)):
        if len(products[p]) < 10:
            continue
        strategy = products[p][9]
        if not strategy:
            continue

        count_start = model.NewIntVar(0, num_machines, f"machines_start_simple_{p}")
        count_end = model.NewIntVar(0, num_machines, f"machines_end_simple_{p}")

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

        weight = strategy_base_weight
        pen: cp_model.IntVar | None = None

        if strategy == "--":
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_simple_{p}")
            model.Add(pen == weight * count_end)
        elif strategy == "-":
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_simple_{p}")
            model.Add(diff == count_end - count_start)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines, f"strategy_posdiff_simple_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_simple_{p}")
            model.Add(pen == weight * pos_diff)
        elif strategy == "=":
            weight2 = strategy_base_weight * 2
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_simple_{p}")
            model.Add(diff == count_end - count_start)
            abs_diff = model.NewIntVar(0, num_machines, f"strategy_absdiff_simple_{p}")
            model.AddAbsEquality(abs_diff, diff)
            max_penalty = num_machines * weight2
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_simple_{p}")
            model.Add(pen == weight2 * abs_diff)
        elif strategy == "+":
            diff = model.NewIntVar(-num_machines, num_machines, f"strategy_diff_simple_{p}")
            model.Add(diff == count_start - count_end)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines, f"strategy_posdiff_simple_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = num_machines * weight
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_simple_{p}")
            model.Add(pen == weight * pos_diff)
        elif strategy == "++":
            weight2 = strategy_base_weight * 2
            diff = model.NewIntVar(-num_machines, num_machines + 1, f"strategy_diff_simple_{p}")
            model.Add(diff == (count_start + 1) - count_end)
            zero = model.NewConstant(0)
            pos_diff = model.NewIntVar(0, num_machines + 1, f"strategy_posdiff_simple_{p}")
            model.AddMaxEquality(pos_diff, [diff, zero])
            max_penalty = (num_machines + 1) * weight2
            pen = model.NewIntVar(0, max_penalty, f"strategy_penalty_simple_{p}")
            model.Add(pen == weight2 * pos_diff)

        if pen is not None:
            strategy_penalty_terms[p] = pen
            if settings.APPLY_STRATEGY_PENALTY:
                strategy_objective_terms.append(pen)

    # В SIMPLE используем такой же масштаб штрафа за простой, как и в LONG:
    # без умножения на объём плана, чтобы "1 простой" имел понятный вес.
    downtime_penalty = max(2, settings.KFZ_DOWNTIME_PENALTY)

    # Отладка: если SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX задан, вместо обычной
    # цели минимизации штрафов максимизируем product_counts[idx], чтобы
    # оценить максимально достижимый объём этого продукта при всех текущих
    # ограничениях.
    debug_max_idx = getattr(settings, "SIMPLE_DEBUG_MAXIMIZE_PRODUCT_IDX", None)
    if debug_max_idx is not None and 0 <= debug_max_idx < len(product_counts):
        model.Maximize(product_counts[debug_max_idx])
    else:
        objective_terms: list[cp_model.LinearExpr] = []

        # Линейная цель LS_OVER по пропорциям: суммарный |fact_days - plan_days|
        # по всем продуктам, если APPLY_PROP_OBJECTIVE=True и SIMPLE_USE_PROP_MULT=False.
        if proportion_objective_terms:
            objective_terms.append(sum(proportion_objective_terms))

        # Штраф за переходы: каждое изменение продукта считается примерно как
        # простой, но вес делаем мягче, чем KFZ_DOWNTIME_PENALTY, чтобы пропорции
        # не ломались слишком сильно. Берём половину KFZ (минимум 1).
        if 'total_changes' in locals() and total_changes is not None:
            transitions_weight = max(1, settings.KFZ_DOWNTIME_PENALTY // 2)
            objective_terms.append(total_changes * transitions_weight)

        # В текущей версии SIMPLE не штрафуем отдельный PRODUCT_ZERO: простои
        # будут учитываться внешними метриками и стратегиями. Внутри модели
        # оставляем только стратегические штрафы (по необходимости).
        if settings.APPLY_STRATEGY_PENALTY and strategy_objective_terms:
            objective_terms.append(sum(strategy_objective_terms))

        # Если штрафов нет, минимизируем константу 0, чтобы модель была чисто
        # выполнимостной.
        if objective_terms:
            model.Minimize(sum(objective_terms))
        else:
            model.Minimize(0)

    # Для совместимости с solver_result возвращаем пустые словари для lday/переходов.
    prev_lday: dict = {}
    start_batch: dict = {}
    batch_end_complite: dict = {}
    days_in_batch: dict = {}
    completed_transition: dict = {}
    pred_start_batch: dict = {}
    same_as_prev: dict = {}

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
                 dedicated_machines: list[int] | None = None,
                 product_divs: list[int] | None = None,
                 machine_divs: list[int] | None = None):
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
        # p: (name, qty, id, machine_type, qty_minus, lday, ...)
        base_lday = p[5]
        qty_val = p[1]
        # Временное правило: если lday=0 для "живого" продукта, используем 10.
        if not long_mode and qty_val > 0 and base_lday <= 0:
            base_lday = 10
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
    # Новая договорённость: станки жёстко разделены по цехам (type=1 или 2),
    # а продукты имеют machine_type=1 или 2. Продукт с machine_type>0 можно
    # производить только на машинах с таким же type. machine_type=0 трактуем
    # как "можно на любых" (для совместимости со старыми данными).
    for p in all_products:
        # Индекс 3 в кортеже продукта - это 'machine_type'
        product_machine_type_req = products[p][3]
        if product_machine_type_req > 0:
            for m in all_machines:
                # Индекс 3 в кортеже машины - это 'type'
                machine_type = machines[m][3]
                if machine_type != product_machine_type_req:
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
    simple_mode = getattr(settings, "HORIZON_MODE", "FULL").upper() == "LONG_SIMPLE"

    # Для SIMPLE-модели (LONG_SIMPLE) очищаем влияние чисток при восстановлении
    # расписания: все дни считаются рабочими.
    cleans_set = set(cleans)

    def is_clean_day(m: int, d: int) -> bool:
        if simple_mode:
            return False
        return (m, d) in cleans_set

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

    # В LONG-режиме jobs[m,d] определены по агрегированным дням (1 модельная смена = 2 фактическим).
    # Для восстановления посменного расписания отображаем фактический день d_real в
    # модельный день d_model = d_real // 2.
    for m in range(num_machines):
        m_old = find_machine_id_old(m)
        logger.debug(f"Loom {m_old}")
        for d in range(count_days):
            if not is_clean_day(m, d):
                if long_mode:
                    d_model = d // 2
                    p = solver.value(jobs[m, d_model])
                else:
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
                if long_mode:
                    # LONG: jobs заданы по агрегированным дням (1 модельная смена = 2 фактическим).
                    # Для первого дня используем jobs[m, 0], для последнего — jobs[m, last_model].
                    first_model = 0
                    last_model = (count_days - 1) // 2
                    if not is_clean_day(m, 0) and solver.Value(jobs[m, first_model]) == p:
                        machines_start += 1
                    if not is_clean_day(m, count_days - 1) and solver.Value(jobs[m, last_model]) == p:
                        machines_end += 1
                else:
                    if not is_clean_day(m, 0) and solver.Value(jobs[m, 0]) == p:
                        machines_start += 1
                    if not is_clean_day(m, count_days - 1) and solver.Value(jobs[m, count_days - 1]) == p:
                        machines_end += 1
            if strategy_penalty_terms is not None and p < len(strategy_penalty_terms):
                penalty_strategy = solver.Value(strategy_penalty_terms[p])

        diff_all += penalty_prop
        qty_model = solver.Value(product_counts[p])
        # Для статистики qty интерпретируем в СМЕНАХ:
        # - в LONG: 1 модельный день = 2 фактическим сменам;
        # - в LONG_SIMPLE: 1 модельный день = 3 фактическим сменам;
        # - в FULL/SHORT: qty_model уже считается в фактических сменах.
        if long_mode:
            qty = qty_model * 2
        elif simple_mode:
            qty = qty_model * 3
        else:
            qty = qty_model

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