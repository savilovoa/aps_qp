from ortools.sat.python import cp_model
from pydantic import BaseModel
from .model_loom import DataLoomIn, LoomPlansOut, Machine, Product, Clean, LoomPlan, LoomPlansViewIn, LoomPlansViewOut
import traceback as tr
from ..config import logger, settings
import pandas as pd
from .loom_plan_html import schedule_to_html
from uuid import uuid4

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
        """Print intermediate solutions."""

        def __init__(self, limit: int = -1):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._solution_count = 0
            self._solution_limit = limit

        def on_solution_callback(self):
            self._solution_count += 1
            print(f"Solution {self._solution_count}: {self.objective_value}")
            if self._solution_limit > 0 and self._solution_count > self._solution_limit:
                self.stop_search()

        def solutionCount(self):
            return self._solution_count

    machines_df = pd.DataFrame(data["machines"])
    products_df = pd.DataFrame(data["products"])
    clean_df = pd.DataFrame(data["cleans"])
    schedule = []
    products_schedule = []
    diff_all = 0

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

    (model, jobs, product_counts, proportion_objective_terms, total_products_count, prev_lday, start_batch,
     batch_end_complite, days_in_batch, completed_transition, pred_start_batch, same_as_prev,
     strategy_penalty_terms) = create_model(
        remains=remains, products=products_new, machines=machines_new, cleans=cleans_new,
        max_daily_prod_zero=max_daily_prod_zero, count_days=count_days)


    # solver.parameters.log_search_progress = True
    #solver.parameters.debug_crash_on_bad_hint = True
    #solver.parameters.num_search_workers = 4

    if settings.SOLVER_ENUMERATE:
        sol_printer = NursesPartialSolutionPrinter(settings.SOLVER_ENUMERATE_COUNT)
        solver.parameters.enumerate_all_solutions = True
        status = solver.solve(model, sol_printer)
    else:
        solver.parameters.max_time_in_seconds = settings.LOOM_MAX_TIME
        solver.parameters.num_search_workers = settings.LOOM_NUM_WORKERS
        status = solver.solve(model)

    logger.info(f"Статус решения: {solver.StatusName(status)}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if prev_lday:
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


def create_model(remains: list, products: list, machines: list, cleans: list, max_daily_prod_zero: int, count_days: int):
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

    if remains:
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
        if isinstance(src_root, int) and src_root >= 0 and src_root < len(remains_batches):
            batches = remains_batches[src_root]
            if isinstance(batches, list) and len(batches) > 0:
                # Отбрасываем короткие партии (< 6 смен)
                long_batches = [int(b) for b in batches if isinstance(b, (int, float)) and int(b) >= 6]
                if long_batches:
                    avg_lday = round(sum(long_batches) / len(long_batches))
                    if avg_lday > 0:
                        lday_eff = avg_lday
        ldays.append(lday_eff)

    initial_products = []
    days_to_constrain = []
    for idx, (_, product_idx, m_id, t, remain_day) in enumerate(machines):
        initial_products.append(product_idx)
        days_to_constrain.append(remain_day)

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
        # Добавляем ограничение НЕ МЕНЬШЕ для некоторых продуктов.
        # products[p][1] - требуемое количество (qty)
        # products[p][4] - признак qty_minus (0 - строго, 1 - можно недопланировать)
        # products[p][7] - qty_minus_min (минимальное количество планирования при qty_minus=1)
        if settings.APPLY_QTY_MINUS and products[p][1] > 0:
            qty_minus_flag = products[p][4]
            if qty_minus_flag == 0:
                # Жёсткое ограничение: плановое количество не меньше требуемого qty
                model.Add(product_counts[p] >= products[p][1])
            else:
                # Мягкое ограничение: количество не меньше qty_minus_min (если задано)
                if len(products[p]) > 7:
                    min_qty = products[p][7]
                    if min_qty > 0:
                        model.Add(product_counts[p] >= min_qty)

    # Сумма PRODUCT_ZERO в смену d не более max_daily_prod_zero
    # Количество нулевого продукта по дням
    # И просто количество нулевого продукта
    if settings.APPLY_DOWNTIME_LIMITS:
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

        if days_to_constrain[m] > 0:
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
            # Запрет на 3-й ZERO
            model.add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(completed_transition[m, pred_idx])
            # Запрет на переход в последние 2 дня
            if d >= count_days - 2:
                model.add(jobs[m, d] != PRODUCT_ZERO)

    # не более 1-го простоя (двухдневного перехода) за неделю на каждую машину.
    # При планировании на более долгий период масштабируем ограничение по числу недель.
    if settings.APPLY_DOWNTIME_LIMITS:
        weeks = max(1, count_days // 21)
        max_zero_per_machine = 2 * weeks  # 2 дня простоя на неделю
        for m in range(num_machines):
            prod_zero_on_machine = []
            for d in all_days:
                if not (m, d) in cleans:
                    prod_zero_on_machine.append(product_produced_bools[PRODUCT_ZERO, m, d])
            model.Add(sum(prod_zero_on_machine) <= max_zero_per_machine)

    # ------------ Мягкое ограничение: Пропорции продукции (для продуктов с индексом > 0) ------------
    # Цель: минимизировать отклонение от заданных пропорций
    # Пропорции касаются только продуктов p > 0.
    # Мы хотим, чтобы product_counts[p1] / product_counts[p2] было близко к proportions_input[p1] / proportions_input[p2]
    # Это эквивалентно product_counts[p1] * proportions_input[p2] ~= product_counts[p2] * proportions_input[p1]

    total_products_count = model.NewIntVar(0, num_machines * num_days, "total_products_count")
    model.Add(total_products_count == sum(product_counts[p] for p in range(1, len(products))))


    total_input_quantity = sum(proportions_input)
    total_input_max = max(proportions_input)
    logger.debug(f"total_input_quantity={total_input_quantity}")
    proportion_objective_terms = []

    for p in range(1, len(products)):  # Skip p == 0
        logger.debug(f"proportions_input[{p}]={proportions_input[p]}")

        # product_counts[p] * total_input_quantity
        term1_expr = model.NewIntVar(0, num_machines * num_days * total_input_quantity,
                                     f"term1_{p}")
        model.AddMultiplicationEquality(term1_expr, [product_counts[p], total_input_quantity])

        # total_products_count * proportions_input[p1_idx]
        term2_expr = model.NewIntVar(0, cp_model.INT32_MAX, f"term2_{p}")
        model.AddMultiplicationEquality(term2_expr, [total_products_count,
                                                     model.NewConstant(proportions_input[p])])

        # diff = term1_expr - term2_expr
        diff_var = model.NewIntVar(-cp_model.INT32_MAX, cp_model.INT32_MAX, f"diff_{p}")
        model.Add(diff_var == (term1_expr - term2_expr))
        abs_diff_var = model.NewIntVar(0, cp_model.INT32_MAX, f"abs_diff_{p}")
        model.AddAbsEquality(abs_diff_var, diff_var)
        proportion_objective_terms.append(abs_diff_var)

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
            strategy_objective_terms.append(pen)

    downtime_penalty = round(total_input_max * settings.KFZ_DOWNTIME_PENALTY)
    if downtime_penalty < 2:
        downtime_penalty = 2

    model.Minimize(
        sum(proportion_objective_terms)
        + product_counts[PRODUCT_ZERO] * downtime_penalty
        + sum(strategy_objective_terms)
    )


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
                db_v = solver.value(days_in_batch[m, d])
                plday = solver.value(prev_lday[m, d])
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
        penalty_prop = 0 if p == 0 else solver.Value(proportion_objective_terms[p - 1])
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
        products_schedule.append(
            {
                "product_idx": p_old,
                "qty": qty,
                "penalty": penalty_prop,
                "penalty_strategy": penalty_strategy,
                "machines_start": machines_start,
                "machines_end": machines_end,
            }
        )
        logger.debug(
            f"  Продукт {p_old}({p}): {qty} единиц, машины {machines_start}-{machines_end}, "
            f"штраф пропорций {penalty_prop}, штраф стратегии {penalty_strategy}"
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