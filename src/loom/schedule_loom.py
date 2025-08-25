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

def ProductsModelToArray(products: list[Product]) -> list[(str, int, str, int, int)]:
    result = []
    first = True
    for item in products:
        if first:
            first = False
            if item.qty > 0:
                raise "Первый элемент продукции должен быть сменой артикула, т.е. количество плана = 0"
        result.append((item.name, item.qty, item.id, item.machine_type, item.qty_minus))
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

        result_calc = schedule_loom_calc(remains=remains, products=products, machines=machines, cleans=cleans,
                                    max_daily_prod_zero=max_daily_prod_zero, count_days=count_days, data=data)

        if result_calc["error_str"] == "" and result_calc["status"] != cp_model.INFEASIBLE:
            machines_view = [name for (name, product_idx,  id, type, remain_day) in machines]
            products_view = [name for (name, qty, id, machine_type, qm) in products]
            title_text = f"{result_calc['status_str']} оптимизационное значение {result_calc['objective_value']}"

            res_html = schedule_to_html(machines=machines_view, products=products_view, schedules=result_calc["schedule"],
                                        days=days, dt_begin=DataIn.dt_begin, title_text=title_text)
            id_html = str(uuid4())


            schedule = [LoomPlan(machine_idx=s["machine_idx"], day_idx=s["day_idx"], product_idx=s["product_idx"])
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

    schedule_init, objective_value, deviation_proportion, count_product_zero = (
        create_schedule_init(data["machines"], data["products"], data["cleans"], count_days, max_daily_prod_zero))

    # schedule = [{"machine_idx": m, "day_idx": d, "product_idx": schedule_init[m][d]}
    #             for m in range(len(machines))
    #             for d in range(count_days)]
    # result = {"status": 20, "status_str": "INIT", "schedule": schedule,
    #           "products": [], "objective_value": objective_value,
    #           "proportion_diff": int(deviation_proportion), "error_str": ""}
    #
    # return result

    machines_new = machines.copy()
    products_new = products.copy()

    #machines_full = update_data_for_schedule_init(machines_new, products_new, cleans, count_days, schedule_init)

    solver = cp_model.CpSolver()

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

    (model, jobs, product_counts, proportion_objective_terms, total_products_count, prev_lday, start_batch,
     batch_end_complite, days_in_batch, completed_transition, pred_start_batch, same_as_prev) = create_model(remains=remains, products=products_new, machines=machines_new, cleans=cleans,
                                        max_daily_prod_zero=max_daily_prod_zero, count_days=count_days, schedule_init=schedule_init)


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
                for d in range(21):
                    p = solver.value(days_in_batch[m, d])
                    s = s + str(p) +","
                    p = solver.value(batch_end_complite[m, d])
                    s_bc = s_bc + str(p) + ","
                logger.info(f"days_in_batch {m}:       [{s}]")
                logger.info(f"batch_end_complite {m}:  [{s_bc}]")

                s = ""
                s_sb = ""
                s_ct = ""
                s_sp =""
                for d in range(1, 21):
                    p = solver.value(prev_lday[m, d])
                    s = s + str(p) +","
                    p = solver.value(start_batch[m, d])
                    s_sb = s_sb + str(p) + ","
                    p = solver.value(completed_transition[m, d])
                    s_ct = s_ct + str(p) + ","
                    p = solver.value(same_as_prev[m, d])
                    s_sp = s_sp + str(p) + ","
                logger.info(f"same_as_prev {m}:          [{s_sp}]")
                logger.info(f"completed_transition {m}:  [{s_ct}]")
                logger.info(f"prev_lday {m}:             [{s}]")
                logger.info(f"start_batch {m}:           [{s_sb}]")

                s_sb = ""
                for d in range(2, 21):
                    p = solver.value(pred_start_batch[m, d])
                    s_sb = s_sb + str(p) + ","
                logger.info(f"pred_start_batch {m}:        [{s_sb}]")


        if proportion_objective_terms:
            logger.info(f"Минимальное значение функции цели (сумма абс. отклонений пропорций): "
                        f"{solver.ObjectiveValue()}")

        schedule, products_schedule, diff_all = solver_result(solver, status, machines, products, machines_new,
                                                              products_new, cleans, count_days, [],
                                                              proportion_objective_terms, product_counts, jobs, total_products_count)
    else:
        schedule = []
        products_schedule = []
        diff_all = 0

    logger.info(solver.ResponseStats())  # Основные статистические данные
    result = {"status": int(status), "status_str": solver.StatusName(status), "schedule": schedule,
              "products": products_schedule, "objective_value": int(solver.ObjectiveValue()),
              "proportion_diff": int(diff_all), "error_str": ""}

    return result


def create_model(remains: list, products: list, machines: list, cleans: list, max_daily_prod_zero: int, count_days: int,
                 schedule_init: list = None):
    # products: [ # ("idx, "name", "qty", "id", "machine_type", "qty_minus")
    #     ("", 0, "", 0),
    #     ("ст87017t3", 42, "7ec17dc8-f3bd-4384-9738-7538ab3dc315", 0, 1),
    #     ("ст87416t1", 15, "9559e2e8-6e72-41f8-9dba-08aab5463623", 0, 1),
    #     ("ст2022УИСt4", 4, "cd825c90-aa80-4b95-9f81-2486b871bf94", 0, 0)
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
    ldays = [p[5] for p in products]
    initial_products = []
    days_to_constrain = []
    for idx, (_, product_idx, m_id, t, remain_day) in enumerate(machines):
        initial_products.append(product_idx)
        days_to_constrain.append(remain_day)

    model = cp_model.CpModel()

    jobs = {}
    work_days = []
    # Значение для отображения чистки в итоговом расписании
    for m in range(num_machines):
        for d in range(num_days):
            if (m, d) not in cleans:
                work_days.append((m, d))
                # Домен переменной: от 0 до num_products - 1
                jobs[(m, d)] = model.new_int_var(0, num_products - 1, f"job_{m}_{d}")

    PRODUCT_ZERO = 0  # Индекс "особенной" продукции

    # ------------ Подсчет общего количества каждого продукта ------------
    # Вспомогательные булевы переменные: product_produced[p, m, d] истинно, если продукт p производится на машине m в день d
    product_produced_bools = {}

    max_lday = max(ldays) if ldays else 1
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
        # Добавляем условие НЕ МЕНЬШЕ для некоторых продуктов
        if products[p][4] == 0 and products[p][1] > 0:
            model.Add(product_counts[p] >= products[p][1])

    # Сумма PRODUCT_ZERO в смену d не более max_daily_prod_zero
    # Количество нулевого продукта по дням
    # И просто количество нулевого продукта
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

    prev_lday = {}
    for m in all_machines:
        for d in all_days:
            prev_lday[m, d] = model.NewIntVar(0, max_lday, f'prev_lday_m{m}_d{d}')
            model.AddElement(jobs[m, d], ldays, prev_lday[m, d])

    batch_end_complite = {}

    # Ограничения ПЕРЕХОДА
    # Переменные для отслеживания завершения двухдневного перехода
    completed_transition = {}
    is_not_zero = {}
    same_as_prev = {}
    prev_is_not_zero = {}
    prev2_is_not_zero = {}
    two_day_zero = {}
    start_batch = {}

    remain_day = [0 for _ in range(num_machines)]
    # Ограничение для первого дня (d=0)
    for m in range(num_machines):
        initial_product = initial_products[m]
        is_initial_product = model.NewBoolVar(f"is_initial_product_{m}_0")
        model.Add(jobs[m, 0] == initial_product).OnlyEnforceIf(is_initial_product)
        model.Add(jobs[m, 0] != initial_product).OnlyEnforceIf(is_initial_product.Not())

        is_not_zero[m, 0] = model.NewBoolVar(f"is_not_zero_{m}_0")
        model.Add(jobs[m, 0] == PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 0].Not())
        model.Add(jobs[m, 0] != PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 0])

        product_lday = ldays[initial_product]
        batch_end_complite[m, 0] = model.NewBoolVar(f"batch_end_complite_m{m}_d0")

        if days_to_constrain[m] > 0:
            remain_day[m] += 1
            model.Add(jobs[m, 0] == initial_product)
            # выставляем начальное значение остатка партии
            start_val = product_lday - days_to_constrain[m] + 1
            model.Add(days_in_batch[m, 0] == start_val)
        elif initial_product == 0:
            model.Add(days_in_batch[m, 0] == 1).OnlyEnforceIf([is_not_zero[m, 0]])
            model.Add(jobs[m, 0] != PRODUCT_ZERO)
        else:

            # Первый день: либо начальный продукт, либо PRODUCT_ZERO
            model.AddBoolOr([is_initial_product, is_not_zero[m, 0].Not()])
            # Первый день: либо начало партии, либо 0, если переход
            model.Add(days_in_batch[m, 0] == 1).OnlyEnforceIf([is_initial_product, is_not_zero[m, 0]])
            model.Add(days_in_batch[m, 0] == 0).OnlyEnforceIf(is_not_zero[m, 0].Not())

        # Устанавливаем completed_transition для дня 0
        completed_transition[m, 0] = model.NewBoolVar(f"completed_transition_{m}_0")
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
            completed_transition[m, d] = model.NewBoolVar(f"completed_transition_{m}_{d}")
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
            # model.AddBoolAnd([same_as_prev[m, d], batch_end_complite[m, pred_idx]]).OnlyEnforceIf(
            #     pred_start_batch[m, d])
            # model.AddBoolOr([same_as_prev[m, d].Not(), batch_end_complite[m, pred_idx].Not()]).OnlyEnforceIf(
            #     pred_start_batch[m, d].Not())
            #
            start_batch[m, d] = model.NewBoolVar(f"start_batch_m{m}_d{d}")
            # model.AddBoolOr([pred_start_batch[m, d], completed_transition[m, d]]).OnlyEnforceIf(
            #     start_batch[m, d])
            # model.AddBoolAnd([pred_start_batch[m, d].Not(), completed_transition[m, d].Not()]).OnlyEnforceIf(
            #     start_batch[m, d].Not())
            #
            # model.Add(jobs[m, d] == jobs[m, pred_idx]).OnlyEnforceIf([batch_end_complite[m, pred_idx].Not(), prev_is_not_zero[m, d]])


            # ### НАЧАЛО НОВОГО БЛОКА: Ограничение на повышение индекса продукта ###
            # Это ограничение срабатывает только в день `d`, когда завершился двухдневный переход,
            # что определяется переменной completed_transition[m, d].

            # 1. Находим индекс рабочего дня перед началом перехода.
            #    Переход занимал дни `pred_idx` и `d`. Ищем день до `pred_idx`.
            day_before_transition_start = pred_idx - 1
            while day_before_transition_start >= 0 and (m, day_before_transition_start) in cleans:
                day_before_transition_start -= 1

            # 2. Применяем ограничение, только если такой день существует в расписании.
            if day_before_transition_start >= 0:
                # Переменная, указывающая на продукт до начала перехода.
                product_before = jobs[(m, day_before_transition_start)]
            else:
                product_before = initial_product

            # 3. Вводим вспомогательную переменную. Она будет истинной, если
            #    продукт до перехода не был PRODUCT_ZERO. Это нужно, чтобы
            #    избежать сравнения, если до этого уже был простой.
            product_before_is_not_zero = model.NewBoolVar(f"prod_before_not_zero_{m}_{d}")
            model.Add(product_before != PRODUCT_ZERO).OnlyEnforceIf(product_before_is_not_zero)
            model.Add(product_before == PRODUCT_ZERO).OnlyEnforceIf(product_before_is_not_zero.Not())

            # 4. Устанавливаем само ограничение.
            #    Оно должно сработать, только если (А) переход завершен И (Б) продукт до перехода не был нулевым.
            #    Существующее ограничение `model.add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(completed_transition[m, d])`
            #    уже гарантирует, что jobs[m, d] не будет нулем, если переход завершен.
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

    # не более 1-го простоя за неделю
    for m in range(num_machines):
        prod_zero_on_machine = []
        for d in all_days:
            if not (m, d) in cleans:
                prod_zero_on_machine.append(product_produced_bools[PRODUCT_ZERO, m, d])
        model.Add(sum(prod_zero_on_machine) <= 2)

    # ### НОВОЕ: Добавление начального расписания как подсказки (hint) ###
    if schedule_init:
        for m in all_machines:
            for d in all_days:
                # Подсказку можно добавить только для существующей переменной (т.е. не в день чистки)
                if (m, d) in work_days:
                    initial_product_idx = schedule_init[m][d]
                    # Игнорируем значения чистки (-2) и другие некорректные
                    if initial_product_idx and initial_product_idx >= 0:
                        model.AddHint(jobs[(m, d)], initial_product_idx)

                        # 2. Подсказки для вспомогательных переменных
                        # 'product_produced_bools'
                        for p in all_products:
                            hint_value = 1 if p == initial_product_idx else 0
                            model.AddHint(product_produced_bools[p, m, d], hint_value)

                        # 'is_not_zero'
                        is_not_zero_hint = 1 if initial_product_idx != PRODUCT_ZERO else 0
                        model.AddHint(is_not_zero[m, d], is_not_zero_hint)

                        # 'same_as_prev' and 'completed_transition'
                        if d > 0 and schedule_init[m][d-1] and schedule_init[m][d-1] > 0:
                            d_prev = d - 1
                            prev_product_id_hint = schedule_init[m][d_prev]

                            # # 'same_as_prev'
                            # same_as_prev_hint = 1 if initial_product_idx == prev_product_id_hint else 0
                            # model.AddHint(same_as_prev[m, d], same_as_prev_hint)

                            if d > 1 and schedule_init[m][d-2] and schedule_init[m][d-2] > 0:
                                d_prev = d - 1
                                d_prev2 = d - 2
                                prev_product_id_hint = schedule_init[m][d_prev]
                                prev2_product_id_hint = schedule_init[m][d_prev2]

                                # 'completed_transition'
                                completed_transition_hint = 1 if (
                                            prev_product_id_hint == PRODUCT_ZERO and prev2_product_id_hint == PRODUCT_ZERO) else 0
                                model.AddHint(completed_transition[m, d], completed_transition_hint)



        # ### END: ADDING INITIAL STATE (HINTS) ###

    # ------------ Мягкое ограничение: Пропорции продукции (для продуктов с индексом > 0) ------------
    # Цель: минимизировать отклонение от заданных пропорций
    # Пропорции касаются только продуктов p > 0.
    # Мы хотим, чтобы product_counts[p1] / product_counts[p2] было близко к proportions_input[p1] / proportions_input[p2]
    # Это эквивалентно product_counts[p1] * proportions_input[p2] ~= product_counts[p2] * proportions_input[p1]

    total_products_count = model.NewIntVar(0, num_machines * num_days, "total_products_count")
    model.Add(total_products_count == sum(product_counts[p] for p in range(1, len(products))))

    total_input_quantity = sum(proportions_input)
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

    downtime_penalty = round(sum(proportions_input)/len(work_days))
    if downtime_penalty < 2:
        downtime_penalty = 2

    model.Minimize(sum(proportion_objective_terms) + product_counts[PRODUCT_ZERO] * downtime_penalty)


    return (model, jobs, product_counts, proportion_objective_terms, total_products_count, prev_lday, start_batch,
            batch_end_complite, days_in_batch, completed_transition, pred_start_batch, same_as_prev)


def create_schedule_init(machines, products, cleans, count_days, max_daily_prod_zero):
    machines_df = pd.DataFrame(machines)
    products_df = pd.DataFrame(products)
    days = [item for item in range(count_days)]
    num_machines = len(machines_df)
    num_products = len(products_df)

    schedule = [[None for _ in range(count_days)] for _ in range(num_machines)]

    # Предварительно заполняем дни очистки (cleans) - это жесткие ограничения
    for clean in cleans:
        machine_idx = clean['machine_idx']
        day_idx = clean['day_idx']
        if 0 <= machine_idx < num_machines and 0 <= day_idx < count_days:
            schedule[machine_idx][day_idx] = -2

    # Подсчет рабочих дней без чисток
    work_days = count_days * len(machines) - sum(
        1 for clean in cleans
    )
    # Функция для проверки возможности установки перехода (prod_zero) в день
    def can_place_zero(day, zeros_per_day, max_daily_prod_zero):
        return zeros_per_day.get(day, 0) < max_daily_prod_zero

    # Счетчик переходов по дням
    zeros_per_day = {day: 0 for day in range(count_days)}

    # --- Шаг 3: Добавление колонки с индексом объекта ---
    machines_df.reset_index(inplace=True)
    machines_df.rename(columns={'index': 'original_index'}, inplace=True)

    # --- Шаг 4: Добавление колонки 'product_qty' ---
    product_quantities = products_df['qty']
    machines_df['product_qty'] = machines_df['product_idx'].map(product_quantities)

    # Правим количество пропорционально
    # Считаем мин и мак чисток
    next_count_min = 0
    for idx, machine in machines_df.iterrows():
        if machine['product_qty'] < count_days / 2:
            next_count_min += 1
    next_count_max = round(count_days / 2)
    if next_count_max < next_count_min:
        next_count = next_count_max
    else:
        next_count = next_count_min * round((next_count_max - next_count_min) / 2)
    work_days = work_days - next_count * 2
    count_qty = 0
    for qty in products_df[products_df['idx'] > 0]['qty']:
        count_qty += qty
    # Считаем коэф увеличения/уменьшения
    kf_count = 0.9 * work_days / count_qty
    for product_idx in range(1, num_products):
        qty = products_df.at[product_idx, 'qty']
        if qty <= 0:
            continue
        qty = round(qty * kf_count)
        products_df.at[product_idx, 'qty'] = qty
    product_quantities = products_df['qty']
    machines_df['product_qty'] = machines_df['product_idx'].map(product_quantities)

    # --- Шаг 5: Добавление колонки 'day_remains' ---
    machines_df['day_remains'] = count_days
    for machine_idx in range(num_machines):
        clean_days = sum(1 for clean in cleans if clean['machine_idx'] == machine_idx)
        machines_df.at[machine_idx, 'day_remains'] -= clean_days

    # --- Первая часть алгоритма ---
    for machine_idx in range(num_machines):
        product_idx = machines_df.at[machine_idx, 'product_idx']
        machines_df.at[machine_idx, 'product_qty'] = products_df.at[product_idx, 'qty']
        qty_needed = products_df.at[product_idx, 'qty']
        if qty_needed >= count_days / 2:
            days_to_plan = min(int(qty_needed), machines_df.at[machine_idx, 'day_remains'])
            day_idx = 0
            days_planned = 0
            while days_planned < days_to_plan and day_idx < count_days:
                if schedule[machine_idx][day_idx] is None:  # Проверяем, что день не занят чисткой
                    schedule[machine_idx][day_idx] = int(product_idx)
                    days_planned += 1
                    machines_df.at[machine_idx, 'day_remains'] -= 1
                    products_df.at[product_idx, 'qty'] -= 1
                    machines_df.at[machine_idx, 'product_qty'] -= 1
                day_idx += 1

            # Если осталось менее 3 дней до конца, продолжаем планировать тот же продукт
            if day_idx >= count_days - 2:
                while day_idx < count_days:
                    if schedule[machine_idx][day_idx] is None:
                        schedule[machine_idx][day_idx] = int(product_idx)
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[machine_idx, 'product_qty'] -= 1
                        machines_df.at[machine_idx, 'day_remains'] -= 1
                    day_idx += 1

            # Планируем переход (2 дня с product_idx = 0)
            if days_planned >= 1 and day_idx + 1 < count_days:
                zero_days = 0
                start_day = day_idx
                while zero_days < 2 and day_idx < count_days:
                    if schedule[machine_idx][day_idx] is None:
                        if can_place_zero(day_idx, zeros_per_day, max_daily_prod_zero):
                            schedule[machine_idx][day_idx] = 0
                            zeros_per_day[day_idx] = zeros_per_day.get(day_idx, 0) + 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                            zero_days += 1
                        else:
                            # Если нельзя поставить переход, продолжаем планировать тот же продукт
                            schedule[machine_idx][day_idx] = int(product_idx)
                            products_df.at[product_idx, 'qty'] -= 1
                            machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                    day_idx += 1
                # Если осталось менее 3 дней до конца, продолжаем планировать тот же продукт
                if day_idx >= count_days - 2:
                    while day_idx < count_days:
                        if schedule[machine_idx][day_idx] is None:
                            schedule[machine_idx][day_idx] = int(product_idx)
                            products_df.at[product_idx, 'qty'] -= 1
                            machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                        day_idx += 1


    # --- Вторая часть алгоритма ---
    def schedule_remaining_days(machine_type_filter=None):
        # Сортировка машин
        machines_to_schedule = machines_df[machines_df['day_remains'] > 0].copy()
        if machine_type_filter is not None:
            machines_to_schedule = machines_to_schedule[machines_to_schedule['type'] == machine_type_filter]
        machines_to_schedule = machines_to_schedule.sort_values(
            by=['type', 'product_qty', 'day_remains'], ascending=[False, True, False]
        )

        # Сортировка продуктов
        products_to_schedule = products_df[products_df['qty'] > 0].copy()
        if machine_type_filter is not None:
            products_to_schedule = products_to_schedule[products_to_schedule['machine_type'] == machine_type_filter]
        products_to_schedule = products_to_schedule.sort_values(
            by=['machine_type', 'qty'], ascending=[False, False]
        )

        while not machines_to_schedule.empty and not products_to_schedule.empty:
            product_idx = products_to_schedule.iloc[0]['idx']
            qty_needed = products_to_schedule.iloc[0]['qty']
            product_machine_type = products_to_schedule.iloc[0]['machine_type']

            # Проверяем, есть ли машина с совпадающим начальным продуктом
            matching_machines = machines_to_schedule[machines_to_schedule['product_idx'] == product_idx]
            if not matching_machines.empty:
                machine_idx = matching_machines.iloc[0]['original_index']
                day_remains = matching_machines.iloc[0]['day_remains']
            else:
                # Выбираем первую машину из отсортированного списка
                machine_idx = machines_to_schedule.iloc[0]['original_index']
                day_remains = machines_to_schedule.iloc[0]['day_remains']


            machine_type = machines_df[machines_df['original_index'] == machine_idx]['type'].iloc[0]
            # Проверяем совместимость типов
            if product_machine_type == 1 and machine_type != 1:
                products_to_schedule = products_to_schedule.iloc[1:]
                continue

            # Находим первый свободный день
            start_day = 0
            for day in range(count_days):
                if schedule[machine_idx][day] is not None:
                    start_day = day + 1
                else:
                    break

            # Если начальный продукт совпадает, планируем его до нужного количества
            if machines_df[machines_df['original_index'] == machine_idx]['product_idx'].iloc[0] == product_idx:
                days_planned = 0
                day_idx = start_day
                for day in range(start_day, count_days):
                    if schedule[machine_idx][day] is None:
                        if products_df.at[product_idx, 'qty'] > 0:
                            schedule[machine_idx][day] = int(product_idx)
                            products_df.at[product_idx, 'qty'] -= 1
                            machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                            day_idx += 1
                        else:
                            break
                    else:
                        day_idx += 1

                # Планируем переход (2 дня с product_idx = 0)
                zero_days = 0
                while zero_days < 2 and day_idx < count_days:
                    if schedule[machine_idx][day_idx] is None:
                        if can_place_zero(day_idx, zeros_per_day, max_daily_prod_zero):
                            # Если осталось менее 3 дней до конца, продолжаем планировать тот же продукт
                            if day_idx >= count_days - 2:
                                while day_idx < count_days:
                                    if schedule[machine_idx][day_idx] is None:
                                        schedule[machine_idx][day_idx] = int(product_idx)
                                        products_df.at[product_idx, 'qty'] -= 1
                                        machines_df.at[machine_idx, 'product_qty'] -= 1
                                        machines_df.at[machine_idx, 'day_remains'] -= 1
                                    day_idx += 1
                                machines_to_schedule = machines_to_schedule[
                                    machines_to_schedule['original_index'] != machine_idx]
                            else:
                                schedule[machine_idx][day_idx] = 0
                                zeros_per_day[day_idx] = zeros_per_day.get(day_idx, 0) + 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                                zero_days += 1
                                day_idx += 1
                                schedule[machine_idx][day_idx] = 0
                                zeros_per_day[day_idx] = zeros_per_day.get(day_idx, 0) + 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                                zero_days += 1
                        else:
                            # Планируем начальный продукт, если переход невозможен
                            schedule[machine_idx][day_idx] = int(product_idx)
                            if products_df.at[initial_product_idx, 'qty'] > 0:
                                products_df.at[initial_product_idx, 'qty'] -= 1
                                machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                        day_idx += 1
                    else:
                        day_idx += 1


            elif day_remains == count_days or day_remains == count_days - 1:
                # Планируем начальный продукт, если он есть в products_df
                initial_product_idx = machines_df[machines_df['original_index'] == machine_idx]['product_idx'].iloc[0]
                initial_qty = products_df[products_df['idx'] == initial_product_idx]['qty']
                day_idx = start_day
                if not initial_qty.empty and initial_qty.iloc[0] > 0:
                    days_planned = 0
                    while day_idx < count_days and days_planned < initial_qty.iloc[0]:
                        if schedule[machine_idx][day_idx] is None:
                            schedule[machine_idx][day_idx] = int(initial_product_idx)
                            products_df.at[initial_product_idx, 'qty'] -= 1
                            machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                            days_planned += 1
                        day_idx += 1
                # Планируем переход (2 дня с product_idx = 0)
                zero_days = 0
                while zero_days < 2 and day_idx < count_days:
                    if schedule[machine_idx][day_idx] is None:
                        if can_place_zero(day_idx, zeros_per_day, max_daily_prod_zero):
                            if day_idx >= count_days - 2:
                                while day_idx < count_days:
                                    if schedule[machine_idx][day_idx] is None:
                                        schedule[machine_idx][day_idx] = int(product_idx)
                                        products_df.at[product_idx, 'qty'] -= 1
                                        machines_df.at[machine_idx, 'product_qty'] -= 1
                                        machines_df.at[machine_idx, 'day_remains'] -= 1
                                    day_idx += 1
                                machines_to_schedule = machines_to_schedule[
                                    machines_to_schedule['original_index'] != machine_idx]
                            else:
                                schedule[machine_idx][day_idx] = 0
                                zeros_per_day[day_idx] = zeros_per_day.get(day_idx, 0) + 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                                zero_days += 1
                                day_idx += 1
                                schedule[machine_idx][day_idx] = 0
                                zeros_per_day[day_idx] = zeros_per_day.get(day_idx, 0) + 1
                                machines_df.at[machine_idx, 'day_remains'] -= 1
                                zero_days += 1
                                day_idx += 1
                        else:
                            schedule[machine_idx][day_idx] = int(initial_product_idx)
                            products_df.at[initial_product_idx, 'qty'] -= 1
                            machines_df.at[machine_idx, 'product_qty'] -= 1
                            machines_df.at[machine_idx, 'day_remains'] -= 1
                            day_idx += 1

                    else:
                        day_idx += 1
                start_day = day_idx

                for day in range(start_day, count_days):
                    if schedule[machine_idx][day] is None:
                        schedule[machine_idx][day] = int(product_idx)
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[machine_idx, 'product_qty'] -= 1
                        machines_df.at[machine_idx, 'day_remains'] -= 1
                        day_idx += 1
                    else:
                        day_idx += 1

                machines_to_schedule = machines_to_schedule[machines_to_schedule['original_index'] != machine_idx]

            else:
                # Если машина уже частично распределена
                start_day = count_days - day_remains
                # Планируем выбранный продукт до конца дней
                for day in range(start_day, count_days):
                    if schedule[machine_idx][day] is None:
                        schedule[machine_idx][day] = int(product_idx)
                        products_df.at[product_idx, 'qty'] -= 1
                        machines_df.at[machine_idx, 'product_qty'] -= 1
                        machines_df.at[machine_idx, 'day_remains'] -= 1

                machines_to_schedule = machines_to_schedule[machines_to_schedule['original_index'] != machine_idx]


            # Обновляем qty для продукта
            products_to_schedule.iloc[0, products_to_schedule.columns.get_loc('qty')] -= min(
                qty_needed, sum(1 for day in range(count_days) if schedule[machine_idx][day] == product_idx)
            )
            # Удаляем продукт, если qty <= 0
            products_to_schedule = products_to_schedule[products_to_schedule['qty'] > 0]
            # Пересортировка
            machines_to_schedule = machines_to_schedule.sort_values(
                by=['type', 'product_qty', 'day_remains'], ascending=[False, True, False]
            )
            products_to_schedule = products_to_schedule.sort_values(
                by=['machine_type', 'qty'], ascending=[False, False]
            )

    # Отдельное распределение для продуктов типа 1
    schedule_remaining_days(machine_type_filter=1)
    # Распределение для всех продуктов
    schedule_remaining_days()

    # --- Подведение итогов ---

    # proportions_input - массив qty продуктов с индексом больше нуля
    proportions_input = products_df[products_df['idx'] > 0]['qty'].values
    total_work_days = sum(
        1 for machine in schedule for day in machine if day not in [-2, 0]
    )

    # Коэффициент для штрафов
    kf_downtime_penalty = round(0.1 * sum(proportions_input) / len([d for d in schedule for day in d if day != -2]))
    if kf_downtime_penalty < 10:
        kf_downtime_penalty = 10

    # Подсчет отклонений пропорций
    proportion_objective_terms = []
    for product_idx in products_df[products_df['idx'] > 0]['idx']:
        planned_qty = sum(
            1 for machine in schedule for day in machine if day == product_idx
        )
        required_qty = products_df[products_df['idx'] == product_idx]['qty'].iloc[0]
        proportion = planned_qty / total_work_days if total_work_days > 0 else 0
        expected_proportion = required_qty / sum(proportions_input) if sum(proportions_input) > 0 else 0
        proportion_objective_terms.append(abs(round(proportion - expected_proportion)))

    # Подсчет переходов
    count_product_zero = sum(1 for machine in schedule for day in machine if day == 0)

    # Итоговый показатель
    objective_value = sum(proportion_objective_terms) + count_product_zero * kf_downtime_penalty
    deviation_proportion = sum(proportion_objective_terms)

    return schedule, objective_value, deviation_proportion, count_product_zero


def update_data_for_schedule_init(machines: list, products: list, cleans: list, count_days: int, schedule_init: list):
    num_machines = len(machines)
    num_products = len(products)
    logger.debug(f"Первичная проверка на полные машины")
    machines_full = []
    products_for_del = []
    products_count = [0 for _ in range(num_products)]
    for m in range(num_machines):
        for d in range(count_days):
            if schedule_init[m][d] and schedule_init[m][d] > 0:
                products_count[schedule_init[m][d]] += 1

    for m in range(num_machines):
        p = schedule_init[m][0]
        if p == None or p <= 0 or products[p][1] < count_days / 2:
            continue
        full_p = True
        for d in range(1, count_days):
            if not schedule_init[m][d] or schedule_init[m][d] != p:
                full_p = False
                break
        if not full_p:
            continue
        machines_full.append((m, p, machines[m][2]))
        if products_count[p] - count_days > 0:
            logger.debug(f"  уменьшаем индекс {p}  и планируем удалять машину {m}")
            products[p] = (products[p][0], products_count[p] - count_days, products[p][2], products[p][3])
        else:
            p_exist = False
            for m1 in range(num_machines):
                if m1 != m and machines[m1][1] == p:
                    p_exist = True
            if p_exist:
                logger.debug(f"  обнуляем индекс {p}  и планируем удалять  машину {m}")
                products[p] = (products[p][0], 0, products[p][2], products[p][3])
            else:
                products_for_del.append((p, products[p][2]))
                logger.debug(f"  планируем удалять продукт {p} и машину {m}")

    machine_del = [m for m, p, id in machines_full]
    machine_del.sort(reverse=True)
    for m_old in machine_del:
        for m in range(m_old + 1, len(machines)):
            for c in range(len(cleans)):
                if cleans[c][0] == m:
                    cleans[c] = (m - 1, cleans[c][1])
        machines.pop(m_old)
        schedule_init.pop(m_old)
        logger.debug(f"  удаляем машину {m_old}")

    product_del = [p for p, id in products_for_del]
    product_del.sort(reverse=True)
    for p_old in product_del:
        for p in range(p_old + 1, len(products)):
            for m in range(len(machines)):
                if machines[m][1] == p:
                    machines[m] = (machines[m][0], p - 1, machines[m][2], machines[m][3])
        products.pop(p_old)
        logger.debug(f"  удаляем индекс {p_old}")

    return machines_full

def solver_result(solver, status, machines_old, products_old, machines, products, cleans, count_days, machines_full,
                  proportion_objective_terms, product_counts, jobs, total_products_count):

    def find_machine_id_old(machine_idx: int):
        for i, machine in enumerate(machines_old):
            if machine[2] == machines[machine_idx][2]:
                return i
        raise f"Не нашли id машины {machines[machine_idx][2]}"

    def find_product_id_old(product_idx: int):
        for i, product in enumerate(products_old):
            if product[2] == products[product_idx][2]:
                return i
        raise f"Не нашли id продукта {products[product_idx][2]}"

    num_products = len(products)
    num_machines = len(machines)

    diff_all = 0
    schedule = []
    products_schedule = []
    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return schedule, products_schedule, diff_all
    for m in range(num_machines):
        m_old = find_machine_id_old(m)
        logger.debug(f"Loom {m_old}")
        for d in range(count_days):
            if not (m, d) in cleans:
                p = solver.value(jobs[m, d])
                p_old = find_product_id_old(p)
            else:
                p_old = None
            schedule.append({"machine_idx": m_old, "day_idx": d, "product_idx": p_old})
            logger.debug(f"  Day {d} works  {p_old}")

    logger.debug("\nОбщее количество произведенной продукции:")
    logger.debug(f"\n{solver.value(total_products_count)}")
    for p in range(num_products):
        diff = 0 if p == 0 else solver.value(proportion_objective_terms[p - 1])
        diff_all += diff
        qty = solver.Value(product_counts[p])
        p_old = find_product_id_old(p)
        products_schedule.append({"product_idx": p_old, "qty": qty, "penalty": diff})
        logger.debug(f"  Продукт {p_old}({p}): {qty} единиц, штраф пропорций {diff}")

    for m, p, id in machines_full:
        logger.debug(f"Loom {m}")
        for d in range(count_days):
            schedule.append({"machine_idx": m, "day_idx": d, "product_idx": p})
            logger.debug(f"  Day {d} works  {p}")

    return schedule, products_schedule, diff_all


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