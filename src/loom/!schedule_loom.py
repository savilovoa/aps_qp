from ortools.sat.python import cp_model
from .model_loom import DataLoomIn, LoomPlansOut, Machine, Product, ProductPlan, LoomPlan, DayZero
import traceback as tr
from ..config import logger, settings

def MachinesModelToArray(machines: list[Machine]) -> list[(str, int)]:
    result = []
    idx = 0
    for item in machines:
        if item.idx != idx:
            break
        result.append((item.name, item.product_idx))
        idx += 1
    return result

def ProductsModelToArray(products: list[Product]) -> list[(str, int, [(int, int)])]:
    result = []
    first = True
    for item in products:
        if first:
            first = False
            if item.qty > 0:
                raise "Первый элемент продукции должен быть сменой артикула, т.е. количество плана = 0"
        result.append((item.name, item.qty, []))
    return result


def schedule_loom_calc_model(DataIn: DataLoomIn) -> LoomPlansOut:
    remains = DataIn.remains
    products = ProductsModelToArray(DataIn.products)
    machines = MachinesModelToArray(DataIn.machines)
    max_daily_prod_zero = DataIn.max_daily_prod_zero
    count_days = DataIn.count_days

    result_calc = schedule_loom_calc(remains=remains, products=products, machines=machines,
                                max_daily_prod_zero=max_daily_prod_zero, count_days=count_days)

    if result_calc["error_str"] == "":
        schedule = [LoomPlan(machine_idx=s["machine_idx"], day_idx=s["day_idx"], product_idx=s["product_idx"])
                    for s in result_calc["schedule"]]
        result = LoomPlansOut(status=result_calc["status"], status_str=result_calc["status_str"],
                              schedule=schedule,products=result_calc["products"],
                              zeros=result_calc["zeros"], objective_value=result_calc["objective_value"],
                              proportion_diff=result_calc["proportion_diff"])
    else:
        result = LoomPlansOut(error_str=result_calc["error_str"], schedule=[], products=[], zeros=[])

    return result


def create_model(remains: list, products: list, machines: list, cleans: list, max_daily_prod_zero: int, count_days: int):
    # products = [  # (name, qty, id, machine_type).
    #     ("ZERO", 0, "0", 0)
    #     ("87416", 21, "1", 1)
    #     , ("18305", 9, "2", 0)
    #     , ("18302", 4, "3", 0)
    # ]
    # machines = [ # (name, product_idx, id, type)
    #     ("t1", 1, "1", 1)
    #     , ("t2", 1, "2", 1)
    #     , ("t3", 2, "3", 0)
    #     , ("t4", 0, "4", 0)
    #     , ("t5", 1, "1", 1)
    # ]
    # cleans = [ # (machine_idx, day_idx)
    #     (1, 2)
    # ]

    # Создаем множество дней чисток для быстрого доступа O(1)
    cleans_set = set(cleans)

    num_days = count_days
    num_machines = len(machines)
    num_products = len(products)

    all_machines = range(num_machines)
    all_days = range(num_days)
    all_products = range(num_products)
    products_with_ratio = range(1, num_products)

    proportions_input = [prop for a, prop, id, m_type in products]
    initial_products = {idx: product_idx for idx, (_, product_idx, id, t) in enumerate(machines)}

    model = cp_model.CpModel()

    jobs = {}

    # Список всех рабочих дней (не чисток)
    work_days = []
    # Значение для отображения чистки в итоговом расписании
    CLEANING_DISPLAY_CODE = -2

    for m in range(num_machines):
        for d in range(num_days):
            if (m, d) not in cleans_set:
                work_days.append((m, d))
                # Домен переменной: от 0 до num_products - 1
                jobs[(m, d)] = model.new_int_var(0, num_products - 1, f"job_{m}_{d}")

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

    # Количество нулевого продукта по дням
    # И просто количество нулевого продукта
    prod_zero_total = []
    for d in all_days:
        daily_prod_zero_on_machines = []
        for m in range(num_machines):
            if (m, d) not in cleans_set:
                # Используем уже созданные product_produced_bools для эффективности
                # product_produced_bools[PRODUCT_ZERO, m, d] истинно, если на машине m в день d производится PRODUCT_ZERO
                daily_prod_zero_on_machines.append(product_produced_bools[PRODUCT_ZERO, m, d])
                prod_zero_total.append(product_produced_bools[PRODUCT_ZERO, m, d])

        # Сумма этих булевых переменных даст количество PRODUCT_ZERO в день d
        model.Add(sum(daily_prod_zero_on_machines) <= max_daily_prod_zero)

    # Ограничения ПЕРЕХОДА

    # 2. Ограничения на переходы между продуктами
    # Переменные для отслеживания завершения двухдневного перехода
    completed_transition = {}
    is_not_zero = {}
    same_as_prev = {}
    prev_is_not_zero = {}
    prev2_is_not_zero = {}
    two_day_zero = {}
    for m in range(num_machines):
        for d in range(num_days):
            completed_transition[m, d] = model.NewBoolVar(f"completed_transition_{m}_{d}")

    # Ограничение для первого дня (d=0)
    for m in range(num_machines):
        initial_product = initial_products[m]
        is_initial_product = model.NewBoolVar(f"is_initial_product_{m}_0")
        is_not_zero[m, 0] = model.NewBoolVar(f"is_not_zero_{m}_0")

        model.Add(jobs[m, 0] == initial_product).OnlyEnforceIf(is_initial_product)
        model.Add(jobs[m, 0] != initial_product).OnlyEnforceIf(is_initial_product.Not())
        model.Add(jobs[m, 0] == PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 0].Not())
        model.Add(jobs[m, 0] != PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 0])

        # Первый день: либо начальный продукт, либо PRODUCT_ZERO
        model.AddBoolOr([is_initial_product, is_not_zero[m, 0].Not()])

        # Устанавливаем completed_transition для дня 0
        model.Add(completed_transition[m, 0] == 0)  # Нет перехода в день 0

    # Ограничение для второго дня (d=1)
    for m in range(num_machines):
        is_not_zero[m, 1] = model.NewBoolVar(f"is_not_zero_{m}_1")
        model.Add(jobs[m, 1] != PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 1])
        model.Add(jobs[m, 1] == PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, 1].Not())

        same_as_prev[m, 1] = model.NewBoolVar(f"same_as_prev_{m}_1")
        model.Add(jobs[m, 1] == jobs[m, 0]).OnlyEnforceIf(same_as_prev[m, 1])
        model.Add(jobs[m, 1] != jobs[m, 0]).OnlyEnforceIf(same_as_prev[m, 1].Not())

        prev_is_zero = model.NewBoolVar(f"prev_is_zero_{m}_1")
        model.Add(jobs[m, 0] == PRODUCT_ZERO).OnlyEnforceIf(prev_is_zero)
        model.Add(jobs[m, 0] != PRODUCT_ZERO).OnlyEnforceIf(prev_is_zero.Not())

        # Если день 0 - PRODUCT_ZERO, день 1 должен быть PRODUCT_ZERO для начала перехода
        model.Add(jobs[m, 1] == PRODUCT_ZERO).OnlyEnforceIf(prev_is_zero)

        # Если день 1 - не PRODUCT_ZERO, должен быть таким же, как день 0 (если день 0 не PRODUCT_ZERO)
        model.AddBoolOr([is_not_zero[m, 1].Not(), same_as_prev[m, 1]]).OnlyEnforceIf(prev_is_zero.Not())

        # completed_transition[m, 1] истинно, если день 0 и день 1 - PRODUCT_ZERO
        model.Add(completed_transition[m, 1] == prev_is_zero)

    # Логика переходов для дней d ≥ 2
    for m in range(num_machines):
        for d in range(2, num_days):
            if (m, d) in cleans_set:
                continue
            elif (m, d - 1) in cleans_set:
                pred_idx = d - 2
                pred_pred_idx = d - 3
            elif (m, d - 2) in cleans_set:
                pred_idx = d - 1
                pred_pred_idx = d - 3
            else:
                pred_idx = d - 1
                pred_pred_idx = d - 2

            is_not_zero[m, d] = model.NewBoolVar(f"is_not_zero_{m}_{d}")
            model.Add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, d])
            model.Add(jobs[m, d] == PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, d].Not())

            prev_is_not_zero[m, d] = model.NewBoolVar(f"prev_is_not_zero_{m}_{d}")
            model.Add(jobs[m, d - pred_idx] != PRODUCT_ZERO).OnlyEnforceIf(prev_is_not_zero[m, d])
            model.Add(jobs[m, d - pred_idx] == PRODUCT_ZERO).OnlyEnforceIf(prev_is_not_zero[m, d].Not())

            prev2_is_not_zero[m, d] = model.NewBoolVar(f"prev2_is_not_zero_{m}_{d}")
            model.Add(jobs[m, d - pred_pred_idx] != PRODUCT_ZERO).OnlyEnforceIf(prev2_is_not_zero[m, d])
            model.Add(jobs[m, d - pred_pred_idx] == PRODUCT_ZERO).OnlyEnforceIf(prev2_is_not_zero[m, d].Not())

            # Проверяем, был ли завершен двухдневный переход
            two_day_zero[m, d] = model.NewBoolVar(f"two_day_zero_{m}_{d}")
            model.AddBoolAnd(prev_is_not_zero[m, d].Not(), prev2_is_not_zero[m, d].Not()).OnlyEnforceIf(
                two_day_zero[m, d])
            model.AddBoolOr(prev_is_not_zero[m, d], prev2_is_not_zero[m, d]).OnlyEnforceIf(
                two_day_zero[m, d].Not())

            # Устанавливаем completed_transition
            model.Add(completed_transition[m, d] == two_day_zero[m, d])

            # Ограничения:
            # Если текущий день - не ноль, то либо:
            # 1) тот же продукт, что и вчера (если вчера не ноль)
            # 2) завершен двухдневный переход
            same_as_prev[m, d] = model.NewBoolVar(f"same_as_prev_{m}_{d}")
            model.Add(jobs[m, d] == jobs[m, d - pred_idx]).OnlyEnforceIf(same_as_prev[m, d])
            model.Add(jobs[m, d] != jobs[m, d - pred_idx]).OnlyEnforceIf(same_as_prev[m, d].Not())

            model.AddBoolOr([
                is_not_zero[m, d].Not(),  # Текущий день - PRODUCT_ZERO
                same_as_prev[m, d],  # Тот же продукт, что вчера
                completed_transition[m, d]  # Завершен двухдневный переход
            ])
            # Запрет на 3-й ZERO
            model.add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(completed_transition[m, d])
            # Запрет на переход в последние 2 дня
            if d >= count_days - pred_pred_idx:
                model.add(jobs[m, d] != PRODUCT_ZERO)

    # # не более 1-го простоя за неделю
    # for m in range(num_machines):
    #     prod_zero_on_machine = []
    #     for d in all_days:
    #         prod_zero_on_machine.append(product_produced_bools[PRODUCT_ZERO, m, d])
    #     model.Add(sum(prod_zero_on_machine) <= 2)

    # 6. Ограничение на "прыжки" - задания одного продукта должны быть сгруппированы
    # for m in all_machines:
    #     # Переменные для отслеживания групп
    #     group_changes = []
    #     for d in range(num_days - 1):
    #         group_changes.append(model.NewBoolVar(f'group_change_m{m}_d{d}'))
    #
    #     # Группы меняются, когда продукт изменяется
    #     for d in range(num_days - 1):
    #         if (m, d) in cleans_set:
    #             model.Add(group_changes[d] == 0)
    #         elif (m, d + 1) in cleans_set:
    #             model.Add(jobs[m, d] != jobs[m, d + 2]).OnlyEnforceIf(group_changes[d])
    #             model.Add(jobs[m, d] == jobs[m, d + 2]).OnlyEnforceIf(group_changes[d].Not())
    #         else:
    #             model.Add(jobs[m, d] != jobs[m, d + 1]).OnlyEnforceIf(group_changes[d])
    #             model.Add(jobs[m, d] == jobs[m, d + 1]).OnlyEnforceIf(group_changes[d].Not())
    #
    #
    #     # Ограничение: не более 2 изменений групп (3 группы) на машину
    #     model.Add(sum(group_changes) <= 2)

    # Ограничение на "прыжки" между продуктами
    # for m in range(num_machines):
    #     for d in range(1, num_days - 1):
    #         # Нельзя иметь последовательность product A -> product B -> product A
    #         p_prev = jobs[(m, d - 1)]
    #         p_curr = jobs[(m, d)]
    #         p_next = jobs[(m, d + 1)]
    #         model.add((p_prev == p_next) == 0).only_enforce_if([p_curr != p_prev, p_curr != p_next])
    #

    # Ограничение: Совместимость типов машин
    # for p in range(1, num_products):
    #     # Если у продукта есть требование к типу машины (тип 1)
    #     if products[p][3] == 1:
    #         for m in range(num_machines):
    #             # И тип машины не соответствует (тип 0)
    #             if machines[m][3] == 0:
    #                 for d in range(num_days):
    #                     # Запрещаем производство этого продукта на этой машине во все дни
    #                     if (m, d) in work_days:
    #                         model.add(jobs[(m, d)] != p)

    # ------------ Мягкое ограничение: Пропорции продукции (для продуктов с индексом > 0) ------------
    # Цель: минимизировать отклонение от заданных пропорций
    # Пропорции касаются только продуктов p > 0.

    # ------------------ Целевая функция (мягкое ограничение) ------------------
    # Цель: минимизировать отклонение от заданных пропорций.

    # Считаем общее количество произведенной продукции (исключая ZERO и дни чистки)
    total_production_days = model.NewIntVar(0, num_days * num_machines, "total_production_days")

    production_days_per_product = []
    for p in range(num_products):
        # Bool-переменные для дней, когда продукт p производится и это не день чистки
        is_produced_on_valid_day = []
        for m in range(num_machines):
            for d in range(num_days):
                if (m, d) not in cleans_set:
                    is_produced_on_valid_day.append(product_produced_bools[(p, m, d)])

        # Количество дней производства для каждого продукта
        count = model.NewIntVar(0, num_days * num_machines, f'count_{p}')
        model.Add(count == sum(is_produced_on_valid_day))
        production_days_per_product.append(count)

    # Суммируем только "полезные" дни производства (продукты > 0)
    model.Add(total_production_days == sum(production_days_per_product[p]
                                           for p in range(1, num_products)))

    # Общая сумма целевых пропорций
    total_proportions = sum(proportions_input[p] for p in range(1, num_products))

    # Минимизируем сумму абсолютных отклонений от пропорций.
    # Для этого используем трюк с перекрестным умножением, чтобы избежать деления.
    # |actual_count * total_prop - total_count * target_prop_p| -> min
    errors = []
    for p in range(1, num_products):
        # actual_count_p * total_proportions
        term1 = model.NewIntVar(-100000, 100000, f'term1_{p}')
        model.AddMultiplicationEquality(term1, [production_days_per_product[p], total_proportions])

        # total_production_days * proportions_input[p]
        term2 = model.NewIntVar(-100000, 100000, f'term2_{p}')
        model.AddMultiplicationEquality(term2, [total_production_days, proportions_input[p]])

        # diff = term1 - term2
        diff = model.NewIntVar(-100000, 100000, f'diff_{p}')
        model.Add(diff == term1 - term2)

        # abs_error = abs(diff)
        abs_error = model.NewIntVar(0, 100000, f'abs_error_{p}')
        model.AddAbsEquality(abs_error, diff)
        errors.append(abs_error)

    downtime_penalty = 10

    model.Minimize(sum(errors) + sum(prod_zero_total) * downtime_penalty)

    return model, jobs, product_counts, errors


def schedule_loom_calc(remains: list, products: list, machines: list, cleans: list, max_daily_prod_zero: int,
                       count_days: int) -> LoomPlansOut:
    try:
        # products = [  # (name, qty, id, machine_type).
        #     ("ZERO", 0, "0", 0)
        #     ("87416", 21, "1", 1)
        #     , ("18305", 9, "2", 0)
        #     , ("18302", 4, "3", 0)
        # ]
        # machines = [ # (name, product_idx, id, type)
        #     ("t1", 1, "1", 1)
        #     , ("t2", 1, "2", 1)
        #     , ("t3", 2, "3", 0)
        #     , ("t4", 0, "4", 0)
        #     , ("t5", 1, "1", 1)
        # ]
        # cleans = [ # (day_idx, machine_idx
        #   (6, 1)
        # ]
        #

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 600
        model, jobs, product_counts, proportion_objective_terms = create_model(
            remains=remains, products=products, machines=machines, cleans=cleans, max_daily_prod_zero=max_daily_prod_zero,
            count_days=count_days)
        status = solver.solve(model)
        machines_full = []
        machines_change = {}
        products_change = {}
        if status == cp_model.FEASIBLE:
            logger.info(f"Первичная проверка")
            product_del = []
            products_old = products.copy()
            machines_old = machines.copy()
            for p in range(len(products)):
                qty = solver.Value(product_counts[p])
                if p > 0 and qty >= count_days:
                    for m in range(len(machines)):
                        if machines[m][1] == p:
                            logger.info(f"Можно убрать индекс {p} на машину {m}")
                            # убираем объем из данных
                            machines_full.append((machines[m][2], products[p][2]))
                            for m_change in range(m + 1, len(machines)):
                                if machines_change.get(m_change - 1):
                                    pass
                                else:
                                    machines_change[m_change - 1] = m_change
                            if products[p][1] - count_days > 0:
                                logger.info(f"  уменьшаем индекс {p}  и удаляем машину {m}")
                                products[p] = (products[p][0], products[p][1] - count_days, products[p][2])
                                machines.pop(m)
                            else:
                                p_exist = False
                                for m1 in range(len(machines)):
                                    if m1 != m and machines[m1][1] == p:
                                        p_exist = True
                                machines.pop(m)
                                if p_exist:
                                    logger.info(f"  обнуляем индекс {p}  и удаляем машину {m}")
                                    products[p] = (products[p][0], 0, products[p][2])
                                else:
                                    product_del.append(p)
                                    logger.info(f"  удаляем машину {m}")
                            break
            product_del.sort(reverse=True)
            for p_old in product_del:
                for p in range(p_old + 1, len(products)):
                    for m in range(len(machines)):
                        if machines[m][1] == p:
                            machines[m] = (machines[m][0], p - 1)
                products.pop(p_old)
                logger.info(f"  удаляем индекс {p_old}")

        num_days = count_days
        num_machines = len(machines)
        num_products = len(products)
        model, jobs, product_counts, proportion_objective_terms = create_model(
            remains=remains, products=products, machines=machines, cleans=cleans,
            max_daily_prod_zero=max_daily_prod_zero, count_days=count_days)
        solver.parameters.max_time_in_seconds = settings.LOOM_MAX_TIME
        status = solver.solve(model)
        diff_all = 0
        schedule = []
        zeros = []
        products_schedule = []

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            logger.info(f"Статус решения: {solver.StatusName(status)}")
            if proportion_objective_terms:
                logger.info(f"Минимальное значение функции цели (сумма абс. отклонений пропорций): {solver.ObjectiveValue()}")

            for m in range(num_machines):
                logger.info(f"Loom {m}")
                for d in range(num_days):
                    p = solver.value(jobs[m, d])
                    schedule.append({"machine_idx": m, "day_idx": d, "product_idx": p})
                    logger.info(f"  Day {d} works  {p}")

            logger.info("\nОбщее количество произведенной продукции:")

            for p in range(num_products):
                diff = 0 if p ==0 else solver.value(proportion_objective_terms[p-1])
                diff_all += diff
                qty = solver.Value(product_counts[p])
                products_schedule.append(ProductPlan(product_idx=p, qty=qty, penalty=diff))
                logger.info(f"  Продукт {p}: {qty} единиц, штраф пропорций {diff}")

        elif status == cp_model.INFEASIBLE:
            logger.info(solver.ResponseStats())  # Основные статистические данные

        # Statistics.
        logger.info("Statistics")
        logger.info(f"  - conflicts: {solver.num_conflicts}")
        logger.info(f"  - branches : {solver.num_branches}")
        logger.info(f"  - wall time: {solver.wall_time}s")
        result = {"status": int(status), "status_str": solver.StatusName(status), "schedule": schedule,
                  "products": products_schedule, "zeros": zeros, "objective_value": int(solver.ObjectiveValue()),
                  "proportion_diff": int(diff_all), "error_str": ""}

    except Exception as e:
        error = tr.TracebackException(exc_type=type(e), exc_traceback=e.__traceback__, exc_value=e).stack[-1]
        error_str = '{} in {} row:{} '.format(e, error.lineno, error.line)
        logger.error(error_str)
        result = {"error_str": error_str}
    return result


