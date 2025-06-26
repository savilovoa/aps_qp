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
    # products = [  # (name, qty, [(reed_idx, qty), (shaft_idx, qty)]).
    #     ("87416", 21, [(0, 1), (0, 7)] )
    #     , ("18305", 9, [(1, 1), (0, 6)])
    #     , ("18302", 4, [(2, 1), (0, 7)])
    # ]
    # machines = [ # (name, product_idx)
    #     ("t1", 1)
    #     , ("t2", 1)
    #     , ("t3", 2)
    #     , ("t4", 0)
    #     , ("t5", 1)
    # ]
    # max_daily_prod_zero = 3

    num_days = count_days
    num_machines = len(machines)
    num_products = len(products)

    all_machines = range(num_machines)
    all_days = range(num_days)
    all_products = range(num_products)
    products_with_ratio = range(1, num_products)

    proportions_input = [prop for a, prop, id, t in products]
    initial_products = {idx: product_idx for idx, (_, product_idx, m_id, t) in enumerate(machines)}

    model = cp_model.CpModel()

    jobs = {}
    for m in all_machines:
        for d in all_days:
            jobs[m, d] = model.new_int_var(0, num_products - 1, f"job_{m}_{d}")

    PRODUCT_ZERO = 0  # Индекс "особенной" продукции

    # ------------ Подсчет общего количества каждого продукта ------------
    # Вспомогательные булевы переменные: product_produced[p, m, d] истинно, если продукт p производится на машине m в день d
    product_produced_bools = {}
    for p in all_products:
        for m in all_machines:
            for d in all_days:
                product_produced_bools[p, m, d] = model.NewBoolVar(f"product_produced_{p}_{m}_{d}")
                # Связь product_produced_bools с jobs
                model.Add(jobs[m, d] == p).OnlyEnforceIf(product_produced_bools[p, m, d])
                model.Add(jobs[m, d] != p).OnlyEnforceIf(product_produced_bools[p, m, d].Not())

    # Общее количество каждого продукта: product_counts[p]
    product_counts = [model.NewIntVar(0, num_machines * num_days, f"count_prod_{p}") for p in range(num_products)]
    for p in all_products:
        model.Add(product_counts[p] == sum(
            product_produced_bools[p, m, d] for m in all_machines for d in all_days))

    # Количество нулевого продукта по дням
    # И просто количество нулевого продукта
    prod_zero_total = []
    for d in all_days:
        daily_prod_zero_on_machines = []
        for m in range(num_machines):
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
            is_not_zero[m, d] = model.NewBoolVar(f"is_not_zero_{m}_{d}")
            model.Add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, d])
            model.Add(jobs[m, d] == PRODUCT_ZERO).OnlyEnforceIf(is_not_zero[m, d].Not())

            prev_is_not_zero[m, d] = model.NewBoolVar(f"prev_is_not_zero_{m}_{d}")
            model.Add(jobs[m, d - 1] != PRODUCT_ZERO).OnlyEnforceIf(prev_is_not_zero[m, d])
            model.Add(jobs[m, d - 1] == PRODUCT_ZERO).OnlyEnforceIf(prev_is_not_zero[m, d].Not())

            prev2_is_not_zero[m, d] = model.NewBoolVar(f"prev2_is_not_zero_{m}_{d}")
            model.Add(jobs[m, d - 2] != PRODUCT_ZERO).OnlyEnforceIf(prev2_is_not_zero[m, d])
            model.Add(jobs[m, d - 2] == PRODUCT_ZERO).OnlyEnforceIf(prev2_is_not_zero[m, d].Not())

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
            model.Add(jobs[m, d] == jobs[m, d - 1]).OnlyEnforceIf(same_as_prev[m, d])
            model.Add(jobs[m, d] != jobs[m, d - 1]).OnlyEnforceIf(same_as_prev[m, d].Not())

            model.AddBoolOr([
                is_not_zero[m, d].Not(),  # Текущий день - PRODUCT_ZERO
                same_as_prev[m, d],  # Тот же продукт, что вчера
                completed_transition[m, d]  # Завершен двухдневный переход
            ])
            # Запрет на 3-й ZERO
            model.add(jobs[m, d] != PRODUCT_ZERO).OnlyEnforceIf(completed_transition[m, d])
            # Запрет на переход в последние 2 дня
            if d >= count_days - 2:
                model.add(jobs[m, d] != PRODUCT_ZERO)

    # # не более 1-го простоя за неделю
    # for m in range(num_machines):
    #     prod_zero_on_machine = []
    #     for d in all_days:
    #         prod_zero_on_machine.append(product_produced_bools[PRODUCT_ZERO, m, d])
    #     model.Add(sum(prod_zero_on_machine) <= 2)

    # 6. Ограничение на "прыжки" - задания одного продукта должны быть сгруппированы
    for m in all_machines:
        # Переменные для отслеживания групп
        group_changes = []
        for d in range(num_days - 1):
            group_changes.append(model.NewBoolVar(f'group_change_m{m}_d{d}'))

        # Группы меняются, когда продукт изменяется
        for d in range(num_days - 1):
            model.Add(jobs[m, d] != jobs[m, d + 1]).OnlyEnforceIf(group_changes[d])
            model.Add(jobs[m, d] == jobs[m, d + 1]).OnlyEnforceIf(group_changes[d].Not())

        # Ограничение: не более 2 изменений групп (3 группы) на машину
        model.Add(sum(group_changes) <= 2)

    # Ограничение на "прыжки" между продуктами
    # for m in range(num_machines):
    #     for d in range(1, num_days - 1):
    #         # Нельзя иметь последовательность product A -> product B -> product A
    #         p_prev = jobs[(m, d - 1)]
    #         p_curr = jobs[(m, d)]
    #         p_next = jobs[(m, d + 1)]
    #         model.add((p_prev == p_next) == 0).only_enforce_if([p_curr != p_prev, p_curr != p_next])
    #

    # ------------ Мягкое ограничение: Пропорции продукции (для продуктов с индексом > 0) ------------
    # Цель: минимизировать отклонение от заданных пропорций
    # Пропорции касаются только продуктов p > 0.
    # Мы хотим, чтобы product_counts[p1] / product_counts[p2] было близко к proportions_input[p1] / proportions_input[p2]
    # Это эквивалентно product_counts[p1] * proportions_input[p2] ~= product_counts[p2] * proportions_input[p1]

    proportion_objective_terms = []
    relevant_product_indices = []
    if num_products > 1:  # Пропорции имеют смысл только если есть хотя бы 2 продукта (один из которых может быть PRODUCT_ZERO)
        for p_idx in range(num_products):
            if p_idx != PRODUCT_ZERO:  # and proportions_input[p_idx] > 0:
                relevant_product_indices.append(p_idx)

    if len(relevant_product_indices) > 1:
        for i in range(len(relevant_product_indices)):
            for j in range(i + 1, len(relevant_product_indices)):
                p1_idx = relevant_product_indices[i]
                p2_idx = relevant_product_indices[j]

                # product_counts[p1_idx] * proportions_input[p2_idx]
                term1_expr = model.NewIntVar(0, num_machines * num_days * max(proportions_input),
                                             f"term1_{p1_idx}_{p2_idx}")
                model.AddMultiplicationEquality(term1_expr, [product_counts[p1_idx],
                                                             model.NewConstant(proportions_input[p2_idx])])

                # product_counts[p2_idx] * proportions_input[p1_idx]
                term2_expr = model.NewIntVar(0, num_machines * num_days * max(proportions_input),
                                             f"term2_{p1_idx}_{p2_idx}")
                model.AddMultiplicationEquality(term2_expr, [product_counts[p2_idx],
                                                             model.NewConstant(proportions_input[p1_idx])])

                # diff = term1_expr - term2_expr
                max_abs_diff_val = num_machines * num_days * max(proportions_input)  # Оценка максимальной разницы
                diff_var = model.NewIntVar(-max_abs_diff_val, max_abs_diff_val, f"prop_diff_{p1_idx}_{p2_idx}")
                model.Add(diff_var == (term1_expr - term2_expr) * (num_products - i))

                abs_diff_var = model.NewIntVar(0, max_abs_diff_val, f"prop_abs_diff_{p1_idx}_{p2_idx}")
                model.AddAbsEquality(abs_diff_var, diff_var)
                proportion_objective_terms.append(abs_diff_var)

    # if proportion_objective_terms:

    downtime_penalty = round(0.1 * sum(proportions_input)/num_machines * num_days)
    if downtime_penalty < 1:
        downtime_penalty = 1

    model.Minimize(sum(proportion_objective_terms) + sum(prod_zero_total) * downtime_penalty)

    return model, jobs, product_counts, proportion_objective_terms


def schedule_loom_calc(remains: list, products: list, machines: list, cleans: list, max_daily_prod_zero: int, count_days: int) -> LoomPlansOut:
    try:

        #
        #
        num_products = len(products)

        # solver.parameters.log_search_progress = True
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 600
        model, jobs, product_counts, proportion_objective_terms = create_model(
            remains=remains, products=products, machines=machines, cleans=cleans, max_daily_prod_zero=max_daily_prod_zero,
            count_days=count_days)
        status = solver.solve(model)
        machines_full = []
        if status == cp_model.FEASIBLE:
            logger.info(f"Первичная проверка")
            product_del = []
            for p in range(num_products):
                qty = solver.Value(product_counts[p])
                if p > 0 and qty >= count_days:
                    for m in range(len(machines)):
                        if machines[m][1] == p:
                            logger.info(f"Можно убрать индекс {p} на машину {m}")
                            # убираем объем из данных
                            machines_full.append((m, p))
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
            remains=remains, products=products, machines=machines, cleans=cleans, max_daily_prod_zero=max_daily_prod_zero,
            count_days=count_days)
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


