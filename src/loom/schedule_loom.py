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

    if result_calc.error_str == "":
        schedule = [LoomPlan(machine_idx=s["machine_idx"], day_idx=s["day_idx"], product_idx=s["product_idx"])
                    for s in result_calc["schedule"]]
        result = LoomPlansOut(status=result_calc["status"], status_str=result_calc["status_str"],
                              schedule=schedule,products=result_calc["products"],
                              zeros=result_calc["zeros"], objective_value=result_calc["objective_value"],
                              proportion_diff=result_calc["proportion_diff"])
    else:
        result = LoomPlansOut(error_str=result_calc.error_str, schedule=[], products=[], zeros=[])

def schedule_loom_calc(remains: list, products: list, machines: list, max_daily_prod_zero: int, count_days: int) -> LoomPlansOut:
    try:

        #
        #
        # products = [  # (name, qty, [(reed_idx, qty), (shaft_idx, qty)]).
        #     ("87416", 21, [(0, 1), (0, 7)] )
        #     , ("18305", 9, [(1, 1), (0, 6)])
        #     , ("18302", 4, [(2, 1), (0, 7)])
        # ]
        # machines = [ # (name, product_idx)
        #     ("t1", 1)
        #     , ("t2", 0)
        #     , ("t3", 0)
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

        proportions_input = [prop for a, prop, l in products]

        solver = cp_model.CpSolver()
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

        # Продукт нельзя планировать, если пропорция = -1 ------------
        for p in range(num_products):
            if proportions_input[p] == -1:
                model.Add(
                    product_counts[p] == 0)  # Этот продукт не должен производиться в течение планового периода

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
                if p_idx != PRODUCT_ZERO: # and proportions_input[p_idx] > 0:
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

        #if proportion_objective_terms:

        downtime_penalty = 10

        model.Minimize(sum(proportion_objective_terms) + sum(prod_zero_total)*downtime_penalty)
        # Если proportion_objective_terms пуст (например, мало продуктов или нулевые пропорции),
        # то любая допустимая конфигурация будет оптимальной с точки зрения пропорций.

        # ------------ Жесткие ограничения: Логика смены продукции ------------
        # Эти ограничения действуют для d > 0 (т.е. со второго дня)

        # Переменная для индикации перехода с PRODUCT_ZERO на другой продукт (не PRODUCT_ZERO)
        # "Ограничение которое разрешает любой индекс, после нулевого сделать через отдельную переменную"
        # Мы создадим эту переменную, чтобы она отражала условие, но само разрешение уже заложено в логике ниже.
        # is_transition_P0_to_nonP0[m, d] будет истинным, если jobs[m,d-1]==PRODUCT_ZERO и jobs[m,d]!=PRODUCT_ZERO
        is_transition_P0_to_nonP0 = {}

        for m in range(num_machines):
            for d in range(num_days):
                curr_prod = jobs[m, d]

                # Вспомогательные булевы переменные для предыдущего продукта
                b_prev_is_P0 = model.NewBoolVar(f"b_prev_is_P0_{m}_{d}")
                b_prev_eq_curr = model.NewBoolVar(f"b_prev_eq_curr_{m}_{d}")

                if d == 0:
                    prev_prod_value_const = machines[m][1]   # Это константа Python

                    # Фиксируем b_prev_is_P0 на основе константы
                    if prev_prod_value_const == PRODUCT_ZERO:
                        model.Add(b_prev_is_P0 == True)
                    else:
                        model.Add(b_prev_is_P0 == False)

                    # Связываем b_prev_eq_curr с curr_prod и константой prev_prod_value_const
                    model.Add(curr_prod == prev_prod_value_const).OnlyEnforceIf(b_prev_eq_curr)
                    model.Add(curr_prod != prev_prod_value_const).OnlyEnforceIf(b_prev_eq_curr.Not())

                    # Новое Ограничение: если machines_first[m] - "запрещенный" продукт (пропорция -1)
                    # то первый день на этой машине должен быть PRODUCT_ZERO
                    if proportions_input[prev_prod_value_const] == -1:
                        model.Add(curr_prod == PRODUCT_ZERO)
                else:
                    prev_prod_var = jobs[m, d - 1]  # Это переменная модели

                    model.Add(prev_prod_var == PRODUCT_ZERO).OnlyEnforceIf(b_prev_is_P0)
                    model.Add(prev_prod_var != PRODUCT_ZERO).OnlyEnforceIf(b_prev_is_P0.Not())

                    model.Add(prev_prod_var == curr_prod).OnlyEnforceIf(b_prev_eq_curr)
                    model.Add(prev_prod_var != curr_prod).OnlyEnforceIf(b_prev_eq_curr.Not())

                b_curr_is_P0 = model.NewBoolVar(f"b_curr_is_P0_{m}_{d}")
                model.Add(curr_prod == PRODUCT_ZERO).OnlyEnforceIf(b_curr_is_P0)
                model.Add(curr_prod != PRODUCT_ZERO).OnlyEnforceIf(b_curr_is_P0.Not())

                # Правило 1: Если предыдущий продукт не PRODUCT_ZERO и текущий продукт не PRODUCT_ZERO,
                # то они должны быть одинаковыми.
                # (prev_prod != P0 AND curr_prod != P0) => prev_prod == curr_prod
                # Эквивалентно: OR(prev_prod == P0, curr_prod == P0, prev_prod == curr_prod)
                # Эквивалентно: OR(b_prev_is_P0, b_curr_is_P0, b_prev_eq_curr)
                model.AddBoolOr([b_prev_is_P0, b_curr_is_P0, b_prev_eq_curr])

                # Правило 2: После PRODUCT_ZERO можно ставить любой продукт, *кроме* PRODUCT_ZERO.
                # То есть, если prev_prod == P0, то curr_prod != P0.
                # prev_prod == P0 => curr_prod != P0
                # Эквивалентно: OR(prev_prod != P0, curr_prod != P0)
                # Эквивалентно: OR(b_prev_is_P0.Not(), b_curr_is_P0.Not())
                model.AddBoolOr([b_prev_is_P0.Not(), b_curr_is_P0.Not()])

                # Явная возможность продукции с нулевым индексом (PRODUCT_ZERO) после любой *ненулевой* продукции.
                # Это покрывается Правилом 1 (если не меняем не-ноль на не-ноль, то можно на ноль)
                # и разрешением перехода X -> 0, которое не запрещено выше.
                # Если prev_prod != P0 и curr_prod == P0:
                #   b_prev_is_P0 = False, b_curr_is_P0 = True, b_prev_eq_curr = False
                #   Правило 1: AddBoolOr([False, True, False]) -> Add(True) - разрешено.
                #   Правило 2: AddBoolOr([True, False]) -> Add(True) - разрешено (не затрагивает, т.к. prev != P0).

                # Создание индикаторной переменной is_transition_P0_to_nonP0[m,d]
                # is_P0_to_nonP0_m_d <=> (b_prev_is_P0 AND b_curr_is_P0.Not())
                var_key = (m, d)
                is_transition_P0_to_nonP0[var_key] = model.NewBoolVar(f"is_P0_to_nonP0_{m}_{d}")

                # 1. is_transition_P0_to_nonP0[var_key] => b_prev_is_P0
                model.AddImplication(is_transition_P0_to_nonP0[var_key], b_prev_is_P0)
                # 2. is_transition_P0_to_nonP0[var_key] => b_curr_is_P0.Not()
                model.AddImplication(is_transition_P0_to_nonP0[var_key], b_curr_is_P0.Not())
                # 3. (b_prev_is_P0 AND b_curr_is_P0.Not()) => is_transition_P0_to_nonP0[var_key]
                #    Эквивалентно OR(b_prev_is_P0.Not(), b_curr_is_P0, is_transition_P0_to_nonP0[var_key])
                model.AddBoolOr([b_prev_is_P0.Not(), b_curr_is_P0, is_transition_P0_to_nonP0[var_key]])

        # solver.parameters.log_search_progress = True
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

            logger.info("\nИндикаторы перехода с Продукта 0 на другой продукт (1=переход был):")

            for d in range(1, num_days):
                change_count = 0
                for m in range(num_machines):
                    if (m, d) in is_transition_P0_to_nonP0:  # Проверяем, существует ли ключ
                        change = solver.Value(is_transition_P0_to_nonP0[m, d])
                        if change > 0:
                            change_count += 1
                            logger.info(f"  Машина {m}, День {d - 1}->{d}: {change}")
                    else:  # Этого не должно произойти, если все ключи были добавлены
                        logger.info(f"  Машина {m}, День {d - 1}->{d}: индикатор не найден (ошибка)")
                if change_count > 0:
                    zeros.append(DayZero(day_idx=d, count_zero=change_count))

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

