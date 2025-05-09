import collections
from ortools.sat.python import cp_model

def main() -> None:

    # Data.
    # массив остатков
    remains = [
        [1, 2, 2] # reed
        , [47] # shaft
    ]

    solver = cp_model.CpSolver()

    products = [  # (name, qty, [(reed_idx, qty), (shaft_idx, qty)]).
        ("87416", 21, [(0, 1), (0, 7)] )
        , ("18305", 9, [(1, 1), (0, 6)])
        , ("18302", 4, [(2, 1), (0, 7)])
    ]
    num_machines = 5
    num_products = len(products) + 1
    num_days = 7
    all_machines = range(num_machines)
    all_products = range(num_products+1)
    relevant_products = range(1, num_products + 1)
    all_days = range(num_days)
    proportions_input = [0] + [prop for a, prop, l in products]

    # Create the model.
    model = cp_model.CpModel()

    jobs = {}
    task_type = collections.namedtuple("task_type", "start end interval")

    machine_to_intervals = collections.defaultdict(list)
    for m in all_machines:
        for d in all_days:
            jobs[m, d] = model.new_int_var(0, num_products - 1, f"job_{m}_{d}")

    PRODUCT_ZERO = 0  # Индекс "особенной" продукции

    # ------------ Подсчет общего количества каждого продукта ------------
    # Вспомогательные булевы переменные: product_produced[p, m, d] истинно, если продукт p производится на машине m в день d
    product_produced_bools = {}
    for p in range(num_products):
        for m in range(num_machines):
            for d in range(num_days):
                product_produced_bools[p, m, d] = model.NewBoolVar(f"product_produced_{p}_{m}_{d}")
                # Связь product_produced_bools с jobs
                model.Add(jobs[m, d] == p).OnlyEnforceIf(product_produced_bools[p, m, d])
                model.Add(jobs[m, d] != p).OnlyEnforceIf(product_produced_bools[p, m, d].Not())

    # Общее количество каждого продукта: product_counts[p]
    product_counts = [model.NewIntVar(0, num_machines * num_days, f"count_prod_{p}") for p in range(num_products)]
    for p in range(num_products):
        model.Add(product_counts[p] == sum(
            product_produced_bools[p, m, d] for m in range(num_machines) for d in range(num_days)))

    # ------------ Мягкое ограничение: Пропорции продукции (для продуктов с индексом > 0) ------------
    # Цель: минимизировать отклонение от заданных пропорций
    # Пропорции касаются только продуктов p > 0.
    # Мы хотим, чтобы product_counts[p1] / product_counts[p2] было близко к proportions_input[p1] / proportions_input[p2]
    # Это эквивалентно product_counts[p1] * proportions_input[p2] ~= product_counts[p2] * proportions_input[p1]

    proportion_objective_terms = []
    relevant_product_indices = []
    if num_products > 1:  # Пропорции имеют смысл только если есть хотя бы 2 продукта (один из которых может быть PRODUCT_ZERO)
        for p_idx in range(num_products):
            if p_idx != PRODUCT_ZERO and proportions_input[p_idx] > 0:
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
                model.Add(diff_var == term1_expr - term2_expr)

                abs_diff_var = model.NewIntVar(0, max_abs_diff_val, f"prop_abs_diff_{p1_idx}_{p2_idx}")
                model.AddAbsEquality(abs_diff_var, diff_var)
                proportion_objective_terms.append(abs_diff_var)

    if proportion_objective_terms:
        model.Minimize(sum(proportion_objective_terms))
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
        for d in range(1, num_days):
            prev_prod = jobs[m, d - 1]
            curr_prod = jobs[m, d]

            # Вспомогательные булевы переменные для текущего перехода
            b_prev_is_P0 = model.NewBoolVar(f"b_prev_is_P0_{m}_{d}")
            model.Add(prev_prod == PRODUCT_ZERO).OnlyEnforceIf(b_prev_is_P0)
            model.Add(prev_prod != PRODUCT_ZERO).OnlyEnforceIf(b_prev_is_P0.Not())

            b_curr_is_P0 = model.NewBoolVar(f"b_curr_is_P0_{m}_{d}")
            model.Add(curr_prod == PRODUCT_ZERO).OnlyEnforceIf(b_curr_is_P0)
            model.Add(curr_prod != PRODUCT_ZERO).OnlyEnforceIf(b_curr_is_P0.Not())

            b_prev_eq_curr = model.NewBoolVar(f"b_prev_eq_curr_{m}_{d}")
            model.Add(prev_prod == curr_prod).OnlyEnforceIf(b_prev_eq_curr)
            model.Add(prev_prod != curr_prod).OnlyEnforceIf(b_prev_eq_curr.Not())

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

    # количество чисток
    count = 0
    product_count_clear = model.NewIntVar(0, num_machines * num_days, f"products_ count")
    for m in all_machines:
        for d in all_days:
            # Создаем булеву переменную для проверки, равен ли jobs[m, d] текущему продукту
            is_clear = model.NewBoolVar(f"is_clear_{m}_{d}")
            model.Add(jobs[m, d] == 0).OnlyEnforceIf(is_clear)
            model.Add(jobs[m, d] != 0).OnlyEnforceIf(is_clear.Not())

                # Добавляем к общему счетчику
            count += is_clear

        # Приравниваем счетчик к переменной
        model.Add(product_count_clear == count)
    #
    # model.minimize(product_count_clear)

    status = solver.solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        products_prop_fact = [0 for l in all_products]
        for m in range(num_machines):
            print(f"Loom {m}")
            for d in range(num_days):
                p = solver.value(jobs[m, d])
                products_prop_fact[p] += 1
                print(f"  Day {d} works  {p}")

        print("\nОбщее количество произведенной продукции:")
        for p in range(num_products):
            print(f"  Продукт {p}: {solver.Value(product_counts[p])} единиц")

        print("\nИндикаторы перехода с Продукта 0 на другой продукт (1=переход был):")
        for m in range(num_machines):
            for d in range(1, num_days):
                if (m, d) in is_transition_P0_to_nonP0:  # Проверяем, существует ли ключ
                    change = solver.Value(is_transition_P0_to_nonP0[m, d])
                    if change > 0:
                        print(f"  Машина {m}, День {d - 1}->{d}: {change}")
                else:  # Этого не должно произойти, если все ключи были добавлены
                    print(f"  Машина {m}, День {d - 1}->{d}: индикатор не найден (ошибка)")
        # print(f"Proportion: {solver.value(product_count_clear)}")
        # for i in range(len(products_prop_fact)):
        #     penalty = 0 if i == 0 else solver.value(penalty_vars[i-1])
        #     #deviation = 0 if i == 0 else solver.value(variables[f"dev_{i}"])
        #     print(f"  {i} - {products_prop_fact[i]}  penalty {penalty} count {solver.value(product_counts[i])} ")

    elif status == cp_model.INFEASIBLE:
        print(solver.ResponseStats())  # Основные статистические данные

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts: {solver.num_conflicts}")
    print(f"  - branches : {solver.num_branches}")
    print(f"  - wall time: {solver.wall_time}s")


if __name__ == "__main__":
    main()