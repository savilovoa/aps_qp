import collections
from ortools.sat.python import cp_model

def main() -> None:
    """Minimal jobshop problem."""
    # Data.
    # массив остатков
    remains = [
        [1, 2, 2] # reed
        , [47] # shaft
    ]

    solver = cp_model.CpSolver()

    products = [  # (name, qty, [(reed_idx, qty), (shaft_idx, qty)]).
        ("87416", 21, [(0, 1), (0, 7)] )
        , ("18305", 8, [(1, 1), (0, 6)])
        , ("18302", 5, [(2, 1), (0, 7)])
    ]
    num_machines = 5
    num_products = len(products)
    num_days = 7
    all_machines = range(num_machines)
    all_jobs = range(num_products+1)
    all_products = range(1, num_products + 1)
    all_days = range(num_days)

    # Create the model.
    model = cp_model.CpModel()

    all_tasks = {}
    task_type = collections.namedtuple("task_type", "start end interval")

    machine_to_intervals = collections.defaultdict(list)
    for m in all_machines:
        for d in all_days:
            all_tasks[m, d] = model.new_int_var(0, num_products, f"job_{m}_{d}")

    proportions = [0] + [prop for art, prop, l in products]
    total_proportion = sum(proportions)
    print(proportions, total_proportion)

    variables = {}

    # Создаем счетчики для каждого продукта
    product_counts = [model.NewIntVar(0, num_machines * num_days, f"count_{p}") for p in all_jobs]
    sum_products = model.new_int_var(1, num_machines * num_days, "sum_products")
    # Подсчитываем количество каждого продукта
    for p in all_jobs:
        count = 0
        for m in all_machines:
            for d in all_days:
                # Создаем булеву переменную для проверки, равен ли jobs[m, d] текущему продукту
                is_product = model.NewBoolVar(f"is_product_{m}_{d}_{p}")
                model.Add(all_tasks[m, d] == p).OnlyEnforceIf(is_product)
                model.Add(all_tasks[m, d] != p).OnlyEnforceIf(is_product.Not())

                # Добавляем к общему счетчику
                count += is_product

        # Приравниваем счетчик к переменной
        model.Add(product_counts[p] == count)
    model.Add(sum_products == num_machines * num_days)

    # Создаем штрафные переменные для отклонений
    penalties = []

    for p in all_products:  # начинаем с 1, так как 0 может быть любым
        if proportions[p] == 0:
            continue

        # Рассчитываем целевое количество
        target_count = ((num_machines * num_days) * proportions[p]) // total_proportion
        print(target_count)

        # Создаем переменную отклонения
        deviation = model.new_int_var(0, num_machines * num_days, f"dev_{p}")
        model.AddAbsEquality(deviation, product_counts[p] - target_count)
        variables[f"dev_{p}"] = deviation

        # Добавляем штраф
        penalty = model.new_int_var(0, 10000, f"penalty_{p}")
        model.Add(penalty == deviation)  # коэффициент штрафа можно настроить
        penalties.append(penalty)

    # Добавляем штрафы в целевую функцию
    model.Minimize(sum(penalties))

    # **Второе ограничение: смена продукции**

    # Создаем отдельную переменную для разрешения любого индекса после 0
    any_after_zero = {}
    for m in range(num_machines):
        for d in range(1, num_days):
            any_after_zero[m, d] = model.NewBoolVar(f"any_after_zero_{m}_{d}")

    for m in range(num_machines):
        for d in range(1, num_days):  # начинаем со второго дня
            # Если продукция изменилась
            is_product_pred = model.NewBoolVar(f"is_product_pred{m}_{d}")
            model.Add(all_tasks[m, d] == all_tasks[m, d-1] ).OnlyEnforceIf(is_product_pred)
            model.Add(all_tasks[m, d] != all_tasks[m, d-1]).OnlyEnforceIf(is_product_pred.Not())

            #model.Add(all_tasks[m, d - 1] == 0).OnlyEnforceIf(is_product_pred.Not())
            # model.Add(all_tasks[m, d - 1] == all_tasks[m, d])

            # После 0 можно ставить любой индекс, кроме 0
            model.Add(all_tasks[m, d] != 0).OnlyEnforceIf(any_after_zero[m, d])
            model.Add(all_tasks[m, d - 1] == 0).OnlyEnforceIf(any_after_zero[m, d])
            model.Add(all_tasks[m, d] == 0).OnlyEnforceIf(any_after_zero[m, d])
    #
    # # Добавляем ограничения пропорции
    # for p in all_products:
    #     # Количество продукта должно быть пропорционально его доле
    #     model.Add(product_counts[p] * total_proportion ==
    #               proportions[p-1] * sum(product_counts))

    count = 0
    product_count_clear = model.NewIntVar(0, num_machines * num_days, f"products_ count")
    for m in all_machines:
        for d in all_days:
            # Создаем булеву переменную для проверки, равен ли jobs[m, d] текущему продукту
            is_clear = model.NewBoolVar(f"is_clear_{m}_{d}")
            model.Add(all_tasks[m, d] == 0).OnlyEnforceIf(is_clear)
            model.Add(all_tasks[m, d] != 0).OnlyEnforceIf(is_clear.Not())

                # Добавляем к общему счетчику
            count += is_clear

        # Приравниваем счетчик к переменной
        model.Add(product_count_clear == count)
    #
    # model.minimize(product_count_clear)

    status = solver.solve(model)
    print(status)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        products_prop_fact = [0 for l in all_jobs]
        print(products_prop_fact)
        for d in range(num_days):
            print(f"Day {d}")
            for m in range(num_machines):
                p = solver.value(all_tasks[m, d])
                products_prop_fact[p] += 1
                print(f"  Loom {m} works  {p}")

        print(f"Proportion: {solver.value(product_count_clear)}")
        for i in range(len(products_prop_fact)):
            penalty = 0 if i == 0 else solver.value(penalties[i-1])
            deviation = 0 if i == 0 else solver.value(variables[f"dev_{i}"])
            print(f"  {i} - {products_prop_fact[i]}  penalty {penalty} count {solver.value(product_counts[i])} deviation {deviation}")

    elif status == cp_model.INFEASIBLE:
        print(solver.ResponseStats())  # Основные статистические данные

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts: {solver.num_conflicts}")
    print(f"  - branches : {solver.num_branches}")
    print(f"  - wall time: {solver.wall_time}s")


if __name__ == "__main__":
    main()