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
        ("87416", 6, [(0, 1), (0, 7)] )
        , ("18305", 3, [(1, 1), (0, 6)])
        , ("18302", 1, [(2, 1), (0, 7)])
    ]
    num_machines = 5
    num_products = len(products)
    num_days = 7
    all_machines = range(num_machines)
    all_products = range(num_products+1)
    all_days = range(num_days)
    clear_idx = num_products


    # Create the model.
    model = cp_model.CpModel()

    shifts = {}
    machine_product = {}
    for m in all_machines:
        for d in all_days:
            for p in all_products:
                shifts[(m, d, p)] = model.new_bool_var(f"shift_m{m}_d{d}_p{p}")


    # На одной машине один продукт в день
    for d in all_days:
        for m in all_machines:
            model.add_at_most_one(shifts[(m, d, p)] for p in all_products)

    # # На одной машине один продукт в день
    # for d in all_days:
    #     for p in all_products:
    #         model.add_exactly_one(shifts[(m, d, p)] for m in all_machines)

    # Добавляем переход
    # for m in all_machines:
    #     for p in range(num_products):
    #         for d in range(1, num_days):
    #             item_pred = shifts[(m, d-1, p)]
    #             item_clear = shifts[(m, d-1, clear_idx)]
    #             model.add(item_pred == 1)
    #             #model.add(item_clear == 1)


    products_sum = []
    all_sum = 0
    for p in range(num_products):
        all_sum += products[p][1]
        for m in all_machines:
            for d in all_days:
                products_sum.append(shifts[(m, d, p)])


    for p in range(num_products):
        k = products[p][1] / all_sum
        k = int(k * num_days * num_machines)
        products_sum_one = []
        for m in all_machines:
            for d in all_days:
                products_sum_one.append(shifts[(m, d, p)])

        if p < num_products:
            model.add(sum(products_sum_one) >= k - 1)
            print(f"k({p}) = {k-1}")

        #model.add(sum(products_sum_one) >= k)
        #     z[p] = model.new_int_var(-100, 100, "z")
    #     model.add_modulo_equality(x[p], sum(products_sum_one), sum(products_sum))
    #     model.add_abs_equality(z[p], k - x[p])
     #model.minimize(sum(z))

    # Objective
    objective_terms = []

    for m in all_machines:
        for d in all_days:
            for p in range(num_products):
                objective_terms.append(shifts[(m,d,p)])

    model.maximize(sum(objective_terms))

    # objective_terms2 = []
    # for m in all_machines:
    #     for d in all_days:
    #         objective_terms2.append(shifts[m,d,clear_idx,])
    # model.minimize(sum(objective_terms2))

    # Creates the solver and solve.


    # solver.parameters.linearization_level = 0
    # # Enumerate all solutions.
    # solver.parameters.enumerate_all_solutions = True

    # class NursesPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
    #     """Print intermediate solutions."""
    #
    #     def __init__(self, shifts, num_machine, num_days, num_products, limit):
    #         cp_model.CpSolverSolutionCallback.__init__(self)
    #         self._shifts = shifts
    #         self._num_machine = num_machine
    #         self._num_days = num_days
    #         self._num_products = num_products
    #         self._solution_count = 0
    #         self._solution_limit = limit
    #
    #     def on_solution_callback(self):
    #         self._solution_count += 1
    #         print(f"Solution {self._solution_count}")
    #         for d in range(self._num_days):
    #             print(f"Day {d}")
    #             for m in range(self._num_machine):
    #                 is_working = False
    #                 for p in range(self._num_products):
    #                     if self.value(self._shifts[(m, d, p)]):
    #                         is_working = True
    #                         print(f"  Loom {m} works  {p}")
    #         if self._solution_count >= self._solution_limit:
    #             print(f"Stop search after {self._solution_limit} solutions")
    #             self.stop_search()
    #
    #     def solutionCount(self):
    #         return self._solution_count
    #
    # # Display the first five solutions.
    # solution_limit = 5
    # solution_printer = NursesPartialSolutionPrinter(
    #     shifts, num_machines, num_days, num_products + 1, solution_limit
    # )
    status = solver.solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for d in range(num_days):
            print(f"Day {d}")
            for m in range(num_machines):
                is_working = False
                for p in range(num_products):
                    if solver.value(shifts[(m, d, p)]):
                        is_working = True
                        print(f"  Loom {m} works  {p}")
        for p in range(num_products):
            p_sum_one = 0
            for m in all_machines:
                for d in all_days:
                    if solver.value(shifts[(m, d, p)]):
                        p_sum_one += 1

            print(f" sum {p} {p_sum_one}")

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts: {solver.num_conflicts}")
    print(f"  - branches : {solver.num_branches}")
    print(f"  - wall time: {solver.wall_time}s")


if __name__ == "__main__":
    main()