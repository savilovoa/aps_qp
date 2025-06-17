import collections
from ortools.sat.python import cp_model
from ortools.graph.python import min_cost_flow

def main() -> None:
    """Minimal jobshop problem."""
    # Data.
    # массив остатков
    remains = [
        [1, 2, 2] # reed
        , [47] # shaft
    ]

    products = [  # (name, proccess_time, qty, [(reed_idx, qty), (shaft_idx, qty)]).
        ("87416", 18.8, 5000, [(0, 1), (0, 7)] )
        , ("18305", 20.5, 5000, [(1, 1), (0, 6)])
        , ("18302", 20, 5000, [(2, 1), (0, 7)])
    ]
    num_machines = 5
    num_products = len(products)
    num_days = 7
    all_machines = range(num_machines)
    all_products = range(num_products)
    all_days = range(num_days)

    supplers = []
    smcf = min_cost_flow.SimpleMinCostFlow

    # Create the model.
    model = cp_model.CpModel()

    for


if __name__ == "__main__":
    main()