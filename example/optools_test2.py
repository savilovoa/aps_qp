"""Linear assignment example."""
from ortools.graph.python import min_cost_flow


def main():
    """Solving an Assignment Problem with MinCostFlow."""
    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()


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
    num_days = 3
    all_machines = range(num_machines)
    all_products = range(num_products)
    all_days = range(num_days)



    # # Define the directed graph for the flow.
    # start_nodes = (
    #     [0, 0, 0, 0] + [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4] + [5, 6, 7, 8]
    # )
    # end_nodes = (
    #     [1, 2, 3, 4] + [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8] + [9, 9, 9, 9]
    # )
    # capacities = (
    #     [1, 1, 1, 1] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]
    # )
    # costs = (
    #     [0, 0, 0, 0]
    #     + [90, 76, 75, 70, 35, 85, 55, 65, 125, 95, 90, 105, 45, 110, 95, 115]
    #     + [0, 0, 0, 0]
    # )

    source = 0
    sink = 9
    tasks = 4
    max_qty = num_machines * num_days * 21
    supplies = [tasks, 0, 0, 0, 0, 0, 0, 0, 0, -tasks]



    # Add each arc.
    for i in range(len(start_nodes)):
        smcf.add_arc_with_capacity_and_unit_cost(
            start_nodes[i], end_nodes[i], capacities[i], costs[i]
        )
    # Add node supplies.
    for i in range(len(supplies)):
        smcf.set_node_supply(i, supplies[i])

    # Find the minimum cost flow between node 0 and node 10.
    status = smcf.solve()

    if status == smcf.OPTIMAL:
        print("Total cost = ", smcf.optimal_cost())
        print()
        for arc in range(smcf.num_arcs()):
            # Can ignore arcs leading out of source or into sink.
            if smcf.tail(arc) != source and smcf.head(arc) != sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.
                if smcf.flow(arc) > 0:
                    print(
                        "Worker %d assigned to task %d.  Cost = %d"
                        % (smcf.tail(arc), smcf.head(arc), smcf.unit_cost(arc))
                    )
    else:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")


if __name__ == "__main__":
    main()