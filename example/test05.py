from ortools.sat.python import cp_model

# Create a CP-SAT model
model = cp_model.CpModel()

# Define the variables
x = model.NewIntVar(0, 10, "x")  # Index variable
target_value = model.NewIntVar(0, 100, "target_value") # Target variable

# Define a list of possible values (can be constants or other IntVar instances)
predefined_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

# Add the AddElement constraint: target_value should be equal to predefined_values[x]
model.AddElement(x, predefined_values, target_value)

# Add a constraint to fix the index variable for demonstration
#model.Add(x == 5)
model.Maximize(target_value)

# Create a solver and solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Print the results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"Solution found:")
    print(f"x = {solver.Value(x)}")
    print(f"target_value = {solver.Value(target_value)}")
else:
    print("No solution found.")