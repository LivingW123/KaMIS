import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# -----------------------------
# Load your data
# -----------------------------
with open("results.json", "r") as f:
    data = json.load(f)

# -----------------------------
# Aggregate runtimes
# -----------------------------
# solver_data[solver][n] = [runtimes...]
solver_data = defaultdict(lambda: defaultdict(list))

for instance_name, instance in data.items():
    n = instance["n"]   # ✅ FIX: use actual field instead of parsing name
    
    for result in instance["results"]:
        solver = result["solver"]
        runtime = result["runtime"]
        
        solver_data[solver][n].append(runtime)

# -----------------------------
# Compute averages
# -----------------------------
avg_solver_data = {}

for solver, n_dict in solver_data.items():
    avg_solver_data[solver] = {}
    for n, runtimes in n_dict.items():
        avg_solver_data[solver][n] = np.mean(runtimes)

# -----------------------------
# Plot
# -----------------------------
plt.figure()

for solver, n_dict in avg_solver_data.items():
    ns = sorted(n_dict.keys())
    runtimes = [n_dict[n] for n in ns]
    print(runtimes)
    plt.plot(ns, runtimes, marker='o', label=solver)

plt.xlabel("n (number of nodes)")
plt.ylabel("Average Runtime (seconds)")
plt.title("MIS Solver Runtime vs Graph Size")
plt.legend()
plt.grid()

plt.show()