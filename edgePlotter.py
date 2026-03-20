import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# -----------------------------
# Load data
# -----------------------------
with open("results.json", "r") as f:
    data = json.load(f)

# -----------------------------
# Aggregate runtimes
# solver_data[solver][num_edges] = [runtimes...]
# -----------------------------
solver_data = defaultdict(lambda: defaultdict(list))

for instance_name, instance in data.items():
    edges = instance["num_edges"]

    for result in instance["results"]:
        solver = result["solver"]
        runtime = result["runtime"]
        solver_data[solver][edges].append(runtime)

# -----------------------------
# Compute averages + plot
# -----------------------------
plt.figure()

for solver, edge_dict in solver_data.items():
    edges_sorted = sorted(edge_dict.keys())
    avg_runtimes = [np.mean(edge_dict[e]) for e in edges_sorted]

    plt.plot(edges_sorted, avg_runtimes, marker='o', label=solver)

plt.xlabel("Number of edges")
plt.ylabel("Average runtime (seconds)")
plt.title("Runtime vs Number of Edges")
plt.legend()
plt.grid()

plt.show()