import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import math

with open("results.json", "r") as f:
    data = json.load(f)

solver_data = defaultdict(lambda: defaultdict(list))

for instance_name, instance in data.items():
    n = instance["n"]
    edges = instance["num_edges"]

    for result in instance["results"]:
        solver = result["solver"]
        runtime = result["runtime"]
        solver_data[solver][(n, edges)].append(runtime)

# -----------------------------
# Prepare subplot grid
# -----------------------------
solvers = list(solver_data.keys())
num_solvers = len(solvers)

cols = 2
rows = math.ceil(num_solvers / cols)

fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
axes = axes.flatten()  # make indexing easy

# -----------------------------
# Plot heatmaps
# -----------------------------
for idx, solver in enumerate(solvers):
    ax = axes[idx]
    point_dict = solver_data[solver]

    ns = sorted(set(n for (n, _) in point_dict.keys()))
    es = sorted(set(e for (_, e) in point_dict.keys()))

    grid = np.zeros((len(es), len(ns)))

    for i, e in enumerate(es):
        for j, n in enumerate(ns):
            if (n, e) in point_dict:
                grid[i, j] = np.log10(np.mean(point_dict[(n, e)]))
            else:
                grid[i, j] = np.nan

    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',
        extent=[min(ns), max(ns), min(es), max(es)]
    )

    ax.set_title(solver)
    ax.set_xlabel("n")
    ax.set_ylabel("edges")

    # add colorbar per subplot
    fig.colorbar(im, ax=ax, label="log10(runtime)", fraction=0.046, pad=0.04)

# -----------------------------
# Remove unused subplots
# -----------------------------
for i in range(len(solvers), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()