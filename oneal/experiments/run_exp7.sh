#!/bin/bash
# run_exp7.sh — Run MaxCLQ vs KaMIS solvers on dense graphs
# Execute from WSL: bash oneal/experiments/run_exp7.sh

set -e
cd "$(dirname "$0")/../.."

DEPLOY="$(pwd)/deploy"
RESULTS="$(pwd)/oneal/benchmark_results/exp7_raw.csv"
TMP="/tmp/exp7_$$"
mkdir -p "$TMP" "$(dirname "$RESULTS")"

TIME_LIMIT=30

echo "instance,solver,size,time,status" > "$RESULTS"

# Generate instances and run all solvers
python3 -c "
import sys, os, math
sys.path.insert(0, 'oneal')
from mis_benchmark_combined import gen_erdos_renyi_planted, write_metis

for n in [100, 200, 500]:
    for p in [0.2, 0.5]:
        for seed in [42, 123, 456, 789, 1337]:
            h = int(math.sqrt(n))
            inst = gen_erdos_renyi_planted(n, h, p, seed)
            tag = f'er_n{n}_p{p}_s{seed}'
            f = '${TMP}/' + tag + '.graph'
            write_metis(inst.graph, f)
            print(f'{tag} {n} {inst.graph.number_of_edges()}')
"

echo "=== Running solvers ==="

for gf in ${TMP}/*.graph; do
    tag=$(basename "$gf" .graph)
    echo "--- $tag ---"

    # MaxCLQ (complement)
    OF="${TMP}/${tag}_maxclq.sol"
    T0=$(date +%s%N)
    MCOUT=$(timeout $((TIME_LIMIT + 5)) "$DEPLOY/max_clique" "$gf" --complement \
        --time_limit=$TIME_LIMIT --output "$OF" 2>/dev/null)
    T1=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($T1 - $T0) / 1000000000" | bc)
    echo "$MCOUT" | grep -E "Maximum|Status" || true
    # Count 1s in solution
    if [ -f "$OF" ]; then
        SZ=$(grep -c "^1$" "$OF" || echo 0)
    else
        SZ=0
    fi
    echo "${tag},MaxCLQ_complement,${SZ},${ELAPSED},done" >> "$RESULTS"
    echo "  MaxCLQ: size=$SZ t=${ELAPSED}s"

    # redumis
    OF="${TMP}/${tag}_redumis.sol"
    timeout $((TIME_LIMIT + 5)) "$DEPLOY/redumis" "$gf" \
        --time_limit=$TIME_LIMIT --seed=42 --output="$OF" --console_log 2>/dev/null >/dev/null || true
    if [ -f "$OF" ]; then
        SZ=$(grep -c "^1$" "$OF" || echo 0)
    else
        SZ=0
    fi
    echo "${tag},KaMIS_redumis,${SZ},${TIME_LIMIT},done" >> "$RESULTS"
    echo "  redumis: size=$SZ"

    # online_mis (ARW)
    OF="${TMP}/${tag}_arw.sol"
    timeout $((TIME_LIMIT + 5)) "$DEPLOY/online_mis" "$gf" \
        --time_limit=$TIME_LIMIT --seed=42 --output="$OF" --console_log 2>/dev/null >/dev/null || true
    if [ -f "$OF" ]; then
        SZ=$(grep -c "^1$" "$OF" || echo 0)
    else
        SZ=0
    fi
    echo "${tag},KaMIS_online_mis,${SZ},${TIME_LIMIT},done" >> "$RESULTS"
    echo "  online_mis: size=$SZ"
done

echo ""
echo "=== Results ==="
cat "$RESULTS"
echo ""
echo "Results saved to: $RESULTS"

# Run Python HILS + analysis
python3 -c "
import sys, os, csv, math, time
sys.path.insert(0, 'oneal')
from mis_benchmark_combined import gen_erdos_renyi_planted
from hils import red_hils_weighted, HilsConfig

cfg = HilsConfig(max_iter=2_000_000, time_limit=${TIME_LIMIT}, seed=42)
results_file = '${RESULTS}'

# Read existing results
with open(results_file) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Run HILS on same instances
new_rows = []
for n in [100, 200, 500]:
    for p in [0.2, 0.5]:
        for seed in [42, 123, 456, 789, 1337]:
            tag = f'er_n{n}_p{p}_s{seed}'
            h = int(math.sqrt(n))
            inst = gen_erdos_renyi_planted(n, h, p, seed)
            G = inst.graph
            w1 = {v: 1.0 for v in G.nodes()}
            t0 = time.time()
            sol, _ = red_hils_weighted(G, w1, cfg)
            rt = time.time() - t0
            new_rows.append([tag, 'Python_RedHILS', str(len(sol)), f'{rt:.2f}', 'done'])
            print(f'  RedHILS {tag}: size={len(sol)} t={rt:.1f}s')

# Append HILS results
with open(results_file, 'a') as f:
    writer = csv.writer(f)
    for row in new_rows:
        writer.writerow(row)

# Analysis
print()
print('=' * 70)
print('  COMPARISON TABLE: Solution Size')
print('=' * 70)
with open(results_file) as f:
    reader = csv.reader(f)
    next(reader)
    all_rows = list(reader)

from collections import defaultdict
import numpy as np

by_config = defaultdict(lambda: defaultdict(list))
for tag, solver, sz, t, status in all_rows:
    parts = tag.split('_')
    n = int(parts[1][1:])
    p = parts[2][1:]
    key = f'n={n:4d} p={p}'
    by_config[key][solver].append(int(sz))

solvers = ['MaxCLQ_complement', 'KaMIS_redumis', 'KaMIS_online_mis', 'Python_RedHILS']
header = f\"{'':>16s}\" + ''.join(f'{s:>20s}' for s in solvers)
print(header)
print('-' * len(header))
for key in sorted(by_config.keys()):
    row = f'{key:>16s}'
    for s in solvers:
        vals = by_config[key][s]
        if vals:
            row += f'{np.mean(vals):>14.1f}+/-{np.std(vals):.1f}'
        else:
            row += f\"{'n/a':>20s}\"
    print(row)

# Win/loss
print()
print('--- Head-to-head vs MaxCLQ ---')
by_tag = defaultdict(dict)
for tag, solver, sz, t, status in all_rows:
    by_tag[tag][solver] = int(sz)

for opp in ['KaMIS_redumis', 'KaMIS_online_mis', 'Python_RedHILS']:
    w, t, l = 0, 0, 0
    for tag, sv in by_tag.items():
        if 'MaxCLQ_complement' in sv and opp in sv:
            mc, op = sv['MaxCLQ_complement'], sv[opp]
            if mc > op: w += 1
            elif mc == op: t += 1
            else: l += 1
    total = w + t + l
    if total:
        print(f'  vs {opp:22s}: W={w} T={t} L={l} ({100*w//total}%/{100*t//total}%/{100*l//total}%)')

# Runtime comparison
print()
print('--- Runtime (seconds) ---')
by_config_t = defaultdict(lambda: defaultdict(list))
for tag, solver, sz, t, status in all_rows:
    parts = tag.split('_')
    n = int(parts[1][1:])
    p = parts[2][1:]
    key = f'n={n:4d} p={p}'
    by_config_t[key][solver].append(float(t))

header = f\"{'':>16s}\" + ''.join(f'{s:>20s}' for s in solvers)
print(header)
print('-' * len(header))
for key in sorted(by_config_t.keys()):
    row = f'{key:>16s}'
    for s in solvers:
        vals = by_config_t[key][s]
        if vals:
            row += f'{np.mean(vals):>16.1f}s'
        else:
            row += f\"{'n/a':>20s}\"
    print(row)
"

rm -rf "$TMP"
