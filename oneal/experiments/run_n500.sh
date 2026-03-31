#!/bin/bash
set -e
cd /mnt/c/Users/oneal/Projects/KaMIS

DEPLOY="$(pwd)/deploy"
RESULTS="$(pwd)/oneal/benchmark_results/exp7_raw.csv"
TMP="/tmp/exp7_n500_$$"
mkdir -p "$TMP"
TIME_LIMIT=30

# Generate n=500 instances
python3 -c "
import sys, math
sys.path.insert(0, 'oneal')
from mis_benchmark_combined import gen_erdos_renyi_planted, write_metis
for p in [0.2, 0.5]:
    for seed in [42, 123, 456, 789, 1337]:
        inst = gen_erdos_renyi_planted(500, int(math.sqrt(500)), p, seed)
        tag = f'er_n500_p{p}_s{seed}'
        f = '$TMP/' + tag + '.graph'
        write_metis(inst.graph, f)
        print(f'{tag} m={inst.graph.number_of_edges()}')
"

echo "=== n=500 solvers ==="
for gf in ${TMP}/*.graph; do
    tag=$(basename "$gf" .graph)
    echo "--- $tag ---"

    # MaxCLQ
    OF="${TMP}/${tag}_maxclq.sol"
    START_NS=$(date +%s%N)
    timeout $((TIME_LIMIT + 5)) "$DEPLOY/max_clique" "$gf" --complement \
        --time_limit=$TIME_LIMIT --output "$OF" 2>/dev/null | grep -E "Maximum|Status" || true
    END_NS=$(date +%s%N)
    ELAPSED_NS=$((END_NS - START_NS))
    ELAPSED_S=$(python3 -c "print(f'{${ELAPSED_NS}/1e9:.2f}')")
    SZ=0; [ -f "$OF" ] && SZ=$(grep -c "^1$" "$OF" || echo 0)
    echo "${tag},MaxCLQ_complement,${SZ},${ELAPSED_S},done" >> "$RESULTS"
    echo "  MaxCLQ: size=$SZ t=${ELAPSED_S}s"

    # redumis
    OF="${TMP}/${tag}_redumis.sol"
    timeout $((TIME_LIMIT + 5)) "$DEPLOY/redumis" "$gf" \
        --time_limit=$TIME_LIMIT --seed=42 --output="$OF" --console_log 2>/dev/null >/dev/null || true
    SZ=0; [ -f "$OF" ] && SZ=$(grep -c "^1$" "$OF" || echo 0)
    echo "${tag},KaMIS_redumis,${SZ},${TIME_LIMIT},done" >> "$RESULTS"
    echo "  redumis: size=$SZ"

    # online_mis
    OF="${TMP}/${tag}_arw.sol"
    timeout $((TIME_LIMIT + 5)) "$DEPLOY/online_mis" "$gf" \
        --time_limit=$TIME_LIMIT --seed=42 --output="$OF" --console_log 2>/dev/null >/dev/null || true
    SZ=0; [ -f "$OF" ] && SZ=$(grep -c "^1$" "$OF" || echo 0)
    echo "${tag},KaMIS_online_mis,${SZ},${TIME_LIMIT},done" >> "$RESULTS"
    echo "  online_mis: size=$SZ"
done

rm -rf "$TMP"
echo "=== n=500 done ==="
