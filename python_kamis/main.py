import time
import os
import sys

from graph_io import read_graph
from greedy_mis import greedy_mis
from local_search import local_search_improvement

def run_evaluation(graph_path):
    print(f"====================================")
    print(f"Evaluating {os.path.basename(graph_path)}")
    print(f"====================================")
    
    # Reading
    t0 = time.perf_counter()
    num_nodes, adj, weights = read_graph(graph_path)
    t1 = time.perf_counter()
    print(f"Nodes: {num_nodes}, Edges: {sum(len(n) for n in adj)//2}")
    
    # Python Greedy Array Iteration
    t2 = time.perf_counter()
    initial_mis = greedy_mis(num_nodes, adj)
    t3 = time.perf_counter()
    
    # Python Local Search (1-swaps)
    t4 = time.perf_counter()
    final_mis = local_search_improvement(num_nodes, adj, initial_mis)
    t5 = time.perf_counter()

    # Complexity comparisons & Benchmarking
    print("\n--- Python Implementations Runtimes ---")
    print(f"1. Graph Load Time: {(t1 - t0)*1000:.3f} ms")
    print(f"2. Greedy MIS Time: {(t3 - t2)*1000:.3f} ms   | Output Size: {len(initial_mis)}")
    print(f"3. Local Search Time: {(t5 - t4)*1000:.3f} ms | Output Size: {len(final_mis)}")

    print("\n--- Time Complexity Ratios: C++ vs Python ---")
    print("1. Greedy MIS (C++): O(V + E) using bucket queues.")
    print("2. Greedy MIS (Py) : O((V + E) log V) using heapq. Expected ratio: Python is a factor of ~log V slower asymptotically + 10x-50x interpreted overhead.")
    print("   -> Why? Python doesn't have a native O(1) bucket-decrease-key queue, but heaps simulate it efficiently.")
    print("3. Local Search (C++): Uses Bitsets and cache-friendly precomputed tight degrees.")
    print("4. Local Search (Py) : Unoptimized arrays and hash lookups. Expected ratio: Python is a factor of 100x+ slower due to memory indirection overheads.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(base_dir, 'examples')
    
    for graph_file in os.listdir(examples_dir):
        if graph_file.endswith(".graph"):
            run_evaluation(os.path.join(examples_dir, graph_file))
