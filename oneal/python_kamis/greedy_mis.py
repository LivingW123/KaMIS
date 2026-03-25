import heapq

def greedy_mis(num_nodes, adj):
    """
    Python implementation of kaMIS greedy_mis.cpp
    
    It iteratively selects the vertex with the minimum residual degree,
    adds it to the maximal independent set, and removes its neighbors.
    
    Time Complexity:
    - O((V + E) * log(V)) using a Priority Queue (heapq)
    (The C++ version uses an array of buckets because maximum degree is bounded by V-1,
     enabling O(V + E) complexity. Python uses a heap for a close asymptotic match).
    """
    degree = [len(n_adj) for n_adj in adj]
    active = [True] * num_nodes
    
    # Priority queue: (degree, node)
    pq = []
    for i in range(num_nodes):
        heapq.heappush(pq, (degree[i], i))
        
    independent_set = []
    
    while pq:
        d, u = heapq.heappop(pq)
        
        # If node is dead or degree is outdated, ignore it
        if not active[u]:
            continue
        if d != degree[u]:
            continue
            
        # Add u to independent set
        independent_set.append(u)
        active[u] = False
        
        # Remove all neighbors and update adjacent degrees
        for v in adj[u]:
            if active[v]:
                active[v] = False
                for w in adj[v]:
                    if active[w]:
                        degree[w] -= 1
                        heapq.heappush(pq, (degree[w], w))
                        
    return independent_set
