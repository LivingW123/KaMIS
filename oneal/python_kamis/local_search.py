from greedy_mis import greedy_mis
import random

def local_search_improvement(num_nodes, adj, current_mis):
    """
    Python port of basic Local Search strategies from local_search.cpp (1-swap direct improvement).
    
    It attempts to find vertices currently in the independent set, remove them, 
    and insert two new non-adjacent vertices from its neighborhood to increase 
    the size of the independent set in O(|MIS| * D^2) steps where D is max degree.
    """
    
    is_solution = [False] * num_nodes
    for node in current_mis:
        is_solution[node] = True
        
    improved = True
    while improved:
        improved = False
        # Shuffle order for stochastic exploration
        mis_nodes = list(current_mis)
        random.shuffle(mis_nodes)
        
        for u in mis_nodes:
            # 1-tight property: A node not in MIS has exactly 1 neighbor in MIS.
            # Building 1-tight neighborhood (vertices adjacent to `u` that only have `u` as an MIS neighbor)
            onetight_neighbors = []
            for v in adj[u]:
                if is_solution[v]:
                    continue
                # Count neighbors of v in MIS
                mis_neighbors = sum(1 for w in adj[v] if is_solution[w])
                if mis_neighbors == 1: # Only connected to 'u'
                    onetight_neighbors.append(v)
            
            # If we can find two 1-tight neighbors that aren't adjacent to each other,
            # we can remove 'u' and add both of them (net gain of +1 in MIS size).
            if len(onetight_neighbors) >= 2:
                for i in range(len(onetight_neighbors)):
                    for j in range(i + 1, len(onetight_neighbors)):
                        v1 = onetight_neighbors[i]
                        v2 = onetight_neighbors[j]
                        if v2 not in adj[v1]:
                            # Perfect! Do the swap.
                            current_mis.remove(u)
                            is_solution[u] = False
                            
                            current_mis.append(v1)
                            current_mis.append(v2)
                            is_solution[v1] = True
                            is_solution[v2] = True
                            
                            improved = True
                            break
                    if improved: break
            if improved: break 

    return current_mis
