import os

def read_graph(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    lines = [L.strip() for L in lines if L.strip() and L[0] != '%']
    
    first_line = lines[0].split()
    num_nodes = int(first_line[0])
    num_edges = int(first_line[1])
    
    format_type = 0
    if len(first_line) > 2:
        format_type = int(first_line[2])
    
    read_ew = format_type in [1, 11]
    read_nw = format_type in [10, 11]
    
    adj = [[] for _ in range(num_nodes)]
    node_weights = [1] * num_nodes
    
    for i in range(1, len(lines)):
        node_id = i - 1
        parts = lines[i].split()
        if not parts:
            continue
            
        start_idx = 0
        if read_nw:
            node_weights[node_id] = int(parts[0])
            start_idx = 1
            
        while start_idx < len(parts):
            target = int(parts[start_idx]) - 1 # 1-based METIS index to 0-based python index
            weight = 1
            if read_ew:
                weight = int(parts[start_idx + 1])
                start_idx += 2
            else:
                start_idx += 1
                
            adj[node_id].append(target)
            
    return num_nodes, adj, node_weights
