/******************************************************************************
 * graph_complement.cpp
 *
 * Converts a graph G into its complement G' (in METIS format).
 * MIS(G) == MaxClique(G'), so you can run a max clique solver on G'.
 *
 * Usage: graph_complement INPUT_FILE OUTPUT_FILE
 *
 * WARNING: The complement of a sparse graph is dense.
 *   A graph with n nodes gets up to n*(n-1)/2 edges.
 *   For n=10000 that's ~50M edges. Use with care on large graphs.
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_set>

#include "data_structure/graph_access.h"
#include "io/graph_io.h"

int main(int argn, char **argv) {
        if (argn != 3) {
                std::cout << "Usage: graph_complement INPUT_FILE OUTPUT_FILE" << std::endl;
                std::cout << std::endl;
                std::cout << "Computes the complement graph G' of G." << std::endl;
                std::cout << "MIS(G) = MaxClique(G'), so you can run a max clique solver on G'." << std::endl;
                exit(1);
        }

        std::string input_filename(argv[1]);
        std::string output_filename(argv[2]);

        // Read the input graph
        graph_access G;
        int ret = graph_io::readGraphWeighted(G, input_filename);
        if (ret != 0) {
                std::cerr << "Error reading graph from " << input_filename << std::endl;
                return 1;
        }

        NodeID n = G.number_of_nodes();
        EdgeID m = G.number_of_edges(); // directed edge count (each undirected edge counted twice)

        long long max_directed_edges = (long long)n * (long long)(n - 1);
        long long complement_directed_edges = max_directed_edges - (long long)m;
        long long complement_undirected_edges = complement_directed_edges / 2;

        std::cout << "Input graph:      " << n << " nodes, " << m / 2 << " undirected edges" << std::endl;
        std::cout << "Complement graph: " << n << " nodes, " << complement_undirected_edges << " undirected edges" << std::endl;

        if (complement_undirected_edges > 500000000LL) {
                std::cerr << "WARNING: Complement has > 500M edges. This will use a lot of memory." << std::endl;
        }

        // Build adjacency set for each node for O(1) lookup
        std::vector<std::unordered_set<NodeID>> adj(n);
        for (NodeID u = 0; u < n; u++) {
                for (EdgeID e = G.get_first_edge(u); e < G.get_first_invalid_edge(u); e++) {
                        adj[u].insert(G.getEdgeTarget(e));
                }
        }

        // Construct the complement graph
        graph_access G_complement;
        G_complement.start_construction(n, complement_directed_edges);

        for (NodeID u = 0; u < n; u++) {
                NodeID node = G_complement.new_node();
                G_complement.setNodeWeight(node, G.getNodeWeight(u));
                G_complement.setPartitionIndex(node, 0);

                // Add edges to all non-neighbors (excluding self)
                for (NodeID v = 0; v < n; v++) {
                        if (v == u) continue;
                        if (adj[u].find(v) == adj[u].end()) {
                                EdgeID e = G_complement.new_edge(node, v);
                                G_complement.setEdgeWeight(e, 1);
                        }
                }
        }

        G_complement.finish_construction();

        // Write the complement graph
        ret = graph_io::writeGraph(G_complement, output_filename);
        if (ret != 0) {
                std::cerr << "Error writing graph to " << output_filename << std::endl;
                return 1;
        }

        std::cout << "Complement graph written to " << output_filename << std::endl;

        return 0;
}
