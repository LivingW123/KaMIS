/******************************************************************************
 * max_clique.cpp
 *
 * MaxCLQ: Maximum Clique solver using MaxSAT-based branch and bound.
 *
 * Usage: max_clique INPUT_FILE [--complement] [--output FILE]
 *
 *   --complement  Complement the graph internally (solves MIS via max clique)
 *   --output FILE Write solution (1/0 per vertex) to FILE
 *
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <queue>
#include <set>
#include <cstring>
#include <climits>
#include <chrono>
#include <cassert>

#include "data_structure/graph_access.h"
#include "io/graph_io.h"

using namespace std;

// ═══ Dynamic Bitset ═══════════════════════════════════════════════════════

static int NWORDS; // global word count for all bitsets

struct Bitset {
        int n;
        vector<uint64_t> w;

        Bitset() : n(0) {}
        explicit Bitset(int n_) : n(n_), w(NWORDS, 0ULL) {}

        void set(int i)        { w[i >> 6] |=  (1ULL << (i & 63)); }
        void reset(int i)      { w[i >> 6] &= ~(1ULL << (i & 63)); }
        bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }

        int count() const {
                int c = 0;
                for (int i = 0; i < NWORDS; i++)
                        c += __builtin_popcountll(w[i]);
                return c;
        }

        bool empty() const {
                for (int i = 0; i < NWORDS; i++)
                        if (w[i]) return false;
                return true;
        }

        int first() const {
                for (int i = 0; i < NWORDS; i++)
                        if (w[i]) return (i << 6) + __builtin_ctzll(w[i]);
                return -1;
        }

        Bitset operator&(const Bitset& o) const {
                Bitset r(n);
                for (int i = 0; i < NWORDS; i++) r.w[i] = w[i] & o.w[i];
                return r;
        }
        Bitset operator~() const {
                Bitset r(n);
                for (int i = 0; i < NWORDS; i++) r.w[i] = ~w[i];
                if (n & 63) r.w[NWORDS - 1] &= (1ULL << (n & 63)) - 1;
                return r;
        }
        Bitset& operator&=(const Bitset& o) {
                for (int i = 0; i < NWORDS; i++) w[i] &= o.w[i];
                return *this;
        }
        Bitset& operator|=(const Bitset& o) {
                for (int i = 0; i < NWORDS; i++) w[i] |= o.w[i];
                return *this;
        }

        template <typename F>
        void for_each(F&& f) const {
                for (int i = 0; i < NWORDS; i++) {
                        uint64_t val = w[i];
                        while (val) {
                                f((i << 6) + __builtin_ctzll(val));
                                val &= val - 1;
                        }
                }
        }

        vector<int> to_vector() const {
                vector<int> v;
                for_each([&](int i) { v.push_back(i); });
                return v;
        }
};

// ═══ Globals ══════════════════════════════════════════════════════════════

static int N;
static vector<Bitset> ADJ;     // adjacency bitsets
static int best_size;
static vector<int> best_clique;
static long long nodes_explored;
static chrono::steady_clock::time_point start_time;
static double time_limit_sec;
static bool timed_out;

// ═══ Greedy Coloring ══════════════════════════════════════════════════════
// Partition 'verts' into independent sets (= graph coloring).
// Vertices ordered by decreasing degree (Max encoding from the paper).
// Number of color classes = upper bound on clique number.

struct ColorClass {
        Bitset members;
        vector<int> verts;
        ColorClass() : members(N) {}
};

static vector<ColorClass> greedy_color(const Bitset& verts) {
        vector<int> order = verts.to_vector();

        // Precompute degrees within subgraph
        vector<int> deg(N, 0);
        for (int v : order)
                deg[v] = (ADJ[v] & verts).count();

        // Sort by decreasing degree
        sort(order.begin(), order.end(), [&](int a, int b) {
                return deg[a] > deg[b];
        });

        vector<ColorClass> colors;

        for (int v : order) {
                bool placed = false;
                for (auto& cc : colors) {
                        // v can join this color class if no neighbor of v is in it
                        if ((ADJ[v] & cc.members).empty()) {
                                cc.members.set(v);
                                cc.verts.push_back(v);
                                placed = true;
                                break;
                        }
                }
                if (!placed) {
                        colors.emplace_back();
                        colors.back().members.set(v);
                        colors.back().verts.push_back(v);
                }
        }
        return colors;
}

// ═══ Failed Literal Detection ═════════════════════════════════════════════
// Returns true if setting x_v = 1 leads to a contradiction via unit
// propagation on the hard clauses (non-edges → can't both be in clique)
// and remaining soft clauses (color classes).

static bool is_failed_literal(
        int v,
        const Bitset& active,
        vector<ColorClass>& classes,
        vector<bool>& removed,
        vector<int>& consumed_indices
) {
        int k = (int)classes.size();

        // Working copies of soft clauses: remaining literals
        vector<Bitset> remaining(k, Bitset(N));
        vector<int> rem_count(k, 0);

        // Map: variable → which soft clauses contain it
        vector<vector<int>> var_in_clause(N);

        for (int i = 0; i < k; i++) {
                if (removed[i]) continue;
                remaining[i] = classes[i].members & active;
                rem_count[i] = remaining[i].count();
                remaining[i].for_each([&](int u) {
                        var_in_clause[u].push_back(i);
                });
        }

        Bitset is_true(N), is_false(N);
        queue<int> forced_true, forced_false;
        set<int> used_soft;

        is_true.set(v);
        forced_true.push(v);

        while (!forced_true.empty() || !forced_false.empty()) {
                // Forced-true: hard clause propagation
                // x=1 → all non-neighbors must be 0
                while (!forced_true.empty()) {
                        int x = forced_true.front(); forced_true.pop();

                        // Non-neighbors of x within active set
                        Bitset non_nbrs = active & ~ADJ[x];
                        non_nbrs.reset(x);

                        non_nbrs.for_each([&](int u) {
                                if (!is_false.test(u)) {
                                        is_false.set(u);
                                        forced_false.push(u);
                                }
                        });

                        // Contradiction: variable forced both true and false
                        if (!(is_true & is_false).empty()) {
                                consumed_indices.assign(used_soft.begin(), used_soft.end());
                                return true;
                        }
                }

                // Forced-false: remove from soft clauses, check for unit/empty
                while (!forced_false.empty()) {
                        int x = forced_false.front(); forced_false.pop();

                        for (int si : var_in_clause[x]) {
                                if (removed[si]) continue;
                                if (!remaining[si].test(x)) continue;

                                remaining[si].reset(x);
                                rem_count[si]--;

                                if (rem_count[si] == 0) {
                                        // Empty clause → contradiction
                                        used_soft.insert(si);
                                        consumed_indices.assign(used_soft.begin(), used_soft.end());
                                        return true;
                                }
                                if (rem_count[si] == 1) {
                                        // Unit clause → force remaining literal true
                                        int w = remaining[si].first();
                                        used_soft.insert(si);

                                        if (is_false.test(w)) {
                                                consumed_indices.assign(used_soft.begin(), used_soft.end());
                                                return true;
                                        }
                                        if (!is_true.test(w)) {
                                                is_true.set(w);
                                                forced_true.push(w);
                                        }
                                }
                        }
                }
        }

        return false; // no contradiction
}

// ═══ Overestimation (Upper Bound) ═════════════════════════════════════════
// Algorithm 2 from the paper: partition + MaxSAT reasoning.
// Returns upper bound on max clique in subgraph induced by 'verts'.

static int overestimation(const Bitset& verts) {
        if (verts.empty()) return 0;

        // Step 1: Greedy coloring → k independent sets (soft clauses)
        vector<ColorClass> classes = greedy_color(verts);
        int k = (int)classes.size();

        if (k <= 1) return k;

        // Step 2: Find disjoint inconsistent subsets via failed literal detection
        vector<bool> tested(k, false);
        vector<bool> removed(k, false);
        int s = 0;

        // Process soft clauses in order of increasing size (smallest first)
        while (true) {
                int best = -1, best_sz = INT_MAX;
                for (int i = 0; i < k; i++) {
                        if (!tested[i] && !removed[i]) {
                                int sz = (int)classes[i].verts.size();
                                if (sz < best_sz) {
                                        best_sz = sz;
                                        best = i;
                                }
                        }
                }
                if (best == -1) break;

                tested[best] = true;

                // Check if EVERY literal in this clause is a failed literal
                Bitset active_in_clause = classes[best].members & verts;
                vector<int> literals = active_in_clause.to_vector();

                bool all_failed = true;
                vector<vector<int>> per_literal_consumed;

                for (int lit : literals) {
                        vector<int> consumed;
                        if (!is_failed_literal(lit, verts, classes, removed, consumed)) {
                                all_failed = false;
                                break;
                        }
                        per_literal_consumed.push_back(consumed);
                }

                if (all_failed) {
                        // Remove this clause and all clauses consumed in contradictions
                        removed[best] = true;
                        for (auto& cs : per_literal_consumed)
                                for (int idx : cs)
                                        removed[idx] = true;
                        s++;
                }
        }

        return k - s;
}

// ═══ MaxCLQ Branch and Bound ══════════════════════════════════════════════
// Algorithm 1 from the paper.

static void maxclq(const Bitset& verts, vector<int>& current_clique) {
        nodes_explored++;

        // Time limit check
        if (time_limit_sec > 0 && (nodes_explored & 0xFF) == 0) {
                auto now = chrono::steady_clock::now();
                double elapsed = chrono::duration<double>(now - start_time).count();
                if (elapsed >= time_limit_sec) {
                        timed_out = true;
                        return;
                }
        }
        if (timed_out) return;

        if (verts.empty()) {
                if ((int)current_clique.size() > best_size) {
                        best_size = (int)current_clique.size();
                        best_clique = current_clique;
                        cerr << "  New best: " << best_size << endl;
                }
                return;
        }

        // Upper bound pruning
        int ub = overestimation(verts) + (int)current_clique.size();
        if (best_size >= ub) return;

        // Select branching vertex: minimum degree in current subgraph
        int v = -1, min_deg = INT_MAX;
        verts.for_each([&](int u) {
                int d = (ADJ[u] & verts).count();
                if (d < min_deg) { min_deg = d; v = u; }
        });

        // Branch 1: cliques containing v → recurse on N(v) ∩ verts
        {
                Bitset new_verts = ADJ[v] & verts;
                current_clique.push_back(v);
                maxclq(new_verts, current_clique);
                current_clique.pop_back();
        }

        // Branch 2: cliques NOT containing v → recurse on verts \ {v}
        {
                Bitset new_verts = verts;
                new_verts.reset(v);
                maxclq(new_verts, current_clique);
        }
}

// ═══ Main ═════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
        if (argc < 2) {
                cout << "Usage: max_clique INPUT_FILE [--complement] [--output FILE]" << endl;
                cout << endl;
                cout << "  --complement    Complement graph internally (solves MIS via max clique)" << endl;
                cout << "  --output FILE   Write 0/1 solution vector to FILE" << endl;
                cout << "  --time_limit=N  Time limit in seconds (0 = unlimited, default)" << endl;
                exit(1);
        }

        string filename(argv[1]);
        bool complement = false;
        string output_file;
        time_limit_sec = 0;

        for (int i = 2; i < argc; i++) {
                string arg(argv[i]);
                if (arg == "--complement") {
                        complement = true;
                } else if (arg == "--output" && i + 1 < argc) {
                        output_file = string(argv[++i]);
                } else if (arg.rfind("--time_limit=", 0) == 0) {
                        time_limit_sec = atof(arg.substr(13).c_str());
                }
        }

        // Read graph
        graph_access G;
        if (graph_io::readGraphWeighted(G, filename) != 0) {
                cerr << "Error reading " << filename << endl;
                return 1;
        }

        N = (int)G.number_of_nodes();
        NWORDS = (N + 63) >> 6;

        cout << "Graph: " << N << " nodes, " << G.number_of_edges() / 2 << " edges" << endl;
        if (complement) cout << "Mode: complement (solving MIS)" << endl;

        // Build adjacency bitsets
        ADJ.assign(N, Bitset(N));

        if (complement) {
                // Build original adjacency, then complement it
                vector<Bitset> orig(N, Bitset(N));
                for (NodeID u = 0; u < (NodeID)N; u++)
                        for (EdgeID e = G.get_first_edge(u); e < G.get_first_invalid_edge(u); e++)
                                orig[u].set(G.getEdgeTarget(e));
                for (int u = 0; u < N; u++) {
                        ADJ[u] = ~orig[u];
                        ADJ[u].reset(u); // no self-loops
                }
                long long comp_edges = 0;
                for (int u = 0; u < N; u++) comp_edges += ADJ[u].count();
                cout << "Complement: " << comp_edges / 2 << " edges" << endl;
        } else {
                for (NodeID u = 0; u < (NodeID)N; u++)
                        for (EdgeID e = G.get_first_edge(u); e < G.get_first_invalid_edge(u); e++)
                                ADJ[u].set(G.getEdgeTarget(e));
        }

        // Solve
        best_size = 0;
        nodes_explored = 0;
        timed_out = false;

        if (time_limit_sec > 0)
                cout << "Time limit: " << time_limit_sec << "s" << endl;

        auto t0 = chrono::steady_clock::now();
        start_time = t0;

        Bitset all_verts(N);
        for (int i = 0; i < N; i++) all_verts.set(i);

        vector<int> clique;
        maxclq(all_verts, clique);

        auto t1 = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();

        cout << endl;
        if (complement)
                cout << "Maximum Independent Set size: " << best_size << endl;
        else
                cout << "Maximum Clique size: " << best_size << endl;

        cout << "Vertices (1-indexed): ";
        sort(best_clique.begin(), best_clique.end());
        for (int v : best_clique) cout << v + 1 << " ";
        cout << endl;
        cout << "Nodes explored: " << nodes_explored << endl;
        cout << "Time: " << elapsed << "s" << endl;
        if (timed_out)
                cout << "Status: TIMED_OUT (best so far)" << endl;
        else
                cout << "Status: OPTIMAL" << endl;

        // Write solution file (0/1 per vertex)
        if (!output_file.empty()) {
                ofstream out(output_file);
                vector<bool> in_solution(N, false);
                for (int v : best_clique) in_solution[v] = true;
                for (int i = 0; i < N; i++)
                        out << (in_solution[i] ? 1 : 0) << "\n";
                out.close();
                cout << "Solution written to " << output_file << endl;
        }

        return 0;
}
