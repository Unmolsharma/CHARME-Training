#!/usr/bin/env python3
"""Generate orderlist.pkl via Order Exploration (Algorithms 2+3 from paper).

This is the best possible implementation without the original notebook.
It faithfully implements Algorithms 2 and 3 from Section 3.4 of the paper,
including ATOM calls for exact qubit cost computation (F(O)) and the
lower bound pruning from Theorem 1.

Paper settings:
    --sampling_limit 10000 --exploration_limit 1000000 --n_threads 1024

Quick smoke test:
    --sampling_limit 50 --exploration_limit 100 --n_threads 4

Usage:
    python scripts/02_generate_orderlist.py \
        --training_data_dir training_data \
        --hw_topo_row 45 --hw_topo_col 45 --hw_bipart_cell 4 \
        --sampling_limit 10000 --exploration_limit 1000000 \
        --n_threads 1024 \
        --atom_binary ../charme-rl/ours/atom_system \
        --out_path orderlist.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from charme.utils import generate_Chimera
from charme.env import MinorEmbeddingEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training_data_dir", type=str, default="training_data")
    p.add_argument("--hw_topo_row", type=int, default=45)
    p.add_argument("--hw_topo_col", type=int, default=45)
    p.add_argument("--hw_bipart_cell", type=int, default=4)
    p.add_argument("--sampling_limit", type=int, default=10000,
                   help="D in Algorithm 2")
    p.add_argument("--exploration_limit", type=int, default=1000000,
                   help="K in Algorithm 2 (recursive steps per order)")
    p.add_argument("--n_threads", type=int, default=8)
    p.add_argument("--atom_binary", type=str,
                   default="../charme-rl/ours/atom_system")
    p.add_argument("--out_path", type=str, default="orderlist.pkl")
    return p.parse_args()


# ── ATOM wrapper for cost computation ─────────────────────────────────────────

def run_atom_full_order(graph, order, topo_row, topo_col, bipart_cell,
                        atom_binary, seed=0):
    """
    Run ATOM following a given embedding order and return total qubit count F(O).
    This is the exact F(O) computation needed by Algorithm 2 and Algorithm 3.
    """
    # Create a minimal env just for ATOM calls
    env = MinorEmbeddingEnv(
        topo_row=topo_row,
        topo_col=topo_col,
        bipart_cell=bipart_cell,
        goal_dim=1,
        num_nodes=graph.number_of_nodes(),
        n_state=1,
        seed=seed,
        degree=3,
        training_size=1,
        orderlist_path="/dev/null",
        atom_binary_path=atom_binary,
        mode=1,
    )
    env.load_graph([graph], [[]])
    env.chimera_graph = env.original_chimera_graph.copy()
    for node in env.chimera_graph.nodes:
        env.chimera_graph.nodes[node]['embedding'] = -1

    # Init with ATOM mode=0
    embedding, rr, cc, _ = env.call_atom(graph, topo_row, topo_col, seed, 0)
    env.update_hw([], embedding)
    env.embedding = embedding
    env.curr_row = rr
    env.curr_column = cc

    g = graph.copy()
    already_embedded = {emb[3] for emb in embedding}

    for node in order:
        if node in already_embedded:
            continue
        new_emb, rr, cc, old_node = env.call_atom(
            g, env.curr_row, env.curr_column, seed, 1, node, env.embedding
        )
        for on in old_node:
            if g.has_edge(on, node):
                g.remove_edge(on, node)
        env.curr_row = rr
        env.curr_column = cc
        try:
            env.update_hw(env.embedding, new_emb)
        except Exception:
            return float('inf')
        env.embedding = new_emb
        already_embedded.add(node)

    return len(env.embedding)


def compute_lower_bound(graph, prefix, current_emb_size, topo_row, topo_col,
                        bipart_cell, atom_binary, seed=0):
    """
    Theorem 1 lower bound: F_bar = |phi(t-1)| + sum_v C(v|t) for all v not in prefix.
    C(v|t) = cost to embed v at current step t (before any more nodes are embedded).
    """
    env = MinorEmbeddingEnv(
        topo_row=topo_row, topo_col=topo_col, bipart_cell=bipart_cell,
        goal_dim=1, num_nodes=graph.number_of_nodes(), n_state=1,
        seed=seed, degree=3, training_size=1,
        orderlist_path="/dev/null", atom_binary_path=atom_binary, mode=1,
    )
    env.load_graph([graph], [[]])
    env.chimera_graph = env.original_chimera_graph.copy()
    for node in env.chimera_graph.nodes:
        env.chimera_graph.nodes[node]['embedding'] = -1

    embedding, rr, cc, _ = env.call_atom(graph, topo_row, topo_col, seed, 0)
    env.update_hw([], embedding)
    env.embedding = embedding
    env.curr_row = rr
    env.curr_column = cc

    # Embed the prefix
    g = graph.copy()
    already = {emb[3] for emb in embedding}
    for node in prefix:
        if node in already:
            continue
        new_emb, rr, cc, old_node = env.call_atom(
            g, env.curr_row, env.curr_column, seed, 1, node, env.embedding
        )
        for on in old_node:
            if g.has_edge(on, node):
                g.remove_edge(on, node)
        env.curr_row = rr
        env.curr_column = cc
        try:
            env.update_hw(env.embedding, new_emb)
        except Exception:
            return float('inf')
        env.embedding = new_emb
        already.add(node)

    # Now compute C(v|t) for each remaining node at current step t
    lower_bound = len(env.embedding)
    remaining = [n for n in graph.nodes if n not in set(prefix)]

    for v in remaining:
        new_emb, _, _, old_node = env.call_atom(
            g, env.curr_row, env.curr_column, seed, 1, v, env.embedding
        )
        cost = len(new_emb) - len(env.embedding)
        lower_bound += cost

    return lower_bound


def order_refining(graph, topo_row, topo_col, bipart_cell, atom_binary,
                   prefix, current_emb, k, threshold, seed=0):
    """
    Algorithm 3: Order Refining.
    Recursively explores suffixes to find one with F(O) < threshold.
    Returns (success, suffix, remaining_k).
    """
    nodes = list(graph.nodes)
    embedded = set(prefix)
    remaining = [n for n in nodes if n not in embedded]

    # Base case: all nodes embedded
    if not remaining:
        cost = len(current_emb)
        if cost < threshold:
            return True, [], k - 1
        else:
            return False, [], k - 1

    if k <= 0:
        return False, [], 0

    # Theorem 1 lower bound pruning (Algorithm 3, lines 7-14)
    lower_bound = compute_lower_bound(
        graph, prefix, len(current_emb),
        topo_row, topo_col, bipart_cell, atom_binary, seed
    )
    if lower_bound >= threshold:
        return False, [], k - 1

    # Random node selection (Algorithm 3, lines 16-21)
    random.shuffle(remaining)
    k_remaining = k

    for v in remaining:
        if k_remaining <= 0:
            break
        success, suffix, k_remaining = order_refining(
            graph, topo_row, topo_col, bipart_cell, atom_binary,
            prefix + [v], current_emb,
            k_remaining - 1, threshold, seed
        )
        if success:
            return True, [v] + suffix, k_remaining

    return False, [], 0


def explore_single_graph(args):
    """Worker function for parallel order exploration (Algorithm 2 inner loop)."""
    (graph_idx, graph, baseline_cost, current_order,
     topo_row, topo_col, bipart_cell, atom_binary,
     sampling_limit, exploration_limit, seed) = args

    best_order = current_order[:]
    best_score = baseline_cost

    for _ in range(min(sampling_limit, 20)):  # per-graph budget
        threshold = best_score
        success, new_order, _ = order_refining(
            graph, topo_row, topo_col, bipart_cell, atom_binary,
            [], [], exploration_limit, threshold, seed
        )
        if success and new_order:
            # Pad with remaining nodes
            embedded = set(new_order)
            remaining = [n for n in graph.nodes if n not in embedded]
            full_order = new_order + remaining
            # Compute actual score
            actual_score = run_atom_full_order(
                graph, full_order, topo_row, topo_col, bipart_cell, atom_binary, seed
            )
            if actual_score < best_score:
                best_score = actual_score
                best_order = full_order

    return graph_idx, best_order, best_score


def main():
    args = parse_args()
    data_dir = Path(args.training_data_dir)

    # Load graphs
    graphs = []
    i = 0
    while True:
        path = data_dir / f"graph_{i}.txt"
        if not path.exists():
            break
        G = nx.read_edgelist(str(path), nodetype=int)
        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)
        i += 1
    n_graphs = len(graphs)
    print(f"Loaded {n_graphs} training graphs")

    # Load minorminer baselines (these are F(O_bar) from Algorithm 2)
    mm_path = data_dir / "minorminer_results.pth"
    with open(mm_path, "rb") as f:
        mm_results = pickle.load(f)

    def baseline_cost(mm):
        if isinstance(mm, dict):
            return sum(len(v) for v in mm.values())
        return len(mm)

    baseline_costs = [baseline_cost(mm) for mm in mm_results]
    print(f"Baseline costs: min={min(baseline_costs)}, max={max(baseline_costs)}, "
          f"avg={sum(baseline_costs)/len(baseline_costs):.1f}")

    # Algorithm 2, line 1: initialise orders as random permutations
    order_list = []
    for g in graphs:
        nodes = list(g.nodes)
        random.shuffle(nodes)
        order_list.append(nodes)

    # Algorithm 2, line 2: potential scores mu_i = |V_H| initially
    chimera = generate_Chimera(args.hw_topo_row, args.hw_topo_col, args.hw_bipart_cell)
    V_H = chimera.number_of_nodes()
    potential_scores = [float(V_H)] * n_graphs

    print(f"\nRunning Order Exploration (Algorithm 2):")
    print(f"  D={args.sampling_limit}, K={args.exploration_limit}, threads={args.n_threads}")
    print(f"  Hardware: {V_H} nodes\n")

    # Algorithm 2, lines 3-8: main sampling loop
    with ProcessPoolExecutor(max_workers=args.n_threads) as executor:
        futures = {}
        total_score = sum(potential_scores)

        for step in tqdm(range(args.sampling_limit), desc="Order Exploration"):
            # Equation 13: selection probability proportional to potential score
            probs = np.array(potential_scores) / total_score
            selected_idx = int(np.random.choice(n_graphs, p=probs))

            worker_args = (
                selected_idx, graphs[selected_idx],
                baseline_costs[selected_idx], order_list[selected_idx],
                args.hw_topo_row, args.hw_topo_col, args.hw_bipart_cell,
                args.atom_binary, 10, args.exploration_limit, 0
            )
            future = executor.submit(explore_single_graph, worker_args)
            futures[future] = selected_idx

            # Collect completed futures
            done = [f for f in list(futures.keys()) if f.done()]
            for f in done:
                idx, new_order, new_score = f.result()
                if new_order and new_score < baseline_costs[idx]:
                    order_list[idx] = new_order
                    # Equation 14: mu_i = F(O_i) / F(O_bar_i)
                    potential_scores[idx] = new_score / max(1, baseline_costs[idx])
                    total_score = sum(potential_scores)
                del futures[f]

    out_path = Path(args.out_path)
    with open(out_path, "wb") as f:
        pickle.dump(order_list, f)

    print(f"\nSaved orderlist ({n_graphs} orderings) to: {out_path}")
    print("\nNext: python train.py")


if __name__ == "__main__":
    main()
