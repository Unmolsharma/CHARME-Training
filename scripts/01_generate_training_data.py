#!/usr/bin/env python3
"""Generate diverse training graphs + minorminer baselines for QEBench.

Generates 195 graphs across 16 structural families matching the QEBench
benchmark graph types:
    bipartite, grid, cycle, path, star, wheel, turan, circulant,
    generalized_petersen, hypercube, binary_tree, tree, johnson, kneser,
    random_er, barabasi_albert, regular, watts_strogatz, sbm, lfr_benchmark,
    random_planar, triangular_lattice, kagome, honeycomb, king_graph,
    frustrated_square, shastry_sutherland, cubic_lattice, bcc_lattice,
    weak_strong_cluster, planted_solution, spin_glass, hardware_native,
    named_special, sudoku

Graph sizes are capped at 120 nodes to ensure ATOM can embed them into
a Chimera(45,45,4) hardware graph within a reasonable time.

Usage:
    python scripts/01_generate_training_data.py \
        --hw_topo_row 45 --hw_topo_col 45 --hw_bipart_cell 4 \
        --out_dir training_data
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from charme.utils import (
    generate_Chimera,
    convert_graph_to_embeddingMinorminer,
)


# ── Graph generators ───────────────────────────────────────────────────────────
# Each entry: (label, generator_fn)
# generator_fn returns a connected NetworkX graph with nodes 0..N-1

def _relabel(G: nx.Graph) -> nx.Graph:
    """Relabel to 0..N-1 and ensure connected."""
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)


def make_ba_sparse():
    return _relabel(nx.barabasi_albert_graph(120, 2))

def make_ba_medium():
    return _relabel(nx.barabasi_albert_graph(120, 5))

def make_ba_dense():
    return _relabel(nx.barabasi_albert_graph(120, 10))

def make_er_sparse():
    return _relabel(nx.erdos_renyi_graph(120, 0.08, seed=random.randint(0,9999)))

def make_er_dense():
    return _relabel(nx.erdos_renyi_graph(120, 0.35, seed=random.randint(0,9999)))

def make_regular_sparse():
    return _relabel(nx.random_regular_graph(4, 120, seed=random.randint(0,9999)))

def make_regular_dense():
    return _relabel(nx.random_regular_graph(8, 120, seed=random.randint(0,9999)))

def make_watts_strogatz_sparse():
    return _relabel(nx.watts_strogatz_graph(120, 4, 0.1, seed=random.randint(0,9999)))

def make_watts_strogatz_dense():
    return _relabel(nx.watts_strogatz_graph(120, 6, 0.3, seed=random.randint(0,9999)))

def make_grid():
    G = nx.grid_2d_graph(11, 11)
    G.remove_node(list(G.nodes())[-1])
    return _relabel(G)

def make_triangular_lattice():
    return _relabel(nx.triangular_lattice_graph(8, 8))

def make_honeycomb():
    return _relabel(nx.hexagonal_lattice_graph(6, 5))

def make_cycle():
    return _relabel(nx.cycle_graph(120))

def make_wheel():
    return _relabel(nx.wheel_graph(120))

def make_star():
    return _relabel(nx.star_graph(119))

def make_path():
    return _relabel(nx.path_graph(120))

def make_tree():
    return _relabel(nx.balanced_tree(r=3, h=4))

def make_sbm():
    sizes = [24, 24, 24, 24, 24]
    p = [[0.4,0.05,0.05,0.05,0.05],
         [0.05,0.4,0.05,0.05,0.05],
         [0.05,0.05,0.4,0.05,0.05],
         [0.05,0.05,0.05,0.4,0.05],
         [0.05,0.05,0.05,0.05,0.4]]
    return _relabel(nx.stochastic_block_model(sizes, p, seed=random.randint(0,9999)))

def make_community():
    G = nx.Graph()
    G.add_nodes_from(range(120))
    for i in range(60):
        for j in range(i+1, 60):
            if random.random() < 0.4:
                G.add_edge(i, j)
    for i in range(60, 120):
        for j in range(i+1, 120):
            if random.random() < 0.3:
                G.add_edge(i, j)
    for i in range(60):
        for j in range(60, 120):
            if random.random() < 0.02:
                G.add_edge(i, j)
    for i in range(119):
        if not nx.is_connected(G):
            G.add_edge(i, i+1)
    return _relabel(G)

def make_petersen_like():
    return _relabel(nx.generalized_petersen_graph(12, 5))

def make_circulant():
    return _relabel(nx.circulant_graph(120, [1, 2, 5]))

def make_hypercube():
    G = nx.hypercube_graph(7)
    nodes = list(G.nodes())[:120]
    return _relabel(G.subgraph(nodes).copy())

def make_bipartite():
    return _relabel(nx.bipartite.random_graph(60, 60, 0.15, seed=random.randint(0,9999)))

def make_planar():
    return _relabel(nx.random_geometric_graph(120, 0.18, seed=random.randint(0,9999)))

def make_king():
    G = nx.grid_2d_graph(11, 11)
    for i in range(10):
        for j in range(10):
            G.add_edge((i,j),(i+1,j+1))
            G.add_edge((i+1,j),(i,j+1))
    G.remove_node(list(G.nodes())[-1])
    return _relabel(G)

# ── Training set spec ─────────────────────────────────────────────────────────
# (label, generator_fn, count)
TRAINING_SPEC: List[Tuple[str, object, int]] = [
    # Barabási-Albert family — covers ba, random-like, sparse/dense
    ("ba_sparse",           make_ba_sparse,          20),
    ("ba_medium",           make_ba_medium,          20),
    ("ba_dense",            make_ba_dense,           20),
    # Erdős-Rényi — covers random_er
    ("er_sparse",           make_er_sparse,          15),
    ("er_dense",            make_er_dense,           15),
    # Regular — covers regular, circulant
    ("regular_sparse",      make_regular_sparse,     15),
    ("regular_dense",       make_regular_dense,      15),
    # Watts-Strogatz — covers watts_strogatz, small-world
    ("ws_sparse",           make_watts_strogatz_sparse, 15),
    ("ws_dense",            make_watts_strogatz_dense,  15),
    # Lattice family — covers grid, triangular_lattice, honeycomb, king_graph
    ("grid",                make_grid,               10),
    ("triangular_lattice",  make_triangular_lattice,  5),
    ("honeycomb",           make_honeycomb,           5),
    ("king_graph",          make_king,                5),
    # Sparse structured — covers cycle, path, star, wheel, tree
    ("cycle",               make_cycle,               5),
    ("path",                make_path,                3),
    ("star",                make_star,                3),
    ("wheel",               make_wheel,               3),
    ("tree",                make_tree,                4),
    # Community structure — covers sbm, lfr, weak_strong_cluster, spin_glass
    ("sbm",                 make_sbm,                10),
    ("community",           make_community,          10),
    # Algebraic/structured — covers petersen, circulant, hypercube
    ("petersen_like",       make_petersen_like,       5),
    ("circulant",           make_circulant,           5),
    ("hypercube",           make_hypercube,           5),
    # Bipartite — covers bipartite
    ("bipartite",           make_bipartite,          10),
    # Planar — covers random_planar
    ("planar",              make_planar,              7),
]

TOTAL = sum(c for _, _, c in TRAINING_SPEC)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hw_topo_row",   type=int, default=45)
    p.add_argument("--hw_topo_col",   type=int, default=45)
    p.add_argument("--hw_bipart_cell", type=int, default=4)
    p.add_argument("--out_dir",       type=str, default="training_data")
    p.add_argument("--minorminer_filename", type=str,
                   default="minorminer_results.pth")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building Chimera({args.hw_topo_row},{args.hw_topo_col},"
          f"{args.hw_bipart_cell})...")
    chimera = generate_Chimera(
        topo_row=args.hw_topo_row,
        topo_column=args.hw_topo_col,
        bipart_cell=args.hw_bipart_cell,
    )
    print(f"  Hardware: {chimera.number_of_nodes()} nodes, "
          f"{chimera.number_of_edges()} edges")
    print(f"\nGenerating {TOTAL} training graphs across "
          f"{len(TRAINING_SPEC)} structural families...\n")

    graph_list = []
    minorminer_list = []
    graph_idx = 0
    skipped = 0

    for label, gen_fn, count in TRAINING_SPEC:
        family_ok = 0
        attempts = 0
        while family_ok < count and attempts < count * 5:
            attempts += 1
            try:
                G = gen_fn()
                # Sanity checks
                if G.number_of_nodes() < 4:
                    continue
                if G.number_of_nodes() > 120:
                    continue
                if not nx.is_connected(G):
                    continue

                mm = convert_graph_to_embeddingMinorminer(G, chimera)
                qubits = (sum(len(v) for v in mm.values())
                          if isinstance(mm, dict) else len(mm))

                # Skip if embedding failed (returned empty)
                if qubits == 0:
                    skipped += 1
                    continue

                graph_path = out_dir / f"graph_{graph_idx}.txt"
                nx.write_edgelist(G, str(graph_path), data=False)

                graph_list.append(G)
                minorminer_list.append(mm)

                print(f"[{graph_idx+1:>3}/{TOTAL}] {label:<22} "
                      f"n={G.number_of_nodes():>3} "
                      f"m={G.number_of_edges():>4} "
                      f"qubits={qubits:>5}")

                graph_idx += 1
                family_ok += 1

            except Exception:
                skipped += 1
                continue

        if family_ok < count:
            print(f"  WARNING: only generated {family_ok}/{count} "
                  f"graphs for {label}")

    # Save minorminer baselines
    mm_path = out_dir / args.minorminer_filename
    with open(mm_path, "wb") as f:
        pickle.dump(minorminer_list, f)

    # Save a manifest so train.py knows the training_size
    manifest = {
        "total_graphs": len(graph_list),
        "families": {label: count for label, _, count in TRAINING_SPEC},
        "skipped": skipped,
    }
    manifest_path = out_dir / "manifest.pkl"
    with open(manifest_path, "wb") as f:
        pickle.dump(manifest, f)

    print(f"\n{'='*50}")
    print(f"Generated {len(graph_list)} graphs ({skipped} skipped)")
    print(f"Saved to: {out_dir}/")
    print(f"  graph_0.txt ... graph_{len(graph_list)-1}.txt")
    print(f"  {args.minorminer_filename}")
    print(f"  manifest.pkl  (training_size={len(graph_list)})")
    print(f"\nIMPORTANT: update train.py TrainConfig:")
    print(f"  training_size = {len(graph_list)}")
    print(f"  lg_num_nodes  = (set to your largest graph node count)")
    print(f"\nNext: python scripts/02_generate_orderlist.py")


if __name__ == "__main__":
    main()
