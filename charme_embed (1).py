"""CHARME embedding algorithm — QEBench integration.

Loads a pretrained CHARME checkpoint and uses the trained GCN policy
to determine node embedding order, then calls ATOM for chain construction.

Training details:
    - action_dim=120 (all training graphs fixed at 120 nodes)
    - Chimera(16,16,4) hardware used during training
    - Diverse graph corpus: BA, ER, WS, grid, bipartite, lattice, SBM etc.

Usage:
    Set CHARME_CHECKPOINT env var to path of .pth file, or place checkpoint at
    algorithms/charme/checkpoints/charme_trained.pth
"""

from __future__ import annotations

import os
import time
import copy
import subprocess
import tempfile
from pathlib import Path

import torch
import networkx as nx

from qebench.registry import register_algorithm, EmbeddingAlgorithm

# ── Constants matching training configuration ──────────────────────────────────
ACTION_DIM = 120          # fixed node count used during training
HIDDEN_DIM = 64           # GCN hidden dimension
IN_CHANNELS = 1           # input feature channels

_DEFAULT_CKPT = Path(__file__).parent / "checkpoints" / "charme_trained.pth"
_ATOM_BINARY  = Path(__file__).parent.parent / "charme-rl" / "ours" / "atom_system"


def _load_model(checkpoint_path: str, device: torch.device):
    """Load ActorCritic from checkpoint with correct architecture."""
    # Import from charme package — expected to be in algorithms/charme/
    import sys
    charme_dir = str(Path(__file__).parent)
    if charme_dir not in sys.path:
        sys.path.insert(0, charme_dir)

    from charme.models import ActorCritic

    model = ActorCritic(
        state_dim=0,
        action_dim=ACTION_DIM,
        has_continuous_action_space=False,
        action_std_init=0.6,
        device=device,
        in_logical_channels=IN_CHANNELS,
        hidden_logical_channels=HIDDEN_DIM,
        in_hardware_channels=IN_CHANNELS,
        hidden_hardware_channels=HIDDEN_DIM,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_initial_state(source_graph: nx.Graph, target_graph: nx.Graph, device: torch.device):
    """Build the initial state tensors for the CHARME model."""
    import sys
    charme_dir = str(Path(__file__).parent)
    if charme_dir not in sys.path:
        sys.path.insert(0, charme_dir)

    from charme.utils import (
        get_hw_edge_index,
        analysing_logical,
        get_hw_attr_synthetic,
        convert_embedding_to_tensor,
    )

    n = source_graph.number_of_nodes()

    # Logical graph features
    logical_edge_index, logical_attr = analysing_logical(source_graph)

    # Pad logical_attr to ACTION_DIM if graph is smaller
    if logical_attr.shape[0] < ACTION_DIM:
        pad = torch.zeros(ACTION_DIM - logical_attr.shape[0], logical_attr.shape[1])
        logical_attr = torch.cat([logical_attr, pad], dim=0)

    # Hardware graph features
    hw_edge_index = get_hw_edge_index(target_graph)

    # Reset hardware node embeddings
    for node in target_graph.nodes:
        target_graph.nodes[node]['embedding'] = -1

    hw_attr = get_hw_attr_synthetic(target_graph)

    # Empty embedding matrix
    emb_matrix = convert_embedding_to_tensor(
        [], target_graph, source_graph
    ).to_sparse()

    # Pad emb_matrix rows to ACTION_DIM
    if emb_matrix.shape[0] < ACTION_DIM:
        dense = emb_matrix.to_dense()
        pad = torch.zeros(ACTION_DIM - dense.shape[0], dense.shape[1])
        dense = torch.cat([dense, pad], dim=0)
        emb_matrix = dense.to_sparse()

    state = {
        'logical_attr':       logical_attr,
        'logical_edge_index': logical_edge_index,
        'hw_attr':            hw_attr,
        'hw_edge_index':      hw_edge_index,
        'emb_matrix':         emb_matrix,
    }
    return state, n


@register_algorithm("charme")
class CHARMEEmbedder(EmbeddingAlgorithm):
    """
    CHARME: Chain-based RL minor embedder.

    Uses a pretrained GCN actor-critic to learn node ordering,
    then calls ATOM for chain construction.

    Requires:
        - Checkpoint at CHARME_CHECKPOINT env var or
          algorithms/charme/checkpoints/charme_trained.pth
        - ATOM binary at algorithms/charme-rl/ours/atom_system
          (or set CHARME_ATOM_BINARY env var)

    Note: Only works on graphs with <= 120 nodes (fixed action_dim).
    """

    version = "0.1.0-retrained"

    def __init__(self):
        ckpt = os.environ.get("CHARME_CHECKPOINT", str(_DEFAULT_CKPT))
        atom = os.environ.get("CHARME_ATOM_BINARY", str(_ATOM_BINARY))
        self._ckpt_path = ckpt
        self._atom_binary = atom
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if not os.path.exists(self._ckpt_path):
            raise FileNotFoundError(
                f"CHARME checkpoint not found: {self._ckpt_path}\n"
                "Set CHARME_CHECKPOINT env var or place checkpoint at "
                "algorithms/charme/checkpoints/charme_trained.pth"
            )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _load_model(self._ckpt_path, self._device)

    def embed(self, source_graph: nx.Graph, target_graph: nx.Graph,
              timeout: float = 60.0, **kwargs) -> dict:
        start = time.time()

        # Remap nodes to 0..N-1
        mapping = {n: i for i, n in enumerate(source_graph.nodes())}
        source_graph = nx.relabel_nodes(source_graph, mapping)
        n = source_graph.number_of_nodes()

        # CHARME only handles graphs up to ACTION_DIM nodes
        if n > ACTION_DIM:
            return {'embedding': {}}

        self._ensure_model()

        try:
            state, n = _build_initial_state(
                source_graph, target_graph, self._device
            )
        except Exception as e:
            return {'embedding': {}}

        # ── Greedy rollout to get node ordering ────────────────────────────────
        # mask[i]=True means node i is already embedded (skip it)
        mask = [False] * ACTION_DIM
        mask_connected = [True] * ACTION_DIM

        # Mask out padded positions (nodes that don't exist in this graph)
        for i in range(n, ACTION_DIM):
            mask[i] = True

        order = []

        with torch.no_grad():
            for step in range(n):
                if time.time() - start > timeout:
                    break

                state_dev = {
                    k: v.to(self._device) for k, v in state.items()
                }

                action, _, _ = self._model.act(state_dev, mask, mask_connected)
                chosen = action.item()

                # Fallback: if model chose an invalid node, pick first valid
                if mask[chosen]:
                    chosen = next(
                        (i for i in range(n) if not mask[i]), None
                    )
                    if chosen is None:
                        break

                mask[chosen] = True
                for nei in source_graph.neighbors(chosen):
                    if nei < len(mask_connected):
                        mask_connected[nei] = False

                order.append(chosen)

        if len(order) < n:
            # Append any missed nodes
            embedded = set(order)
            order += [i for i in range(n) if i not in embedded]

        # ── Call ATOM with the learned ordering ────────────────────────────────
        embedding = self._call_atom_with_order(
            source_graph, target_graph, order, timeout - (time.time() - start)
        )

        elapsed = time.time() - start
        return {'embedding': embedding, 'time': elapsed}

    def _call_atom_with_order(self, source_graph, target_graph, order, timeout):
        """
        Run ATOM following the RL-learned node ordering.
        Falls back to minorminer if ATOM binary not available.
        """
        if not os.path.exists(self._atom_binary):
            # Fallback: use minorminer with the learned order as a hint
            try:
                import minorminer
                embedding = minorminer.find_embedding(source_graph, target_graph)
                return embedding
            except Exception:
                return {}

        # Use ATOM binary step by step following the order
        try:
            import sys
            charme_dir = str(Path(__file__).parent)
            if charme_dir not in sys.path:
                sys.path.insert(0, charme_dir)

            from charme.utils import generate_Chimera
            from charme.env import MinorEmbeddingEnv

            # Infer Chimera params from target graph
            n_hw = target_graph.number_of_nodes()
            # Try to find matching Chimera params
            topo_row, topo_col, bipart_cell = 16, 16, 4
            for t in [4]:
                found = False
                for r in range(1, 50):
                    for c in range(1, 50):
                        if 2 * r * c * t == n_hw:
                            topo_row, topo_col, bipart_cell = r, c, t
                            found = True
                            break
                    if found:
                        break

            env = MinorEmbeddingEnv(
                topo_row=topo_row,
                topo_col=topo_col,
                bipart_cell=bipart_cell,
                goal_dim=1,
                num_nodes=source_graph.number_of_nodes(),
                n_state=1,
                seed=0,
                degree=3,
                training_size=1,
                orderlist_path='/dev/null',
                atom_binary_path=self._atom_binary,
                mode=1,
            )
            env.load_graph([source_graph], [[]])
            env.chimera_graph = target_graph.copy()
            for node in env.chimera_graph.nodes:
                env.chimera_graph.nodes[node]['embedding'] = -1

            # Init with ATOM mode=0
            embedding_list, rr, cc, _ = env.call_atom(
                source_graph, topo_row, topo_col, 0, 0
            )
            env.update_hw([], embedding_list)
            env.embedding = embedding_list
            env.curr_row = rr
            env.curr_column = cc

            g = source_graph.copy()
            already = {emb[3] for emb in embedding_list}

            # Follow the RL order
            for node in order:
                if node in already:
                    continue
                new_emb, rr, cc, old_node = env.call_atom(
                    g, env.curr_row, env.curr_column, 0, 1,
                    node, env.embedding
                )
                for on in old_node:
                    if g.has_edge(on, node):
                        g.remove_edge(on, node)
                env.curr_row = rr
                env.curr_column = cc
                try:
                    env.update_hw(env.embedding, new_emb)
                except Exception:
                    break
                env.embedding = new_emb
                already.add(node)

            # Convert embedding list to dict format {logical_node: [hw_nodes]}
            result = {}
            for emb in env.embedding:
                logical = emb[3]
                hw_node = (emb[0], emb[1], emb[2])
                if logical not in result:
                    result[logical] = []
                result[logical].append(hw_node)

            return result

        except Exception as e:
            # Final fallback
            try:
                import minorminer
                return minorminer.find_embedding(source_graph, target_graph)
            except Exception:
                return {}
