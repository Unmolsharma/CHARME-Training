"""Environment for minor embedding.

Source: authors' exact code from charme-rl repo.
Fixes applied:
  1. orderlist_path passed as constructor arg instead of hardcoded '../orderlist.pkl'
  2. atom binary path passed as constructor arg instead of hardcoded './atom/atom_system'
  3. atom_log path derived from atom binary location instead of hardcoded 'atom/atom_log/PPO.txt'
  4. topo_column renamed to topo_col in constructor to match paper terminology
     (internal usage preserved as topo_column for compatibility)
All logic — reset(), step(), update_hw(), call_atom() — is authors' exact code.
"""

from .utils import *
import hashlib
import torch
import os


class MinorEmbeddingEnv(object):
    def __init__(self, topo_row, topo_col, bipart_cell, goal_dim,
                 num_nodes, n_state, seed, degree, training_size,
                 orderlist_path, atom_binary_path, mode=0):

        self.topo_row = topo_row
        self.topo_column = topo_col   # authors use topo_column internally
        self.curr_row = None
        self.curr_column = None
        self.bipart_cell = bipart_cell
        self.goal_dim = goal_dim
        self.num_nodes = num_nodes
        self.n_state = n_state
        self.seed = seed
        self.degree = degree
        self.training_size = training_size
        self.training_index = -1

        # FIX: paths as constructor args, not hardcoded
        self.orderlist_path = str(orderlist_path)
        self.atom_binary = str(atom_binary_path)
        # atom log file sits next to the binary in atom_log/PPO.txt
        atom_dir = os.path.dirname(self.atom_binary)
        self.atom_file = os.path.join(atom_dir, 'atom_log', 'PPO.txt')
        os.makedirs(os.path.dirname(self.atom_file), exist_ok=True)

        self.original_chimera_graph = generate_Chimera(
            topo_row=self.topo_row,
            topo_column=self.topo_column,
            bipart_cell=self.bipart_cell
        )
        for node in self.original_chimera_graph.nodes:
            self.original_chimera_graph.nodes[node]['mapping'] = (
                list(self.original_chimera_graph.nodes).index(node)
            )

        self.training_graph_list = []
        self.minorminer_list = []
        self.fix_state_list = []

        if mode == 0:
            hw_edge_index = get_hw_edge_index(self.original_chimera_graph)
            for i in range(self.training_size):
                graph = nx.barabasi_albert_graph(self.num_nodes, self.degree)
                self.training_graph_list.append(graph)
                minorminer = convert_graph_to_embeddingMinorminer(
                    graph, self.original_chimera_graph
                )
                self.minorminer_list.append(minorminer)
                logical_edge_index, logical_attr = analysing_logical(graph)
                self.fix_state_list.append({
                    'logical_edge_index': logical_edge_index,
                    'logical_attr': logical_attr,
                    'hw_edge_index': hw_edge_index.clone()
                })

        self.done = None
        self.num_steps = None
        self.state = None
        self.target = None
        self.agent_position = 0
        self.barabasi_albert_graph = init_logical_graph(self.num_nodes, self.degree)
        self.original_barabasi_albert_graph = self.barabasi_albert_graph.copy()

        self.logical_embedding = None
        self.mask = None
        self.mask_connected = None
        self.chimera_graph = None
        self.chimera_embedding = None

        self.state_dim = 0
        self.max_ep_steps = 0
        self.minorminer = None
        self.minorminer_solution = None
        self.dict_graph = {}
        self.encounter_graph_num = 0
        self.order_gamma = 1

    def load_graph(self, graph_list, minorminer_list):
        # Authors' exact — unchanged
        self.training_graph_list = []
        self.minorminer_list = []
        self.fix_state_list = []
        hw_edge_index = get_hw_edge_index(self.original_chimera_graph)
        for i in range(0, self.training_size):
            graph = graph_list[i]
            self.training_graph_list.append(graph)
            self.minorminer_list.append(minorminer_list[i])
            logical_edge_index, logical_attr = analysing_logical(graph)
            self.fix_state_list.append({
                'logical_edge_index': logical_edge_index,
                'logical_attr': logical_attr,
                'hw_edge_index': hw_edge_index.clone()
            })

    def reset(self, mode):
        # Authors' exact reset() with one fix: orderlist_path from self
        self.done = False
        
        self.mask = [False] * self.num_nodes
        self.mask_connected = [True] * self.num_nodes
        self.num_steps = 0
        self.agent_position = 0
        self.max_ep_steps = self.num_nodes
        self.barabasi_albert_graph = self.original_barabasi_albert_graph.copy()

        import pickle
        # FIX: use self.orderlist_path instead of hardcoded '../orderlist.pkl'
        with open(self.orderlist_path, 'rb') as f:
            order_list = pickle.load(f)

        self.state = {}
        if mode == 'change_logical_graph':
            self.barabasi_albert_graph = init_logical_graph(self.num_nodes, self.degree)
            self.original_barabasi_albert_graph = self.barabasi_albert_graph.copy()

        if mode == 'batch_change_logical_graph':
            self.training_index = self.training_index + 1
            index = (self.training_index) % self.training_size
            self.orders = order_list[index].copy()
            self.barabasi_albert_graph = self.training_graph_list[index].copy()
            self.original_barabasi_albert_graph = self.barabasi_albert_graph.copy()
            self.minorminer = self.minorminer_list[index].copy()
            self.state['logical_edge_index'] = self.fix_state_list[index]['logical_edge_index'].clone()
            self.state['logical_attr'] = self.fix_state_list[index]['logical_attr'].clone()
            self.state['hw_edge_index'] = self.fix_state_list[index]['hw_edge_index'].clone()

        if mode == 'batch_change_logical_graph_with_crash':
            index = (self.training_index) % self.training_size
            self.orders = order_list[index].copy()
            self.barabasi_albert_graph = self.training_graph_list[index].copy()
            self.original_barabasi_albert_graph = self.barabasi_albert_graph.copy()
            self.minorminer = self.minorminer_list[index].copy()
            self.state['logical_edge_index'] = self.fix_state_list[index]['logical_edge_index'].clone()
            self.state['logical_attr'] = self.fix_state_list[index]['logical_attr'].clone()
            self.state['hw_edge_index'] = self.fix_state_list[index]['hw_edge_index'].clone()

        self.chimera_graph = self.original_chimera_graph.copy()
        for node in self.chimera_graph.nodes:
            self.chimera_graph.nodes[node]['embedding'] = -1

        # call atom mode 0 — initialization
        embedding, rr, cc, _ = self.call_atom(
            self.barabasi_albert_graph, self.topo_row, self.topo_column, self.seed, 0
        )
        self.update_hw([], embedding)
        self.embedding = embedding

        self.state['hw_attr'] = get_hw_attr_synthetic(self.chimera_graph)
        self.curr_row = rr
        self.curr_column = cc

        for emb in embedding:
            self.mask[emb[3]] = True

        for node in self.barabasi_albert_graph.nodes:
            if self.mask[node]:
                for nei_node in self.barabasi_albert_graph.neighbors(node):
                    self.mask_connected[nei_node] = False

        self.state['emb_matrix'] = convert_embedding_to_tensor(
            embedding, self.chimera_graph, self.original_barabasi_albert_graph
        ).to_sparse()

        return self.state

    def step(self, action):
        # Authors' exact step() — unchanged
        self.agent_position = self.agent_position + 1
        self.mask[action] = True
        for nei_node in self.barabasi_albert_graph.neighbors(action):
                    self.mask_connected[nei_node] = False

        curr_emb = self.embedding.copy()

        new_emb, rr, cc, old_node = self.call_atom(
            self.barabasi_albert_graph, self.curr_row, self.curr_column,
            self.seed, 1, action, curr_emb
        )

        for node in old_node:
            if self.barabasi_albert_graph.has_edge(node, action):
                self.barabasi_albert_graph.remove_edge(node, action)

        self.curr_row = rr
        self.curr_column = cc
        self.embedding = new_emb

        try:
            self.update_hw(curr_emb, new_emb)
        except:
            return 0, 0, -1

        self.state['emb_matrix'] = update_embedding_matrix(
            self.state['emb_matrix'], self.chimera_graph, curr_emb, new_emb
        )
        self.state['hw_attr'] = update_hw_attr_synthetic(
            self.state['hw_attr'], self.chimera_graph, curr_emb, new_emb
        )

        # Equation 9: reward = -qubit_gain - exploration_term
        reward = -(len(new_emb) - len(curr_emb))
        reward_distance = -self.order_gamma * (self.orders[0] - action) * (self.orders[0] - action)
        done = True
        del self.orders[0]

        for node in self.barabasi_albert_graph.nodes:
            if not self.mask[node]:
                done = False
                break

        self.num_steps += 1
        return reward, reward_distance, done

    def update_hw(self, curr_emb, new_emb):
        # Authors' exact — unchanged
        for emb in curr_emb:
            self.chimera_graph.nodes[(emb[0], emb[1], emb[2])]['embedding'] = -1
        for emb in new_emb:
            self.chimera_graph.nodes[(emb[0], emb[1], emb[2])]['embedding'] = emb[3]

    def call_atom(self, P, topo_row, topo_column, seed_limit, is_beginning,
                  curr_node=None, embedding=None):
        # Authors' exact call_atom() with path fix
        # FIX: use self.atom_binary and self.atom_file instead of hardcoded paths
        command = [self.atom_binary]
        atom_file = self.atom_file

        n = len(P.nodes)
        m = len(P.edges)
        command.append(str(n))
        command.append(str(m))

        for ed in P.edges:
            command.append(str(ed[0]))
            command.append(str(ed[1]))

        command.append(str(topo_row))
        command.append(str(topo_column))
        command.append(str(seed_limit))
        command.append(str(is_beginning))

        old_node = []
        if is_beginning == 1:
            command.append(str(curr_node))

            check = [False] * n
            for emb in embedding:
                check[emb[3]] = True

            old_node = []
            for node in P.neighbors(curr_node):
                if check[node]:
                    old_node.append(node)

            old_len = len(old_node)
            command.append(str(old_len))
            for node in old_node:
                command.append(str(node))

            emb_len = len(embedding)
            command.append(str(emb_len))
            for emb in embedding:
                command.append(str(emb[0]))
                command.append(str(emb[1]))
                command.append(str(emb[2]))
                command.append(str(emb[3]))

        # Authors' exact command construction
        command_s = ''
        for i in command:
            command_s = command_s + i + ' '
        command_s = command_s + atom_file
        os.system(command_s)

        new_embedding = []
        f = open(atom_file, "r")
        rr = -1
        cc = -1
        while True:
            line = f.readline()
            line_arr = line[:-1].split(' ')
            if len(line) == 0:
                break
            if len(line_arr) == 4:
                new_embedding.append((
                    int(line_arr[0]), int(line_arr[1]),
                    int(line_arr[2]), int(line_arr[3])
                ))
            else:
                rr = int(line_arr[0])
                cc = int(line_arr[1])
                break
        f.close()
        return new_embedding, rr, cc, old_node
