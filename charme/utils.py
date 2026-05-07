import networkx as nx
from karateclub import DeepWalk
from minorminer import find_embedding
import torch
import numpy as np


def analysing_hw(hw_graph):
    hw_edge_index = []
    for ed in hw_graph.edges:
        node_0 = hw_graph.nodes[ed[0]]['mapping']
        node_1 = hw_graph.nodes[ed[1]]['mapping']
        hw_edge_index.append([node_0, node_1])
        hw_edge_index.append([node_1, node_0])
    hw_edge_index = torch.tensor(np.array(hw_edge_index).T)
    #hw_edge_index = np.array(hw_edge_index).T

    hw_attr = []
    for node in hw_graph.nodes:
        hw_attr.append(hw_graph.nodes[node]['embedding'])
        
    hw_attr = torch.tensor(np.array([hw_attr]).T, dtype=torch.float)
    #hw_attr = np.array([hw_attr]).T
    return hw_edge_index, hw_attr

def convert_embedding_to_tensor(embedding, hw_graph, logical_graph):
    emb_matrix = torch.zeros([len(logical_graph.nodes),len(hw_graph.nodes)])
    #node_set = []
    for emb in embedding:
        #node_set.append(emb[3])
        emb_matrix[emb[3]][hw_graph.nodes[(emb[0],emb[1],emb[2])]['mapping']] = 1
    return emb_matrix

def update_embedding_matrix(emb_matrix, hw_graph, curr_emb, new_emb):
    emb_matrix = emb_matrix.to_dense()
    node_indices = [emb[3] for emb in curr_emb]
    mapping_indices = [hw_graph.nodes[(emb[0], emb[1], emb[2])]['mapping'] for emb in curr_emb]
    emb_matrix[node_indices,mapping_indices] = 0
    
    node_indices = [emb[3] for emb in new_emb]
    mapping_indices = [hw_graph.nodes[(emb[0], emb[1], emb[2])]['mapping'] for emb in new_emb]
    emb_matrix[node_indices,mapping_indices] = 1
    return emb_matrix.to_sparse()

def get_embedding_dict(new_emb, hw_graph, limit, device):
    keys = list(range(0,limit))
    values = [[] for i in range(0,limit)]
    emb_dict = dict(zip(keys,values ))
    for emb in new_emb:
        emb_dict[emb[3]].append(hw_graph.nodes[(emb[0], emb[1], emb[2])]['mapping'])
    emb_dict_ = {}
    for k in emb_dict.keys():
        emb_dict_[k] = torch.IntTensor(emb_dict[k]).detach().to(device)
    return emb_dict_
        

def get_hw_edge_index(hw_graph):
    hw_edge_index = []
    for ed in hw_graph.edges:
        node_0 = hw_graph.nodes[ed[0]]['mapping']
        node_1 = hw_graph.nodes[ed[1]]['mapping']
        hw_edge_index.append([node_0, node_1])
        hw_edge_index.append([node_1, node_0])
    hw_edge_index = torch.tensor(np.array(hw_edge_index).T)
    #hw_edge_index = np.array(hw_edge_index).T
    return hw_edge_index

def get_hw_attr(hw_graph):
    hw_attr = torch.zeros([len(hw_graph.nodes), 8])
    idx = 0
    for node in hw_graph.nodes:
        decimal_num = hw_graph.nodes[node]['embedding']
 
        # Initialize an empty list to hold the binary digits
        binary_list = []
        if decimal_num == -1:
            binary_list = [-1]*8
        else:
            countt = 0
            while countt < 8:
                binary_list.append(decimal_num % 2)
                decimal_num //= 2
                countt += 1
            binary_list.reverse()
        for j in range(0,8):
            hw_attr[idx][j] = binary_list[j]#hw_graph.nodes[node]['embedding']
        idx = idx + 1
    return hw_attr

def get_hw_attr_synthetic(hw_graph):
    hw_attr = torch.zeros([len(hw_graph.nodes), 1])
    idx = 0
    for node in hw_graph.nodes:
        hw_attr[idx][0] = hw_graph.nodes[node]['embedding']
        idx = idx + 1
    return hw_attr


def update_hw_attr_synthetic(hw_attr, hw_graph, curr_emb, new_emb):
    #node_indices = [emb[3] for emb in curr_emb]
    mapping_indices = [hw_graph.nodes[(emb[0], emb[1], emb[2])]['mapping'] for emb in curr_emb]
    hw_attr[mapping_indices] = torch.tensor([-1]).float()
    node_indices = [[emb[3]] for emb in new_emb]
    mapping_indices = [hw_graph.nodes[(emb[0], emb[1], emb[2])]['mapping'] for emb in new_emb]
    #print(node_indices)
    #print(mapping_indices)
    hw_attr[mapping_indices] = torch.tensor(node_indices).float()
    
    return hw_attr

def update_hw_attr(hw_attr, hw_graph, curr_emb, new_emb):
    #node_indices = [emb[3] for emb in curr_emb]
    mapping_indices = [hw_graph.nodes[(emb[0], emb[1], emb[2])]['mapping'] for emb in curr_emb]
    hw_attr[mapping_indices] = torch.tensor([-1]*8).float()
    
    #node_indices = [emb[3] for emb in new_emb]
    node_indices = []
    for emb in new_emb:
        decimal_num = emb[3]
        binary_list = []
        countt = 0
        while countt < 8:
            binary_list.append(decimal_num % 2)
            decimal_num //= 2
            countt += 1
        binary_list.reverse()
        node_indices.append(binary_list)
    mapping_indices = [hw_graph.nodes[(emb[0], emb[1], emb[2])]['mapping'] for emb in new_emb]
    hw_attr[mapping_indices] = torch.tensor(node_indices).float()
    
    return hw_attr

def analysing_logical(logical_graph):
    X = np.array(logical_graph.edges)
    Y = X.copy()
    tmp = Y[:,0].copy()
    Y[:,0] = Y[:,1].copy()
    Y[:,1] = tmp.copy()
    Z = np.concatenate((X.T, Y.T), axis = 1)
    logical_edge_index = torch.tensor(Z)
    #logical_edge_index = Z
    logical_attr = torch.tensor(np.array([[1]]*len(logical_graph.nodes)), dtype=torch.float)
    #logical_attr = np.array([[1]]*len(logical_graph.nodes))
    return logical_edge_index, logical_attr

# Khoi tao logical graph
def init_logical_graph(num_nodes, degree):
    P = nx.barabasi_albert_graph(num_nodes, degree)
    return P

def init_logical_graph_erdos(num_nodes, degree):
    P = nx.erdos_renyi_graph(num_nodes, degree)
    return P


# Khoi tao Chimera map
def generate_Chimera(topo_row=10, topo_column=10, bipart_cell=4):
    edge_list = []

    for i in range(topo_row):
        for j in range(topo_column):
            for k1 in range(bipart_cell):
                for k2 in range(bipart_cell, 2 * bipart_cell):
                    edge_list.append(((i, j, k1), (i, j, k2)))
                    edge_list.append(((i, j, k2), (i, j, k1)))

    for i in range(topo_row):
        for j in range(topo_column):
            for k in range(bipart_cell, 2 * bipart_cell):
                if (j != topo_column - 1):
                    edge_list.append(((i, j, k), (i, j + 1, k)))
                if (j != 0):
                    edge_list.append(((i, j, k), (i, j - 1, k)))

            for k in range(bipart_cell):
                if (i != topo_row - 1):
                    edge_list.append(((i, j, k), (i + 1, j, k)))
                if (i != 0):
                    edge_list.append(((i, j, k), (i - 1, j, k)))

    return nx.from_edgelist(edge_list)


# Chuyển đổi graph thành embedding su dung DEEPWALK
def convert_graph_to_latent_embedding(G, dim):
    # Chuyển đổi chỉ số của các đỉnh
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    # Áp dụng thuật toán DeepWalk để nhúng đồ thị
    model = DeepWalk(dimensions=dim)
    model.fit(G)
    embedding = model.get_embedding()
    return embedding.reshape((-1, 1)).squeeze()


# Minorminer Embedding
def convert_graph_to_embeddingMinorminer(lgc, hw):
    embedding = find_embedding(lgc, hw, random_seed=10, chainlength_patience = 0)

    converted_embedding = []
    for i in range(0, len(lgc.nodes)):
        for j in embedding[i]:
            converted_embedding.append((i, j))
    return converted_embedding


# Convert minorminer to tensor
def convert_minorminer_to_tensor(embedding, topo_row, topo_column, bipart_cell):
    matrix = torch.zeros(topo_row, topo_column, (bipart_cell * 2))

    for minor in embedding:
        x, y, z = minor[1]
        matrix[x][y][z] = minor[0] + 1
    return matrix.reshape(-1)


# Chuyen doi map
def convert_from_map_to_embedding(grid_map):
    embedding = []
    grid_map = grid_map - 1
    for i in range(len(grid_map)):
        for j in range(len(grid_map[i])):
            for k in range(len(grid_map[i][j])):
                if grid_map[i][j][k] != -1:
                    value = grid_map[i][j][k]
                    embedding.append((int(value.item()), (i, j, k)))
    return embedding


def test_chain_connection(H, P, embedding, max_bipartite, H_max_column, H_max_row):
    n_P = len(P.nodes)
    isin_embedding = [
        [[[False for i in range(max_bipartite)] for col in range(H_max_column)] for row in range(H_max_row)] for i in
        range(n_P)]
    nums = [0] * n_P
    for s in embedding:
        nums[s[0]] = nums[s[0]] + 1
        isin_embedding[s[0]][s[1][0]][s[1][1]][s[1][2]] = True
    dx = [[[False for i in range(max_bipartite)] for col in range(H_max_column)] for row in range(H_max_row)]

    total_check = True

    for s in embedding:
        if dx[s[1][0]][s[1][1]][s[1][2]] == True:
            continue
        else:
            dx[s[1][0]][s[1][1]][s[1][2]] = True
            Queue_c = []
            Queue_c.append(s[1])
            countt = 0
            color = s[0]
            while len(Queue_c) > 0:
                head = Queue_c[0]
                Queue_c.remove(head)
                countt = countt + 1

                for tail in H.neighbors(head):
                    if isin_embedding[color][tail[0]][tail[1]][tail[2]] and not dx[tail[0]][tail[1]][tail[2]]:
                        Queue_c.append(tail)
                        dx[tail[0]][tail[1]][tail[2]] = True

            if countt != nums[color]:
                # print(countt, nums[color])
                total_check = False
                # print("Wrong at ", s)
    return total_check


def test_global_connection(H, P, embedding, max_bipartite, H_max_column, H_max_row):
    n_P = len(P.nodes)
    isin_embedding = [
        [[[False for i in range(max_bipartite)] for col in range(H_max_column)] for row in range(H_max_row)] for i in
        range(n_P)]
    for s in embedding:
        isin_embedding[s[0]][s[1][0]][s[1][1]][s[1][2]] = True

    final_check = True
    edge_list = P.edges
    for ed in edge_list:
        check = False
        for s in embedding:
            if check:
                break
            color = s[0]
            if color == ed[0]:
                expected_color = ed[1]
                head = s[1]
                for tail in H.neighbors(head):
                    if isin_embedding[expected_color][tail[0]][tail[1]][tail[2]]:
                        check = True

        if not check:
            final_check = False
            # print("Wrong at ", ed)
            break
    return final_check


def convert_back_to_logical(H, embedding, max_bipartite, H_max_column, H_max_row):
    color_embedding = [[[-1 for i in range(max_bipartite)] for col in range(H_max_column)] for row in range(H_max_row)]
    for s in embedding:
        color_embedding[s[1][0]][s[1][1]][s[1][2]] = s[0]

    new_ed_list = []
    for ed in H.edges:
        node_0 = ed[0]
        node_1 = ed[1]
        if color_embedding[node_0[0]][node_0[1]][node_0[2]] != -1 and color_embedding[node_1[0]][node_1[1]][
            node_1[2]] != -1 and color_embedding[node_0[0]][node_0[1]][node_0[2]] != \
                color_embedding[node_1[0]][node_1[1]][node_1[2]]:
            new_ed_list.append(
                (color_embedding[node_0[0]][node_0[1]][node_0[2]], color_embedding[node_1[0]][node_1[1]][node_1[2]]))

    # new_ed_list = sorted(new_ed_list,key=lambda l:l[0], reverse=False)
    # print(new_ed_list)
    return nx.from_edgelist(new_ed_list)


# Kiểm tra một node có liền kề không
def is_adjacent_values(matrix, value):
    """
    Hàm kiểm tra xem tất cả các số cần kiểm tra có liền kề nhau trong ma trận 3 chiều hay không, sử dụng thuật toán DFS.
    """
    # Kích thước tensor
    m, n, p = matrix.shape

    # Lưu trữ các vị trí các giá trị đã được duyệt qua
    visited = set()

    # Hàm DFS
    def dfs(i, j, k):
        if (i, j, k) not in visited:
            visited.add((i, j, k))
            if matrix[i, j, k] == value:
                # Kiểm tra các giá trị liền kề
                if i - 1 >= 0:
                    dfs(i - 1, j, k)
                if i + 1 < m:
                    dfs(i + 1, j, k)
                if j - 1 >= 0:
                    dfs(i, j - 1, k)
                if j + 1 < n:
                    dfs(i, j + 1, k)
                if k - 1 >= 0:
                    dfs(i, j, k - 1)
                if k + 1 < p:
                    dfs(i, j, k + 1)

    # Duyệt qua các phần tử của tensor để tìm giá trị đầu tiên
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if matrix[i, j, k] == value:
                    dfs(i, j, k)
                    # Nếu đã duyệt qua tất cả các giá trị liền kề thì trả về True
                    if all(matrix[i, j, k] != value or (i, j, k) in visited for i in range(m) for j in range(n) for k in
                           range(p)):
                        return True
                    # Ngược lại, tiếp tục duyệt các phần tử khác của tensor
                    visited = set()

    # Nếu không tìm thấy giá trị nào thì trả về False
    return False


# Kiểm tra array có tạo thành graph liên thông không
def check_graph(array, n_nodes):
    for node in range(1, n_nodes + 1):
        if is_adjacent_values(array, node) == False:
            return False

    return True


def normalize_state(state, num_node, topo_row, topo_column, bipart_cell):
    new_state = state.to(torch.float)
    for i, item in enumerate(state[:-3]):
        new_state[i] = float((state[i].item() / num_node) * 6 - 3)
    new_state[-1] = float((state[-1].item() / (topo_row - 1)) * 6 - 3)
    new_state[-2] = float((state[-2].item() / (topo_column - 1)) * 6 - 3)
    new_state[-3] = float((state[-3].item() / (bipart_cell * 2 - 1)) * 6 - 3)

    return new_state


def is_subset(list1, list2):
    if len(list1) > len(list2):
        smaller_list = list2
        larger_list = list1
    else:
        smaller_list = list1
        larger_list = list2
    for item in smaller_list:
        if item not in larger_list and (item[1], item[0]) not in larger_list:
            return False
    return True
