import numpy as np
import torch
import scipy.sparse as sp
from input_data import load_data
import time
from torch_geometric.utils import add_self_loops

seed = 146
np.random.seed(seed)

def sparse_to_tuple(sparse_mx):
# Input: a sparse matrix
# Output: Coords, values and shape
# coords.shape = ($Edges, 2)
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose() # 得到每一个数据对应的行列索引
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
# return symmetrically normalized adjacency matrix
    edge_feature = adj.shape[0]
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(edge_feature)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def ismember(a, b, tol=5):
# If a is/has a member of b
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)

def getEdges(adj):
# Input: Sparse adjacent
# Output: Single-direction graph edges with and without self-loop: List
#         edges.shpae = ($Edges, 2)
    adj = adj - sp.dia_matrix(
        (adj.diagonal()[np.newaxis, :], [0]),
        shape=adj.shape) # 消去自环
    adj.eliminate_zeros() # 放弃0值对应的索引信息
    assert np.diag(adj.todense()).sum() == 0
    adj_triu = sp.triu(adj) #取出稀疏矩阵的上三角部分的非零元素
    adj_tuple = sparse_to_tuple(adj_triu) # 这个函数的作用就是 返回一个稀疏矩阵的非0值坐标、非0值和整个矩阵的shape
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    return edges, edges_all # 1.二者都不包含自环，2.edges包含adj上三角包含的边【适用于无向图】，3.edges_all包含所有的边

def makeFalseEdges(nums, size, edgeSet):
    """
    input:
        nums: the number of false edges
        size: the number of nodes
        edgeSet: the set of edges,[all edges]
    output:
        false edges,
    """
    start_time = time.time()
    edges_false = []
    edgeSet = torch.from_numpy(edgeSet).t()
    edgeSet, _ = add_self_loops(edgeSet, num_nodes=size)
    edgeSet = edgeSet.t().numpy()
    all_edges = np.array(range(size**2))
    # while len(edges_false) < nums:
    #     h = np.random.randint(0, size)
    #     t = np.random.randint(0, size)
    #     if h == t: continue # 确保没有自环
    #     e = [[h, t], [t, h]]
    #     if edges_false and ismember(e, np.array(edges_false)): continue # 防止重复添加
    #     if ismember(e, edgeSet): continue # 确保是错误的边
    #     edges_false.append([h, t])
    # assert ~ismember(edges_false, edgeSet)
    edges = edgeSet[:,0]*size+edgeSet[:,1]
    false_edge_set = np.setdiff1d(all_edges, edges, assume_unique=True)
    false_edges = np.random.choice(false_edge_set, nums, replace=False)
    row_id = false_edges//size
    col_id = false_edges%size
    edges_false = np.vstack((row_id, col_id)).T
    # print(time.time() - start_time)
    return edges_false

def mask_edges_det(adjs_list):
# Input: Time series Sparse adjacent
# Output: Adjacents used to train: Sparse
#         Train(70%)/validate(20%)/test(10%) edges: List
    adj_train_l, train_edges_l = [], []
    val_edges_l, val_edges_false_l = [], []
    test_edges_l, test_edges_false_l = [], []
    edges_list = []
    edge_feature = adjs_list[0].shape[0] #事实上是节点的数量？
    for i in range(0, len(adjs_list)):
        adj = adjs_list[i]
        edges, edges_all = getEdges(adj) # 注意edges是代表无向图的边
        # num_test = int(np.floor(edges.shape[0] / 10.))
        # num_val = int(np.floor(edges.shape[0] / 20.))

        # all_edge_idx = range(edges.shape[0])
        # np.random.shuffle([all_edge_idx])
        # val_edge_idx = all_edge_idx[:num_val]
        # test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        # test_edges = edges[test_edge_idx]
        # val_edges = edges[val_edge_idx]
        # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0) # 在每个时刻都选择85%的边？对于预测任务不应该是全部的边吗？
        train_edges = edges_all
        # edges_list.append(edges)

        # test_edges_false = makeFalseEdges(len(test_edges), edge_feature, edges_all)
        # val_edges_false = makeFalseEdges(len(val_edges), edge_feature, edges_all)

        # 确保集合之间无交集
        # assert ~ismember(val_edges, train_edges)
        # assert ~ismember(test_edges, train_edges)
        # assert ~ismember(val_edges, test_edges)

        # data = np.ones(train_edges.shape[0])

        # adj_train = sp.csr_matrix(
        #     (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        # adj_train = adj_train + adj_train.T # 为什么要这么做？上三角+下三角得到一个对称的邻接矩阵？

        # adj_train_l.append(adj_train)
        train_edges_l.append(train_edges)
        # val_edges_l.append(val_edges)
        # val_edges_false_l.append(val_edges_false)
        # test_edges_l.append(test_edges)
        # test_edges_false_l.append(test_edges_false)

    return adj_train_l, train_edges_l, val_edges_l, val_edges_false_l, test_edges_l, test_edges_false_l # edges都是边的二元组表示，adj是矩阵形式


def mask_edges_prd(adjs_list):
# Input: A list of Sparse adjacent
# Output: Ture edges and false edges
#         pos_edges_l.shape = (t, #Edges, 2)
    pos_edges_l, false_edges_l = [], []
    edge_feature = adjs_list[0].shape[0]
    for i in range(0, len(adjs_list)):
        adj = adjs_list[i]
        edges, edges_all = getEdges(adj)
        num_false = int(edges_all.shape[0])
        pos_edges_l.append(edges_all)
        edges_false = makeFalseEdges(num_false, edge_feature, edges_all)
        false_edges_l.append(edges_false)
    return pos_edges_l, false_edges_l


def mask_edges_prd_new(adjs_list, adj_orig_dense_list):
# Input: A list of Sparse adjacent
# Output: Ture new edges and false new edges
#         pos_edges_l.shape = (t, #New Edges, 2)
    pos_edges_l, false_edges_l = [], []
    edge_feature = adjs_list[0].shape[0]
    edges, edges_all = getEdges(adjs_list[0])
    num_false = int(edges.shape[0])
    pos_edges_l.append(edges)
    edges_false = makeFalseEdges(num_false, edge_feature, edges_all)
    false_edges_l.append(np.asarray(edges_false))

    for i in range(1, len(adjs_list)):
        edges_pos = np.transpose(np.asarray(
            np.where((adj_orig_dense_list[i] - adj_orig_dense_list[i-1]) > 0)))
        num_false = int(edges_pos.shape[0])
        adj = adjs_list[i]
        edges, edges_all = getEdges(adj)
        edges_false = makeFalseEdges(len(edges_false), edge_feature, edges_all)
        false_edges_l.append(np.asarray(edges_false))
        pos_edges_l.append(edges_pos)

    return pos_edges_l, false_edges_l

def transpose_list(train_edges_l):
# Input: Edges list for train: List
# Output: Transpose of input
    edge_idx_list = []
    for i in range(len(train_edges_l)):
        edge_idx_list.append(torch.tensor(
        np.transpose(train_edges_l[i]), dtype=torch.long))
    return edge_idx_list