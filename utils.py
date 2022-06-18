import math
import numpy as np
import torch.nn as nn
import os
import scipy.sparse as sp
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.metrics import balanced_accuracy_score

# utility functions
def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))
    return out

# all_prior_mu
def get_roc_scores(edges_pos, edges_neg, adj_orig_dense_list, embs, model):
    """
    Input:
        edge_pos: positive edges
        edge_neg: negative edges
        adj_orig_dense_list: adj matrixs
        embs: node embeddings
    Output:
        auc_scores:
        ap_scores:
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    auc_scores = []
    ap_scores = []

    for i in range(len(edges_pos)): # i 代表时刻
        # Predict on test set of edges
        emb = embs[i].detach()
        adj_rec = model.decoder(emb).cpu().numpy() # 计算节点对的内积，以便接下来的sigmoid激活
        adj_orig_t = adj_orig_dense_list[i]
        preds = []
        pos = []
        for e in edges_pos[i]:
            preds.append(adj_rec[e[0], e[1]]) # e[0],e[1]分别代表边的顶点
            # pos.append(adj_orig_t[e[0], e[1]]) # 记录真边

        preds_neg = []
        neg = []
        for e in edges_neg[i]:
            preds_neg.append(adj_rec[e[0], e[1]])
            # neg.append(adj_orig_t[e[0], e[1]]) # 记录假边

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        auc_scores.append(roc_auc_score(labels_all, preds_all))
        ap_scores.append(average_precision_score(labels_all, preds_all))

    return auc_scores, ap_scores

def get_mse_scores(adj_orig_dense_list, embs, transfer_matrix):
    mse_scores = []
    for i in range(len(adj_orig_dense_list)): # i 代表时刻
        # Predict on test set of edges
        emb = embs[i].detach().cpu().numpy()
        adj_rec = emb@transfer_matrix@emb.t()
        adj_orig_t = adj_orig_dense_list[i]
        mse_scores.append(mean_squared_error(adj_rec, adj_orig_t))
    return mse_scores

def get_balance_acc_scores(adj_orig_dense_list, embs, model):
    scores = []
    for i in range(len(adj_orig_dense_list)):
        adj_pred = model.decoder(embs[i])
        adj_pred = adj_pred.detach().cpu().numpy()
        adj_true = adj_orig_dense_list[i].copy().A
        # breakpoint()
        adj_pred = adj_pred.flatten()
        adj_true = adj_true.flatten()
        scores.append(balanced_accuracy_score(adj_true, adj_pred))
    return scores


class ODData(Dataset):
    def __init__(self, folder_path:str, seq_len:int, start_idx:int, train=True) -> None:
        super(ODData, self).__init__()
        """
            folder_path:数据文件所在路径
            seq_len:序列长度
            start_idx:采样的起点
        """
        suffix = "/train/" if train else "/test/"
        folder_path = folder_path + suffix
        assert os.path.exists(folder_path), "folder not found."
        self.oddata = []
        self.edge = []
        files = os.listdir(folder_path)
        for file in files:
            flow_data = np.load(folder_path+file) # (N,N,T)
            flow_data = np.transpose(flow_data, axes=(2,0,1)) # (T,N,N)
            T,_,_ = flow_data.shape
            flow_data = flow_data.astype(np.float32)
            edge_data = []
            for i in range(T):
                sparse_mx = sp.csr_matrix(flow_data[i])
                sparse_mx.eliminate_zeros()
                sparse_mx = sparse_mx.tocoo()
                coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
                edge_data.append(coords)
            for i in range(start_idx, T-seq_len+1):
                self.oddata.append(flow_data[i:i+seq_len]) # 增加一个样本
                self.edge.append(edge_data[i:i+seq_len]) # 增加一个样本
        self.len = len(self.oddata)
        print(f"data size: {self.len}")
    def __len__(self,):
        return self.len

    def __getitem__(self, index: int):
        return self.oddata[index], self.edge[index] # 一个样本包括t-1个历史数据，1个预测数据
