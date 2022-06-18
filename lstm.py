import torch
import time
import args
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import VGRNN
from torch.autograd import Variable
from utils import ODData, get_mse_scores, get_balance_acc_scores
from input_data import load_data
from preprocessing import mask_edges_det, mask_edges_prd, mask_edges_prd_new, transpose_list
import logging
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, mean_squared_error

class Lstm(nn.Module):
    def __init__(self, hid_fea, num_nodes):
        super(Lstm, self).__init__()
        """
            in_fea: cnn输入、lstm输入的维度
            hid_fea: lstm输出的维度
        """

        # self.fc1 = nn.Conv2d(in_channels=num_nodes*num_fea, out_channels=hid_fea, kernel_size=1)
        self.lstm = nn.LSTM(input_size=num_nodes * num_nodes, hidden_size=hid_fea, batch_first=True)
        self.fc2 = nn.Conv2d(in_channels=hid_fea, out_channels=num_nodes * num_nodes, kernel_size=1)


    def forward(self, edge_fea):
        """
            edge_fea: B*T*N*N
        """
        # breakpoint()
        x = []
        for t in range(len(edge_fea)):
            x.append(edge_fea[t].A)
        edge_fea = torch.tensor(x).to(device).float()
        edge_fea = edge_fea.unsqueeze(dim=0)
        B,T,N,_ = edge_fea.shape
        edge_fea = edge_fea.reshape(B, T, -1)
        # edge_fea = edge_fea.permute(0, 3, 2, 1)
        # edge_fea = self.fc1(edge_fea)
        # edge_fea = F.leaky_relu(edge_fea)
        # edge_fea = edge_fea.permute(0, 3, 2, 1).squeeze(dim=2)
        # breakpoint()
        edge_fea = self.lstm(edge_fea)[1][0]
        edge_fea = edge_fea.permute(0, 2, 1).unsqueeze(dim=-1)
        edge_fea = self.fc2(edge_fea)
        edge_fea = torch.sigmoid(edge_fea)
        edge_fea = edge_fea.permute(0, 3, 2, 1)
        edge_fea = edge_fea.reshape(B, 1, N, -1)
        # breakpoint()
        return edge_fea


def get_roc_scores(edges_pos, edges_neg, adj_orig_dense_list, embs):
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
        emb = emb.squeeze()
        adj_orig_t = adj_orig_dense_list[i]
        preds = []
        pos = []
        for e in edges_pos[i]:
            preds.append(emb[e[0], e[1]]) # e[0],e[1]分别代表边的顶点
            # pos.append(adj_orig_t[e[0], e[1]]) # 记录真边

        preds_neg = []
        neg = []
        for e in edges_neg[i]:
            preds_neg.append(emb[e[0], e[1]])
            # neg.append(adj_orig_t[e[0], e[1]]) # 记录假边

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        auc_scores.append(roc_auc_score(labels_all, preds_all))
        ap_scores.append(average_precision_score(labels_all, preds_all))

    return auc_scores, ap_scores




device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# Load time series adjacent in Sparse and Dense List
adj_time_list, adj_orig_dense_list = load_data(args.dataset)
# Get training edges set and false set uesd to score
train_edges_l = mask_edges_det(adj_time_list)[1]    # (t, #Training Edges, 2),注意，这是边探测任务而非预测，每个时刻只有85%的边（无向边）参与训练，这似乎对于预测任务有一些影响
pos_edges_l, false_edges_l = mask_edges_prd(adj_time_list)  # (t, #Edges, 2)
# pos_edges_l_n, false_edges_l_n = mask_edges_prd_new(adj_time_list, adj_orig_dense_list) # (t, #New Edges, 2)
edge_idx_list = transpose_list(train_edges_l)   # (t, 2, #Training Edges)

seq_len = len(train_edges_l)
print(f"seq len: {seq_len}")
num_nodes = adj_orig_dense_list[seq_len-1].shape[0]
x_dim = num_nodes # 没有原始特征的话，使用编号作为节点特征

x_in_list = []
for i in range(0, seq_len):
    x_temp = torch.tensor(np.eye(num_nodes).astype(np.float32), device=device)
    x_in_list.append(torch.tensor(x_temp))
x_in = Variable(torch.stack(x_in_list))

model = Lstm(1024, num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
lossfunction = nn.BCELoss()


# seq_start = args.seq_start
seq_margin = 6 # 序列长度
train_num = int(seq_len * 0.1)



for epoch in range(args.epochs):

    #############################
    ### train ###################
    #############################
    t = time.time()
    for i in range(train_num - seq_margin + 1):
        model.train()
        optimizer.zero_grad()
        output = model(adj_orig_dense_list[i:i+seq_margin]) # x_in:(6,663,663), edge_idx_list:(6,2,$edges), adj_orig_dense_list:(6,663,663), type: Tensor
        output = output.flatten()
        label = torch.from_numpy(adj_orig_dense_list[i+seq_margin].A).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
        label = label.flatten()
        nll_loss = lossfunction(output, label)
        assert not torch.isnan(nll_loss).any() and not torch.isinf(nll_loss).any()
        nll_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if (i+1)%10==0:
            print('epoch: {k}, nll_loss = {nll:.3f}'.format(
                k=epoch,
                nll=nll_loss.mean().item(),
                end='\n'))
    print(f"training time: {time.time()-t}")


    #############################
    ### test in every epoch #####
    #############################
    with torch.no_grad():
        embeddings = []
        model.eval()
        for i in range(train_num, seq_len - seq_margin + 1):
            output= model( adj_orig_dense_list[i:i+seq_margin])
            embeddings.append(output.detach().cpu())
        auc_scores_prd, ap_scores_prd = get_roc_scores(
            pos_edges_l[train_num+seq_margin-1:],
            false_edges_l[train_num+seq_margin-1:],
            adj_orig_dense_list[train_num+seq_margin-1:],
            embeddings)


        auc = np.mean(np.array(auc_scores_prd))
        ap = np.mean(np.array(ap_scores_prd))
        print(f'AUC = {auc:.4f}, AP = {ap:.4f}')

