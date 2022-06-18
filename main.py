import torch
import time
import args
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from models import VGRNN
from torch.autograd import Variable
from utils import ODData, get_mse_scores, get_roc_scores, get_balance_acc_scores
from input_data import load_data
from preprocessing import mask_edges_det, mask_edges_prd, mask_edges_prd_new, transpose_list
import logging

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
# x_in = np.load('data/metro/source_data.npy')
# x_in = torch.from_numpy(x_in).float()
# x_in = x_in.permute(2, 0, 1)
# x_in = x_in.to(device)
model = VGRNN(x_dim, args.h_dim, args.z_dim, args.n_layers, args.dropout, args.alpha, device, bias=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# seq_start = args.seq_start
seq_margin = 6 # 序列长度
train_num = int(seq_len * 0.1)
# tst_after = args.tst_after
# train_set = ODData(folder_path=args.folder_path, seq_len=args.seq_len, start_idx=args.start_idx)
# test_set = ODData(folder_path=args.folder_path, seq_len=args.seq_len, start_idx=args.start_idx, train=False)
# train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
for epoch in range(args.epochs):

    #############################
    ### train ###################
    #############################
    t = time.time()
    for i in range(train_num - seq_margin + 1):
        model.train()
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _, _ = model(
            x_in[i:i+seq_margin],
            edge_idx_list[i:i+seq_margin],
            adj_orig_dense_list[i:i+seq_margin]) # x_in:(6,663,663), edge_idx_list:(6,2,$edges), adj_orig_dense_list:(6,663,663), type: Tensor
        loss = kld_loss + nll_loss
        assert not torch.isnan(loss).any() and not torch.isinf(loss).any()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if (i+1)%100==0:
            print('epoch: {k}, kld_loss = {kld:.3f}, nll_loss = {nll:.3f}, loss = {loss:.4f}'.format(
                k=epoch, kld=kld_loss.mean().item(),
                nll=nll_loss.mean().item(),
                loss=loss.mean().item()),
                end='\n')
    print(f"training time: {time.time()-t}")
    #############################
    ### test in every epoch #####
    #############################
    with torch.no_grad():
        embeddings = []
        model.eval()
        # if epoch > tst_after:
        for i in range(train_num, seq_len - seq_margin + 1):
            _, _, _, all_prior_mu, _ = model(
                x_in[i:i+seq_margin],
                edge_idx_list[i:i+seq_margin],
                adj_orig_dense_list[i:i+seq_margin])
            embeddings.append(all_prior_mu[-1])
        auc_scores_prd, ap_scores_prd = get_roc_scores(
            pos_edges_l[train_num+seq_margin-1:],
            false_edges_l[train_num+seq_margin-1:],
            adj_orig_dense_list[train_num+seq_margin-1:],
            embeddings,
            model)

        # balance_acc_scores = get_balance_acc_scores(
        #     adj_orig_dense_list[seq_end:seq_end+1],
        #     all_prior_mu,
        #     model
        # )
        # auc_scores_prd_new, ap_scores_prd_new = get_roc_scores(
        #     pos_edges_l_n[seq_end:],
        #     false_edges_l_n[seq_end:],
        #     adj_orig_dense_list[seq_end:], all_enc_mu)

        # balance_acc_score = np.mean(balance_acc_scores)
        auc = np.mean(np.array(auc_scores_prd))
        ap = np.mean(np.array(ap_scores_prd))
        # nauc = np.mean(np.array(auc_scores_prd_new))
        # nap = np.mean(np.array(ap_scores_prd_new))
        print(f'AUC = {auc:.4f}, AP = {ap:.4f}')



##############################
## calulate average metrics ##
##############################
# with torch.no_grad():
#     model.eval()
#     for i in range(2):
#         _, mse_loss, all_enc_mu, all_prior_mu, _ = model(
#             x_in[seq_end:seq_end+1],
#             edge_idx_list[seq_end:seq_end+1],
#             adj_orig_dense_list[seq_end:seq_end+1], hidden_st) # 为什么模型可以看到未来的输入？,猜测这里作者是“连续预测”，即{1~6}->{7},{1~7}->{8},{1~8}->{9}，所以能看到“未来的输入”

#         # auc_scores_prd, ap_scores_prd = get_roc_scores(
#         #     pos_edges_l[seq_end:],
#         #     false_edges_l[seq_end:],
#         #     adj_orig_dense_list[seq_end:], all_prior_mu) # 使用z的条件先验分布作为节点的embedding
#         # auc_scores_prd_new, ap_scores_prd_new = get_roc_scores(
#         #     pos_edges_l_n[seq_end:],
#         #     false_edges_l_n[seq_end:],
#         #     adj_orig_dense_list[seq_end:], all_prior_mu)

#         mse = np.mean(mse_loss.detach().cpu().numpy())
#         # ap = np.mean(np.array(ap_scores_prd))
#         # nauc = np.mean(np.array(auc_scores_prd_new))
#         # nap = np.mean(np.array(ap_scores_prd_new))
#         # print('AUC = {auc:.4f}, AP = {ap:.4f}'.format(auc=auc, ap=ap))
#         print('MSE = {mse:.4f}'.format(mse=mse))

# GVRNN with GRU
# epoch: 499, kld_loss = 0.075, mse_loss = 3.107, loss = 3.1820, AUC = 0.8771, AP = 0.8576, New AUC = 0.8630, New AP = 0.9100

# VGRNN with LSTM
# epoch: 499, kld_loss = 0.076, mse_loss = 3.094, loss = 3.1698, AUC = 0.8767, AP = 0.8559, New AUC = 0.8558, New AP = 0.9072