import torch
import time
import args
import numpy as np
import torch.nn as nn
from models import VGRNN
from torch.autograd import Variable
from utils import get_roc_scores
from input_data import load_data
from preprocessing import mask_edges_det, mask_edges_prd, mask_edges_prd_new, transpose_list

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# Load time series adjacent in Sparse and Dense List
adj_time_list, adj_orig_dense_list = load_data(args.dataset)
# Get training edges set and false set uesd to score
train_edges_l = mask_edges_det(adj_time_list)[1]    # (t, #Training Edges, 2)
pos_edges_l, false_edges_l = mask_edges_prd(adj_time_list)  # (t, #Edges, 2)
pos_edges_l_n, false_edges_l_n = \
    mask_edges_prd_new(adj_time_list, adj_orig_dense_list) # (t, #New Edges, 2)
edge_idx_list = transpose_list(train_edges_l)   # (t, 2, #Training Edges)

seq_len = len(train_edges_l)
num_nodes = adj_orig_dense_list[seq_len-1].shape[0]
x_dim = num_nodes # 没有原始特征的话，使用编号作为节点特征

x_in_list = []
for i in range(0, seq_len):
    x_temp = torch.tensor(np.eye(num_nodes).astype(np.float32), device=device)
    x_in_list.append(torch.tensor(x_temp))
x_in = Variable(torch.stack(x_in_list))

model = VGRNN(x_dim, args.h_dim, args.z_dim, args.n_layers, device, bias=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

seq_start = args.seq_start
seq_end = seq_len - 3
tst_after = args.tst_after

for epoch in range(args.epochs):

    #############################
    ### train ###################
    #############################
    t = time.time()
    optimizer.zero_grad()
    kld_loss, nll_loss, _, _, hidden_st = model(
        x_in[seq_start:seq_end],
        edge_idx_list[seq_start:seq_end],
        adj_orig_dense_list[seq_start:seq_end])
    loss = kld_loss + nll_loss
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    print('epoch: {k}, kld_loss = {kld:.3f}, nll_loss = {nll:.3f}, loss = {loss:.4f}'.format(
        k=epoch, kld=kld_loss.mean().item(),
        nll=nll_loss.mean().item(),
        loss=loss.mean().item()),
        end=', ' if epoch!=0 else '\n')

    #############################
    ### test in every epoch #####
    #############################
    if epoch > tst_after:
        _, _, all_enc_mu, all_prior_mu, _ = model(
            x_in[seq_end:],
            edge_idx_list[seq_end:],
            adj_orig_dense_list[seq_end:], hidden_st)

        auc_scores_prd, ap_scores_prd = get_roc_scores(
            pos_edges_l[seq_end:],
            false_edges_l[seq_end:],
            adj_orig_dense_list[seq_end:], all_enc_mu)

        auc_scores_prd_new, ap_scores_prd_new = get_roc_scores(
            pos_edges_l_n[seq_end:],
            false_edges_l_n[seq_end:],
            adj_orig_dense_list[seq_end:], all_enc_mu)

        auc = np.mean(np.array(auc_scores_prd))
        ap = np.mean(np.array(ap_scores_prd))
        nauc = np.mean(np.array(auc_scores_prd_new))
        nap = np.mean(np.array(ap_scores_prd_new))
        print('AUC = {auc:.4f}, AP = {ap:.4f}, New AUC = {nauc:.4f}, New AP = {nap:.4f}'.format(
            auc=auc, ap=ap, nauc=nauc, nap=nap))



##############################
## calulate average metrics ##
##############################
for i in range(100):
    _, _, all_enc_mu, all_prior_mu, _ = model(
        x_in[seq_end:],
        edge_idx_list[seq_end:],
        adj_orig_dense_list[seq_end:], hidden_st)

    auc_scores_prd, ap_scores_prd = get_roc_scores(
        pos_edges_l[seq_end:],
        false_edges_l[seq_end:],
        adj_orig_dense_list[seq_end:], all_prior_mu)

    auc_scores_prd_new, ap_scores_prd_new = get_roc_scores(
        pos_edges_l_n[seq_end:],
        false_edges_l_n[seq_end:],
        adj_orig_dense_list[seq_end:], all_prior_mu)

    auc = np.mean(np.array(auc_scores_prd))
    ap = np.mean(np.array(ap_scores_prd))
    nauc = np.mean(np.array(auc_scores_prd_new))
    nap = np.mean(np.array(ap_scores_prd_new))
    print('AUC = {auc:.4f}, AP = {ap:.4f}, New AUC = {nauc:.4f}, New AP = {nap:.4f}'.format(
        auc=auc, ap=ap, nauc=nauc, nap=nap))

# GVRNN with GRU
# epoch: 499, kld_loss = 0.075, nll_loss = 3.107, loss = 3.1820, AUC = 0.8771, AP = 0.8576, New AUC = 0.8630, New AP = 0.9100

# VGRNN with LSTM
# epoch: 499, kld_loss = 0.076, nll_loss = 3.094, loss = 3.1698, AUC = 0.8767, AP = 0.8559, New AUC = 0.8558, New AP = 0.9072