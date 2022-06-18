from statistics import variance
from turtle import forward
# from matplotlib.pyplot import scatter
# from sympy import false
import torch
# from zmq import device
from MessagePassing import MessagePassing
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
# from torch_geometric.nn import GATConv
from torch_scatter import scatter_add, scatter
from utils import *


class GCNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, device, act=F.relu, bias=False):
        super(GCNConv, self).__init__(device)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.device = device
        self.weight = Parameter(torch.Tensor(in_dim, out_dim).to(device))
        if bias:
            self.bias = Parameter(torch.Tensor(out_dim).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_idx, edge_weight=None):
        # Initialize edge weights
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_idx.size(1), ), dtype=x.dtype, device=self.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_idx.size(1)
        # Add self-loop edge weights to edge_weight
        edge_idx, _ = add_self_loops(edge_idx, num_nodes=x.size(0))
        loop_weight = torch.full(
            (x.size(0), ), 1, dtype=x.dtype, device=self.device)
        edge_weight = torch.cat(
            [edge_weight, loop_weight], dim=0).to(self.device)
        # symmetrically normalized edge weights
        row, col = edge_idx     # start in row and end in col
        deg = scatter_add(edge_weight, row.to(
            self.device), dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]

        x = torch.matmul(x, self.weight)
        out = self.propagate('add', edge_idx, x=x, norm=norm)
        return self.act(out)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self) -> str:
        return '{}({}, {})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape) # Constructs a sparse tensor in COO(rdinate) format with specified values at the given indices.
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, act=F.elu, concat=False):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.act = act
        self.special_spmm = SpecialSpmm()

    def forward(self, input, edge):
        dv = 'cuda' if input.is_cuda else 'cpu'
        # breakpoint()
        N = input.size()[0]
        # edge = adj.nonzero().t() # 获取非0元素的下标，并转置
        edge, _ = add_self_loops(edge, num_nodes=N)

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        # h_prime = torch.where(torch.isnan(h_prime), torch.full_like(h_prime, 0.0), h_prime)

        return self.act(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, device, bias=True) -> None:
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.device = device
        # GRU weights
        self.weight_xz = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xr = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xh = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hz = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hr = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hh = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        for i in range(1, self.n_layer):
            self.weight_xz.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xr.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xh.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hz.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hr.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hh.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))

    def forward(self, x, edge_idx, h, c):
        h_out = torch.zeros_like(h)
        for i in range(self.n_layer):
            # get the Update gate(X) and Reset gate(R)
            z_g = torch.sigmoid(
                self.weight_xz[i](x if i == 0 else h_out[i-1], edge_idx) +
                self.weight_hz[i](h[i], edge_idx))
            r_g = torch.sigmoid(
                self.weight_xr[i](x if i == 0 else h_out[i-1], edge_idx) +
                self.weight_hr[i](h[i], edge_idx))
            # get candidate hidden state
            h_tilde_g = torch.tanh(
                self.weight_xh[i](x if i == 0 else h_out[i-1], edge_idx) +
                self.weight_hh[i](r_g * h[i], edge_idx))
            # get new hidden state
            h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g

        return h_out, c

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, device, bias=True):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.device = device
        self.weight_xi = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xf = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xo = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xc = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hi = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hf = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_ho = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hc = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        for i in range(1, self.n_layer):
            self.weight_xi.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xf.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xo.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xc.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hi.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hf.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_ho.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hc.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))

    def forward(self, x, edge_idx, h, c):
        h_out = torch.zeros_like(h)
        c_out = torch.zeros_like(c)
        for i in range(self.n_layer):
            I_g = torch.sigmoid(self.weight_xi[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hi[i](h[i], edge_idx))
            F_g = torch.sigmoid(self.weight_xf[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hf[i](h[i], edge_idx))
            O_g = torch.sigmoid(self.weight_xo[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_ho[i](h[i], edge_idx))
            c_tilde_g = torch.tanh(self.weight_xc[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hc[i](h[i], edge_idx))
            c_out[i] = F_g * h[i] + I_g * c_tilde_g
            h_out[i] = O_g * torch.tanh(c[i])
        return h_out, c_out


class LSTMSE(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, dropout, alpha, bias=True):
        super(LSTMSE, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.weight_xi = nn.ModuleList([
            SpGraphAttentionLayer(in_dim, hidden_dim, dropout, alpha, act=lambda x: x)])
        self.weight_xf = nn.ModuleList([
            SpGraphAttentionLayer(in_dim, hidden_dim, dropout, alpha, act=lambda x: x)])
        self.weight_xo = nn.ModuleList([
            SpGraphAttentionLayer(in_dim, hidden_dim, dropout, alpha, act=lambda x: x)])
        self.weight_xc = nn.ModuleList([
            SpGraphAttentionLayer(in_dim, hidden_dim, dropout, alpha, act=lambda x: x)])
        self.weight_hi = nn.ModuleList([
            SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x)])
        self.weight_hf = nn.ModuleList([
            SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x)])
        self.weight_ho = nn.ModuleList([
            SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x)])
        self.weight_hc = nn.ModuleList([
            SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x)])
        for i in range(1, self.n_layer):
            self.weight_xi.append(
                SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x))
            self.weight_xf.append(
                SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x))
            self.weight_xo.append(
                SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x))
            self.weight_xc.append(
                SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x))
            self.weight_hi.append(
                SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x))
            self.weight_hf.append(
                SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x))
            self.weight_ho.append(
                SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x))
            self.weight_hc.append(
                SpGraphAttentionLayer(hidden_dim, hidden_dim, dropout, alpha, act=lambda x: x))

    def forward(self, x, edge_idx, h, c):
        h_out = torch.zeros_like(h)
        c_out = torch.zeros_like(c)
        for i in range(self.n_layer):
            I_g = torch.sigmoid(self.weight_xi[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hi[i](h[i], edge_idx))
            F_g = torch.sigmoid(self.weight_xf[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hf[i](h[i], edge_idx))
            O_g = torch.sigmoid(self.weight_xo[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_ho[i](h[i], edge_idx))
            c_tilde_g = torch.tanh(self.weight_xc[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hc[i](h[i], edge_idx))
            c_out[i] = F_g * c[i] + I_g * c_tilde_g     # 原作者这里应该是写错了
            h_out[i] = O_g * torch.tanh(c_out[i])       #
        return h_out, c_out


class VGRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layer, dropout, alpha, device, bias=False):
        super(VGRNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layer = n_layer
        self.device = device
        self.dropout = dropout
        self.alpha = alpha

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU())


        # encode过程使用了图卷积操作
        self.enc = SpGraphAttentionLayer(2 * h_dim, h_dim, dropout, alpha)
        self.enc_mu = SpGraphAttentionLayer(h_dim, z_dim, dropout, alpha, act=lambda x: x)
        self.enc_logvar = SpGraphAttentionLayer(h_dim, z_dim, dropout, alpha, act=F.softplus)

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ELU())
        self.prior_mu = nn.Sequential(nn.Linear(h_dim, z_dim))
        self.prior_logvar = nn.Sequential(
            nn.Linear(h_dim, z_dim), nn.Softplus())

        # self.gru = GRU(2 * h_dim, h_dim, n_layer, device, bias)
        self.lstm = LSTMSE(2 * h_dim, h_dim, n_layer, dropout, alpha, bias)
        self.decoder = MlpDecoder(z_dim, 1)

    def forward(self, x, edge_idx_list, adj_orig_dense_list, hidden_in=None):
        assert len(adj_orig_dense_list) == len(edge_idx_list)   # Snapshots
        kld_loss = 0.0
        nll_loss = 0.0
        mse_loss = 0.0
        all_enc_mu, all_enc_logvar = [], []
        all_prior_mu, all_prior_logvar = [], []
        all_dec_t, all_z_t = [], []
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layer, x.size(1),
                         self.h_dim)).to(self.device)
            c = Variable(torch.zeros(self.n_layer, x.size(1),
                         self.h_dim)).to(self.device)
        else:
            h = Variable(hidden_in).to(self.device)
            c = Variable(hidden_in).to(self.device) # c也使用h初始化吗？？？
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            # Encoder, Inference step
            enc_t = self.enc(
                torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t].to(self.device)) # -1 代表最后一层
            enc_mu_t = self.enc_mu(enc_t, edge_idx_list[t].to(self.device))
            enc_logvar_t = self.enc_logvar(enc_t, edge_idx_list[t].to(self.device))
            # Prior
            # breakpoint()
            prior_t = self.prior(h[-1])
            prior_mu_t = self.prior_mu(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)
            # Reparameterization
            z_t = self._reparameterized(enc_mu_t, enc_logvar_t)
            phi_z_t = self.phi_z(z_t)
            # Decode from z_t
            dec_t = self.decoder(z_t) # shape:(N,N)

            # Recurrence
            h, c = self.lstm(
                torch.cat([phi_x_t, phi_z_t], 1),
                edge_idx_list[t].to(self.device), h, c)
            # Question: Is n_nodes dynamic? Yes
            if not torch.is_tensor(adj_orig_dense_list[t]):
                adj_orig_dense_t = adj_orig_dense_list[t].copy()
                adj_orig_dense_t = torch.from_numpy(adj_orig_dense_t.A)
            else:
                adj_orig_dense_t = adj_orig_dense_list[t].clone()
            adj_orig_dense_t = adj_orig_dense_t.float().to(self.device)
            n_nodes = adj_orig_dense_t.size()[0] # 针对稠密矩阵
            enc_mu_t = enc_mu_t[:n_nodes, :]
            enc_logvar_t = enc_logvar_t[:n_nodes, :]
            prior_mu_t = prior_mu_t[:n_nodes, :]
            prior_logvar_t = prior_logvar_t[:n_nodes, :]
            dec_t = dec_t[:n_nodes, :n_nodes]

            # Calculate and accumulate the KL-Divergence and Binary Cross-entropy
            kld_loss = kld_loss + self._kld_gauss(enc_mu_t, enc_logvar_t, prior_mu_t, prior_logvar_t) # 两个高斯分布的dl散度，有解析解
            # breakpoint()
            nll_loss = nll_loss + self._nll_bernoulli(dec_t, adj_orig_dense_t) #重构损失
            # mse_loss = mse_loss + self._mse(dec_t, adj_orig_dense_t) #重构损失
            # mse_loss = mse_loss + self._focalloss(dec_t, adj_orig_dense_t) #重构损失
            # Save the parameters learned at time t
            all_enc_mu.append(enc_mu_t)
            all_enc_logvar.append(enc_logvar_t)
            all_prior_mu.append(prior_mu_t)
            all_prior_logvar.append(prior_logvar_t)
            all_dec_t.append(dec_t)
            all_z_t.append(z_t)
        return kld_loss, nll_loss, all_enc_mu, all_prior_mu, h

    # def decoder(self, z):
    #     # outputs = InnerProductDecoder(act=lambda x: x)(z)
    #     outputs = MlpDecoder(in_dim=z.shape[1], out_dim=3).to(self.device)(z)
    #     return outputs

    def _reparameterized(self, mu, logvar):
        epsilon = Variable(torch.FloatTensor(
            logvar.size()).normal_()).to(self.device)
        return epsilon.mul(logvar).add_(mu)

    def _kld_gauss(self, mu_1, logvar_1, mu_2, logvar_2):
        num_nodes = mu_1.size()[0]
        # kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
        #                (torch.pow(std_1, 2) + torch.pow(mean_1 - mean_2, 2)) /
        #                torch.pow(std_2, 2) - 1)
        kld_element = 2 * (logvar_2 - logvar_1) + (torch.exp(2*logvar_1) +
                                                   torch.pow(mu_1-mu_2, 2)) / torch.exp(2*logvar_2)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        # Negtive Edges / Positive Edges, positive weight
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / \
            float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(
            input=logits, target=target_adj_dense, pos_weight=posw, reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return -nll_loss

    def _mse(self, pred, target_adj):
        return F.mse_loss(pred, target_adj)

    def _focalloss(self, pred, target_adj):
        alpha = 0.25
        gamma = 2
        # breakpoint()
        pred = torch.reshape(pred, (-1, 2))
        target_adj = target_adj.flatten()
        ce_loss = F.cross_entropy(pred, target_adj.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
        return focal_loss

class InnerProductDecoder(nn.Module): # 注意：此解码器是无参数的！！
    def __init__(self, act=torch.sigmoid, dropout=0.5, training=True):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.dropout = dropout
        self.training = training

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)

class MatrixDecoder(nn.Module):
    def __init__(self, act=torch.relu, dropout=0.5, z_dim=0, training=True) -> None:
        super(MatrixDecoder, self).__init__()
        self.act = act
        self.dropout = dropout
        self.training = training
        self.transfer_matrix = nn.Parameter(torch.empty(size=(z_dim, z_dim)))
        nn.init.xavier_uniform_(self.transfer_matrix.data, gain=1.414)

    def forward(self, inp):
        # self.transfer_matrix = self.transfer_matrix.to(inp.device)
        # breakpoint()
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = inp@self.transfer_matrix@inp.t()
        return self.act(x)

class MlpDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, training=True) -> None:
        super(MlpDecoder, self).__init__()
        self.dropout=dropout
        self.training = training
        self.in_dim = in_dim
        self.a = nn.Parameter(torch.empty(size=(2*in_dim, out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, inp):
        # inp:(N,C)
        # output: (N,N)
        inp = F.dropout(inp, self.dropout, training=self.training)
        h1 = torch.mm(inp, self.a[:self.in_dim, :])
        h2 = torch.mm(inp, self.a[self.in_dim:, :])
        # h1 = torch.unsqueeze(h1, dim=1)
        # h2 = torch.unsqueeze(h2, dim=0)
        e = h1 + h2.T
        return torch.sigmoid(e)