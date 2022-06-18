import numpy as np
import scipy.sparse as sp
import os
import pickle
# import torch

data = np.load('source_data.npy')
_,_,t = data.shape
adj_time_list = []
adj_orig_dense_list = []
filename1 = 'adj_time_list.pickle'
filename2 = 'adj_orig_dense_list.pickle'
for i in range(t):
    # 离散化处理，得到每个节点对的标签，>=30代表边标签为1，意为此节点对之间需求较大，存在“边”，否则不存在“边”，标签为0.
    x = data[:,:,i]
    x = x.astype(np.int8)
    # x[(0<x) & (x<30)] = 1
    # x[100<=x] = 1 # 这里好像不起作用？？
    x = np.where(x>=30, 1, 0)
    adj_time_list.append(sp.csr_matrix(x))
    # adj_orig_dense_list.append(torch.from_numpy(data[:,:,i]))

with open(filename1, 'wb') as file:
    pickle.dump(adj_time_list, file)

with open(filename2, 'wb') as file:
    pickle.dump(adj_time_list, file)