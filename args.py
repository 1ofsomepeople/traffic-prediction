### CONFIGS ###
# datasets: 'dblp', 'enron10', 'fb'
from torch import dropout


dataset = 'metro'
h_dim = 32
z_dim = 16
n_layers = 1
clip = 10
learning_rate = 1e-3
epochs = 10
eps = 1e-10
seq_start = 0
tst_after = 0
predict = 306 # 34 per day * 9 days
folder_path = './data/metro'
seq_len = 12
start_idx = 10
batch_size = 1
dropout = 0.2
alpha = 0.2