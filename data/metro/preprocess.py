import numpy as np
import os
import scipy.sparse as sp


start, end = 11, 45
filename = 'source_data.npy'
folder_path = './origin_data'
def func(folder_path):
    assert os.path.exists(folder_path)
    target_data = []
    for file in os.listdir(folder_path):
        cur_data = np.load(folder_path+'/'+file)
        target_data.append(cur_data[:,:,start:end]) # 避开了那些无人的时段
    target_data = np.concatenate(target_data, axis=-1)
    np.save(filename, target_data)

if __name__=="__main__":
    assert not os.path.exists(filename)
    func(folder_path)