import os
import numpy as np

#data_path = os.path.join('../data/PeMSD4/pems04.npz')
path = os.getcwd()
data = np.load(path + '/data/pems04.npz')['data'] #shape is (sequence_length, num_of_vertices, num_of_features)
print(data.shape) # (16992, 307, 3)