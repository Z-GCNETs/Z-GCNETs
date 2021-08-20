import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # matplotlib version 3.1.1
path = os.getcwd()


def load_st_dataset(dataset):
    # output B, N, D
    if dataset == 'Decentraland':
        data_path = os.path.join(path + '/token_data/Decentraland_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0]  #1 dimension, degree
    elif dataset == 'Bytom':
        data_path = os.path.join(path + '/token_data/Bytom_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0]  #1 dimension, degree
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

def load_topo_dataset(dataset, H_type):
    # H_type opens H0 for token networks
    # Bytom - (259, 100, 100)
    # Decentraland - (193, 100, 100)
    if dataset == 'Decentraland':
        topo_data = np.load(path + '/token_ZPI/Decentraland_' + H_type + '_100.npz')['arr_0']
    elif dataset == 'Bytom':
        topo_data = np.load(path + '/token_ZPI/Bytom_' + H_type + '_100.npz')['arr_0']

    return topo_data

def bytedate2num(fmt):
    def converter(b):
        return mdates.strpdate2num(fmt)(b.decode('ascii'))
    return converter

date_converter = bytedate2num('%m/%d/%Y')
def load_token_price(dataset):
    if dataset == 'Decentraland':
        token_price = np.loadtxt(path + '/Prices_Closed/' + dataset + '.txt', delimiter='\t', converters={0: date_converter}, usecols=range(0, 5))
    elif dataset == 'Bytom':
        token_price = np.loadtxt(path + '/Prices_Closed/' + dataset + '.txt', delimiter='\t', converters={0: date_converter}, usecols=range(0, 5))
    token_close_price = token_price[:,4].reshape(-1,1)
    sequence_token_close_price = token_close_price[::-1]

    return sequence_token_close_price
