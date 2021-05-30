import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
path = os.getcwd()


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join(path + '/data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  #three dimensions, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join(path + '/data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  #three dimensions, traffic flow data
        print(data.shape)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

def load_topo_dataset(data_type, H_type):
    topo_data_train = np.load(path + '/tda_data/tda_PEMS0' + data_type + '/PEMSD'+ data_type + '_train_' + H_type + '_100.npz')['arr_0']
    topo_data_val = np.load(path + '/tda_data/tda_PEMS0' + data_type + '/PEMSD'+ data_type + '_val_' + H_type + '_100.npz')['arr_0']
    topo_data_test = np.load(path + '/tda_data/tda_PEMS0' + data_type + '/PEMSD'+ data_type + '_test_' + H_type + '_100.npz')['arr_0']

    return topo_data_train, topo_data_val, topo_data_test


def load_window_topo_dataset(H_type):
    topo_data_train = np.load(path + '/tda_data/tda_PEMS04/nested_train_' + H_type + '.npz')['arr_0']
    topo_data_val = np.load(path + '/tda_data/tda_PEMS04/nested_val_' + H_type + '.npz')['arr_0']
    topo_data_test = np.load(path + '/tda_data/tda_PEMS04/nested_test_' + H_type + '.npz')['arr_0']

    return topo_data_train, topo_data_val, topo_data_test

def load_add_topo_dataset(H_type):
    topo_data_train = np.load(path + '/tda_data/tda_PEMS04/add_nested_train_' + H_type + '.npz')['arr_0']
    topo_data_val = np.load(path + '/tda_data/tda_PEMS04/add_nested_val_' + H_type + '.npz')['arr_0']
    topo_data_test = np.load(path + '/tda_data/tda_PEMS04/add_nested_test_' + H_type + '.npz')['arr_0']

    return topo_data_train, topo_data_val, topo_data_test

def load_full_topo_dataset(H_type):
    topo_data_train = np.load(path + '/tda_data/tda_PEMS04/full_nested_train_' + H_type + '.npz')['arr_0']
    topo_data_val = np.load(path + '/tda_data/tda_PEMS04/full_nested_val_' + H_type + '.npz')['arr_0']
    topo_data_test = np.load(path + '/tda_data/tda_PEMS04/full_nested_test_' + H_type + '.npz')['arr_0']

    return topo_data_train, topo_data_val, topo_data_test
