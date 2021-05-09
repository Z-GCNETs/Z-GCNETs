import torch
import numpy as np
import torch.utils.data
from add_window import Add_Window_Horizon, PI_Window_Horizon
from load_dataset import load_st_dataset, load_topo_dataset, load_full_topo_dataset
from normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def triple_data_loader(X, PI, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, PI, Y = TensorFloat(X), TensorFloat(PI), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, PI, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # (B, N, D)
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio

    print("================+++++++++++++================")
    print(args.test_ratio)
    print("================+++++++++++++================")

    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

    #add time window
    single = False
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single) # lag (i.e., window) = 12, horizon = 12, single = True
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ######################get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


def get_topo_dataloader(args, data_type, H_type, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    #load raw st dataset
    data = load_st_dataset(args.dataset) # (B, N, D)
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio

    topo_data_train, topo_data_val, topo_data_test = load_topo_dataset(data_type, H_type)

    print("================+++++++++++++================")
    print(args.test_ratio)
    print("================+++++++++++++================")

    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

    single = False
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    topo_tra = topo_data_train
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    topo_val = topo_data_val
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    topo_test = topo_data_test
    print('Train: x, ZPI, y ->', x_tra.shape, topo_tra.shape, y_tra.shape)
    print('Val: x, ZPI, y ->', x_val.shape, topo_val.shape, y_val.shape)
    print('Test: x, ZPI, y ->', x_test.shape, topo_test.shape, y_test.shape)
    ######################get triple dataloader######################
    train_dataloader = triple_data_loader(x_tra, topo_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = triple_data_loader(x_val, topo_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = triple_data_loader(x_test, topo_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


import argparse
DATASET = 'PEMSD4'
NODE_NUM = 307
parser = argparse.ArgumentParser(description='PyTorch dataloader')
parser.add_argument('--dataset', default=DATASET, type=str)
parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--test_ratio', default=0.2, type=float)
parser.add_argument('--lag', default=12, type=int)
parser.add_argument('--horizon', default=12, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--column_wise', default=False, type=bool)
args = parser.parse_args()

train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False)
