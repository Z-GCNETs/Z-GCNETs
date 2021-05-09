import os
import numpy as np
import pandas as pd
import networkx as nx
import zigzag.zigzagtools as zzt
from scipy.spatial.distance import squareform
import dionysus as d
import matplotlib.pyplot as plt
import time
from ripser import ripser
from persim import plot_diagrams, PersImage
path = os.getcwd()

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def load_st_fulldataset(dataset, val_ratio, test_ratio):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join(path + '/data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0:4]  #three dimensions, traffic flow data
        data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
    elif dataset == 'PEMSD8':
        data_path = os.path.join(path + '/data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0:4]  #three dimensions, traffic flow data
        data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data_train, data_val, data_test


# parameter setting
alpha = 0.5
NVertices = 307 # Number of vertices
scaleParameter = 1.0 # Scale Parameter (Maximum) # the maximal edge weight #
maxDimHoles = 2 # Maximum Dimension of Holes (It means.. 0 and 1)
sizeWindow = 12 # Number of Graphs

# Zigzag persistence diagram (ZPD) for the regular sliding window
def zigzag_persistence_diagrams(dataset, index, alpha, NVertices, scaleParameter, maxDimHoles, sizeWindow, train = True, val = False, val_ratio = None, test_ratio = None):
    train_data, val_data, test_data = load_st_fulldataset(dataset=dataset, val_ratio = val_ratio, test_ratio = test_ratio)
    if train:
        PEMS_features = train_data
    elif val:
        PEMS_features = val_data
    else:
        PEMS_features = test_data
    PEMS_net_dataset = pd.read_csv(path + '/data/PEMS0' + str(dataset)[5] + '/distance.csv', header=0)
    PEMS_net_edges = PEMS_net_dataset.values[:, 0:2]
    PEMS_net_edgelist = [(int(u), int(v)) for u, v in PEMS_net_edges]
    PEMS_net = nx.Graph()
    PEMS_net.add_edges_from(PEMS_net_edgelist)

    PEMS_networks = np.zeros(shape=(sizeWindow, len(PEMS_net_edgelist), 3), dtype=np.float32)
    for i in range(index, index+sizeWindow):
        PEMS_networks[i - index, :, 0:2] = np.array(PEMS_net_edgelist)
        tmp_features = PEMS_features[i, :, :].reshape(NVertices, -1)
        for j in range(len(PEMS_net_edgelist)):
            u, v = PEMS_net_edgelist[j]
            if np.sqrt(np.sum((tmp_features[int(u), :] - tmp_features[int(v), :]) ** 2)) == 0:
                PEMS_networks[i - index, j, 2] = 1e-5
            else:
                PEMS_networks[i - index, j, 2] = np.sqrt(np.sum((tmp_features[int(u), :] - tmp_features[int(v), :]) ** 2))

        tmp_max = np.max(PEMS_networks[i - index, :, 2])
        PEMS_networks[i - index, :, 2] = PEMS_networks[i - index, :, 2] / tmp_max
        # Cut edges, i.e., with \alpha threshold
        tmp_cut_pair = np.where(PEMS_networks[i - index, :, 2] > alpha)
        PEMS_networks[i - index, tmp_cut_pair[0], 2] = 0.

    # To measure time
    start_time = time.time()

    # Open all sets (point-cloud/Graphs)
    Graphs = []
    for i in range(0, sizeWindow):
        edgesList = PEMS_networks[i, :, :]
        Graphs.append(edgesList)
    print("  --- End Loading...")  # Ending

    # Generate Graph
    GraphsNetX = []
    for ii in range(0, sizeWindow):
        g = nx.Graph()
        g.add_nodes_from(list(range(0, NVertices)))  # Add vertices...
        if (Graphs[ii].ndim == 1 and len(Graphs[ii]) > 0):
            g.add_edge(int(Graphs[ii][0]), int(Graphs[ii][1]), weight=Graphs[ii][2])
        elif (Graphs[ii].ndim == 2):
            for k in range(0, Graphs[ii].shape[0]):
                g.add_edge(int(Graphs[ii][k, 0]), int(Graphs[ii][k, 1]), weight=Graphs[ii][k, 2])
        GraphsNetX.append(g)

    # Building unions and computing distance matrices
    print("Building unions and computing distance matrices...")  # Beginning
    GUnions = []
    MDisGUnions = []
    for i in range(0, sizeWindow - 1):
        # --- To concatenate graphs
        unionAux = []
        MDisAux = np.zeros((2 * NVertices, 2 * NVertices))
        A = nx.adjacency_matrix(GraphsNetX[i]).todense()
        B = nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
        # ----- Version Original (2)
        C = (A + B) / 2
        A[A == 0] = 1.1
        A[range(NVertices), range(NVertices)] = 0
        B[B == 0] = 1.1
        B[range(NVertices), range(NVertices)] = 0
        MDisAux[0:NVertices, 0:NVertices] = A
        C[C == 0] = 1.1
        C[range(NVertices), range(NVertices)] = 0
        MDisAux[NVertices:(2 * NVertices), NVertices:(2 * NVertices)] = B
        MDisAux[0:NVertices, NVertices:(2 * NVertices)] = C
        MDisAux[NVertices:(2 * NVertices), 0:NVertices] = C.transpose()
        # Distance in condensed form
        pDisAux = squareform(MDisAux)
        # --- To save unions and distances
        GUnions.append(unionAux)  # To save union
        MDisGUnions.append(pDisAux)  # To save distance matrix
    print("  --- End unions...")  # Ending

    # To perform Ripser computations
    print("Computing Vietoris-Rips complexes...")  # Beginning

    GVRips = []
    for jj in range(0, sizeWindow - 1):
        print(jj)
        ripsAux = d.fill_rips(MDisGUnions[jj], maxDimHoles, scaleParameter)
        GVRips.append(ripsAux)
    print("  --- End Vietoris-Rips computation")  # Ending

    # Shifting filtrations...
    print("Shifting filtrations...")  # Beginning
    GVRips_shift = []
    GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
    for kk in range(1, sizeWindow - 1):
        shiftAux = zzt.shift_filtration(GVRips[kk], NVertices * kk)
        GVRips_shift.append(shiftAux)
    print("  --- End shifting...")  # Ending

    # To Combine complexes
    print("Combining complexes...")  # Beginning
    completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
    for uu in range(2, sizeWindow - 1):
        completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
    print("  --- End combining")  # Ending

    # To compute the time intervals of simplices
    print("Determining time intervals...")  # Beginning
    time_intervals = zzt.build_zigzag_times(completeGVRips, NVertices, sizeWindow)
    print("  --- End time")  # Beginning

    # To compute Zigzag persistence
    print("Computing Zigzag homology...")  # Beginning
    G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
    print("  --- End Zigzag")  # Beginning

    # To show persistence intervals
    window_PD = []
    # Personalized plot
    for vv, dgm in enumerate(G_dgms):
        print("Dimension:", vv)
        if (vv < 2):
            matBarcode = np.zeros((len(dgm), 2))
            k = 0
            for p in dgm:
                matBarcode[k, 0] = p.birth
                matBarcode[k, 1] = p.death
                k = k + 1
            matBarcode = matBarcode / 2
            window_PD.append(matBarcode)

    # Timing
    print("TIME: " + str((time.time() - start_time)) + " Seg ---  " + str(
        (time.time() - start_time) / 60) + " Min ---  " + str((time.time() - start_time) / (60 * 60)) + " Hr ")

    return window_PD


# Zigzag persistence diagram (ZPD) for the nested sliding window
def nested_zigzag_persistence_diagrams(dataset, index, alpha, NVertices, scaleParameter, maxDimHoles, sizeWindow, train = True, val = False, val_ratio = None, test_ratio = None):
    train_data, val_data, test_data = load_st_fulldataset(dataset=dataset, val_ratio = val_ratio, test_ratio = test_ratio)
    if train:
        PEMS_features = train_data
    elif val:
        PEMS_features = val_data
    else:
        PEMS_features = test_data
    PEMS_net_dataset = pd.read_csv(path + '/data/PEMS0' + str(dataset)[5] + '/distance.csv', header=0)
    PEMS_net_edges = PEMS_net_dataset.values[:, 0:2]
    PEMS_net_edgelist = [(int(u), int(v)) for u, v in PEMS_net_edges]
    PEMS_net = nx.Graph()
    PEMS_net.add_edges_from(PEMS_net_edgelist)

    PEMS_networks = np.zeros(shape=(sizeWindow, len(PEMS_net_edgelist), 3), dtype=np.float32)
    for i in range(index-sizeWindow, index):
        PEMS_networks[i - (index-sizeWindow), :, 0:2] = np.array(PEMS_net_edgelist)
        tmp_features = PEMS_features[i, :, :].reshape(NVertices, -1)
        for j in range(len(PEMS_net_edgelist)):
            u, v = PEMS_net_edgelist[j]
            if np.sqrt(np.sum((tmp_features[int(u), :] - tmp_features[int(v), :]) ** 2)) == 0:
                PEMS_networks[i - (index-sizeWindow), j, 2] = 1e-5
            else:
                PEMS_networks[i - (index-sizeWindow), j, 2] = np.sqrt(np.sum((tmp_features[int(u), :] - tmp_features[int(v), :]) ** 2))

        tmp_max = np.max(PEMS_networks[i - (index-sizeWindow), :, 2])
        PEMS_networks[i - (index-sizeWindow), :, 2] = PEMS_networks[i - (index-sizeWindow), :, 2] / tmp_max
        # Cut edges, i.e., with \alpha threshold
        tmp_cut_pair = np.where(PEMS_networks[i - (index-sizeWindow), :, 2] > alpha)
        PEMS_networks[i - (index-sizeWindow), tmp_cut_pair[0], 2] = 0.

    # To measure time
    start_time = time.time()

    # Open all sets (point-cloud/Graphs)
    Graphs = []
    for i in range(0, sizeWindow):
        edgesList = PEMS_networks[i, :, :]
        Graphs.append(edgesList)
    print("  --- End Loading...")  # Ending

    # Generate Graph
    GraphsNetX = []
    for ii in range(0, sizeWindow):
        g = nx.Graph()
        g.add_nodes_from(list(range(0, NVertices)))  # Add vertices...
        if (Graphs[ii].ndim == 1 and len(Graphs[ii]) > 0):
            g.add_edge(int(Graphs[ii][0]), int(Graphs[ii][1]), weight=Graphs[ii][2])
        elif (Graphs[ii].ndim == 2):
            for k in range(0, Graphs[ii].shape[0]):
                g.add_edge(int(Graphs[ii][k, 0]), int(Graphs[ii][k, 1]), weight=Graphs[ii][k, 2])
        GraphsNetX.append(g)

    # Building unions and computing distance matrices
    print("Building unions and computing distance matrices...")  # Beginning
    GUnions = []
    MDisGUnions = []
    for i in range(0, sizeWindow - 1):
        # --- To concatenate graphs
        unionAux = []
        MDisAux = np.zeros((2 * NVertices, 2 * NVertices))
        A = nx.adjacency_matrix(GraphsNetX[i]).todense()
        B = nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
        # ----- Version Original (2)
        C = (A + B) / 2
        A[A == 0] = 1.1
        A[range(NVertices), range(NVertices)] = 0
        B[B == 0] = 1.1
        B[range(NVertices), range(NVertices)] = 0
        MDisAux[0:NVertices, 0:NVertices] = A
        C[C == 0] = 1.1
        C[range(NVertices), range(NVertices)] = 0
        MDisAux[NVertices:(2 * NVertices), NVertices:(2 * NVertices)] = B
        MDisAux[0:NVertices, NVertices:(2 * NVertices)] = C
        MDisAux[NVertices:(2 * NVertices), 0:NVertices] = C.transpose()
        # Distance in condensed form
        pDisAux = squareform(MDisAux)
        # --- To save unions and distances
        GUnions.append(unionAux)  # To save union
        MDisGUnions.append(pDisAux)  # To save distance matrix
    print("  --- End unions...")  # Ending

    # To perform Ripser computations
    print("Computing Vietoris-Rips complexes...")  # Beginning

    GVRips = []
    for jj in range(0, sizeWindow - 1):
        print(jj)
        ripsAux = d.fill_rips(MDisGUnions[jj], maxDimHoles, scaleParameter)
        GVRips.append(ripsAux)
    print("  --- End Vietoris-Rips computation")  # Ending

    # Shifting filtrations...
    print("Shifting filtrations...")  # Beginning
    GVRips_shift = []
    GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
    for kk in range(1, sizeWindow - 1):
        shiftAux = zzt.shift_filtration(GVRips[kk], NVertices * kk)
        GVRips_shift.append(shiftAux)
    print("  --- End shifting...")  # Ending

    # To Combine complexes
    print("Combining complexes...")  # Beginning
    completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
    for uu in range(2, sizeWindow - 1):
        completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
    print("  --- End combining")  # Ending

    # To compute the time intervals of simplices
    print("Determining time intervals...")  # Beginning
    time_intervals = zzt.build_zigzag_times(completeGVRips, NVertices, sizeWindow)
    print("  --- End time")  # Beginning

    # To compute Zigzag persistence
    print("Computing Zigzag homology...")  # Beginning
    G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
    print("  --- End Zigzag")  # Beginning

    # To show persistence intervals
    window_PD = []
    # Personalized plot
    for vv, dgm in enumerate(G_dgms):
        print("Dimension:", vv)
        if (vv < 2):
            matBarcode = np.zeros((len(dgm), 2))
            k = 0
            for p in dgm:
                matBarcode[k, 0] = p.birth
                matBarcode[k, 1] = p.death
                k = k + 1
            matBarcode = matBarcode / 2
            window_PD.append(matBarcode)

    # Timing
    print("TIME: " + str((time.time() - start_time)) + " Seg ---  " + str(
        (time.time() - start_time) / 60) + " Min ---  " + str((time.time() - start_time) / (60 * 60)) + " Hr ")

    return window_PD


# Zigzag persistence image
def zigzag_persistence_images(dgms, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
    PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    # Compute Zigzag persistence image
    P0, P1 = np.reshape(dgms[int(dimensional)][:, 0], [1, 1, -1]), np.reshape(dgms[int(dimensional)][:, 1], [1, 1, -1])
    weight = np.abs(P1 - P0)
    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    if return_raw:
        lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
        lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
    else:
        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

    output = [lw, lsum] if return_raw else Zfinal

    if normalization:
        norm_output = (output - np.min(output))/(np.max(output) - np.min(output))
    else:
        norm_output = output

    return norm_output