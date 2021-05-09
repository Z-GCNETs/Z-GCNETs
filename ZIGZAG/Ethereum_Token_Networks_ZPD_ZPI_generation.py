import numpy as np
import matplotlib.pyplot as plt
import glob 
from datetime import datetime, timedelta
import os
import networkx as nx
import zigzag.zigzagtools as zzt
from scipy.spatial.distance import squareform
import dionysus as d
import time

path = os.getcwd()

# Ethereum token networks zigzag persistence diagram (ZPD)
def token_zigzag_persistence_diagrams(dataset = None, index = None, NVertices = 100, scaleParameter = 1., maxDimHoles = 2, sizeWindow = 7):
    # To measure time
    start_time = time.time()
    # Generate Graph #
    GraphsNetX = []
    index = index + 1 # since data label is from 1
    for ii in range(index, index + sizeWindow):
        g = nx.empty_graph(NVertices)
        tmp_W_doc = path + '/ReduceDataset100/EdgeList/' + str(dataset) + '/EdgeList_W/W_Bytom' + str(ii) + '.txt'
        tmp_W = np.loadtxt(tmp_W_doc)
        if tmp_W.shape[0] == 0:
            pass
        else:
            tmp_W = tmp_W.reshape(-1, 3)
            for i in range(tmp_W.shape[0]):
                g.add_edge(int(tmp_W[i, 0] - 1), int(tmp_W[i, 1] - 1), weight=tmp_W[i, 2])
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
        # To save unions and distances
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
    window_ZPD = []
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
            window_ZPD.append(matBarcode)

    # Timing
    print("TIME: " + str((time.time() - start_time)) + " Seg ---  " + str(
        (time.time() - start_time) / 60) + " Min ---  " + str((time.time() - start_time) / (60 * 60)) + " Hr ")

    return window_ZPD


# Ethereum token networks zigzag persistence image (ZPI)
def token_zigzag_persistence_images(dgms, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
    PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    # Compute zigzag persistence image
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
