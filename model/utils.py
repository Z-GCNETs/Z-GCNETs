import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

path = os.getcwd()

def get_adjacency_matrxix(dataset, number_nodes):
    PEMS_net_dataset = pd.read_csv(path + '/data/PEMS0' + str(dataset)[5] + '/distance.csv', header=0)
    PEMS_net_edges = PEMS_net_dataset.values[:, 0:2]
    A = np.zeros((number_nodes, number_nodes), dtype= np.float32)
    for i in range(PEMS_net_edges.shape[0]):
        A[int(PEMS_net_edges[i,0] -1 ), int(PEMS_net_edges[i,1] -1 )] = 1.
        A[int(PEMS_net_edges[i, 1] - 1), int(PEMS_net_edges[i, 0] -1 )] = 1.
    A = sp.csr_matrix(A)
    return A


# Fractional power
def fractional_fltr(adj, number_nodes, sigma, gamma):
    degrees = np.array(adj.sum(1)).flatten()
    degrees[np.isinf(degrees)] = 0.
    D = sp.diags(degrees, 0)
    L_darray = (D - adj).toarray()
    D, V = np.linalg.eigh(L_darray, 'U')
    M_gamma_Lambda = D
    M_gamma_Lambda[M_gamma_Lambda < 1e-5] = 0
    M_V = V
    M_gamma_Lambda = np.float_power(M_gamma_Lambda, gamma)
    M_gamma_Lambda = np.diag(M_gamma_Lambda, 0)
    M_gamma_Lambda = sp.csr_matrix(M_gamma_Lambda)
    M_V = sp.csr_matrix(M_V)
    Lg = M_V * M_gamma_Lambda
    Lg = Lg * sp.csr_matrix.transpose(M_V)
    Lg = Lg.toarray()
    Lg = Lg.reshape(1, -1)
    Lg[abs(Lg) < 1e-5] = 0.
    Lg = Lg.reshape(number_nodes, -1)
    Dg = np.diag(np.diag(Lg))
    Ag = Dg - Lg
    Ag = sp.csr_matrix(Ag)
    power_Dg_l = np.float_power(np.diag(Dg), -sigma)
    power_Dg_l = sp.csr_matrix(np.diag(power_Dg_l))
    power_Dg_r = np.float_power(np.diag(Dg), (sigma - 1))
    power_Dg_r = sp.csr_matrix(np.diag(power_Dg_r))
    fractional_fltr = power_Dg_l * Ag
    fractional_fltr = fractional_fltr * power_Dg_r
    return fractional_fltr

trans_adj = get_adjacency_matrxix(dataset= 'PEMS04', number_nodes= 307)
#frac_fltr = fractional_fltr(adj= trans_adj, number_nodes= 307, sigma= 0.5, gamma= 2.)

sp.save_npz('PEMSD4_adj.npz', trans_adj)
#test = sp.load_npz('PEMSD4_fltr.npz')



