import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import pdist,squareform
from DNN import CNN
import numpy as np


# ZPI * Spatial GC Layer || ZPI * Temporal GC Layer
class TLSGCNCNN(nn.Module):
    def __init__(self, dim_in, dim_out, window_len, link_len, embed_dim):
        super(TLSGCNCNN, self).__init__()
        self.link_len = link_len
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(dim_out/2)))
        if (dim_in-1)%16 ==0:
            self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, link_len, 1, int(dim_out / 2)))
        else:
            self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, link_len, int(dim_in/2), int(dim_out / 2)))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.T = nn.Parameter(torch.FloatTensor(window_len))
        self.cnn = CNN(int(dim_out / 2))
    def forward(self, x, x_window, node_embeddings, zigzag_PI):
        #S1: Laplacian construction
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        #S2: Laplacianlink
        for k in range(2, self.link_len):
            support_set.append(torch.mm(supports, support_set[k-1]))
        supports = torch.stack(support_set, dim=0)

        #S3: spatial graph convolution
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #N, link_len, dim_in, dim_out/2
        bias = torch.matmul(node_embeddings, self.bias_pool) #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x) #B, link_len, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3) #B, N, link_len, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) #B, N, dim_out/2

        #S4: temporal graph convolution
        weights_window = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_window) #N, link_len, dim_in, dim_out/2
        x_w1 = torch.einsum("knm,btmi->btkni",supports, x_window) #B, T, link_len, N, dim_in
        x_w1 = x_w1.permute(0,1,3,2,4) #B, T, N, link_len, dim_in
        x_w = torch.einsum('btnki,nkio->btno', x_w1, weights_window) #B, T, N, dim_out/2
        x_w = x_w.permute(0, 2, 3, 1) #B, N, dim_out/2, T
        x_wconv = torch.matmul(x_w, self.T) #B, N, dim_out/2

        #S5: zigzag persistence representation learning
        topo_cnn = self.cnn(zigzag_PI) #B, dim_out/2, dim_out/2
        x_tgconv = torch.einsum('bno,bo->bno',x_gconv, topo_cnn)
        x_twconv = torch.einsum('bno,bo->bno',x_wconv, topo_cnn)

        #S6: combination operation
        x_gwconv = torch.cat([x_tgconv, x_twconv], dim = -1) + bias #B, N, dim_out
        return x_gwconv


# ZPI * Spatial GC Layer || ZPI * Feature Transformation (on Temporal Features)
# with less parameters
class TFLSGCNCNN(nn.Module):
    def __init__(self, dim_in, dim_out, window_len, link_len, embed_dim):
        super(TFLSGCNCNN, self).__init__()
        self.link_len = link_len
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(dim_out/2)))
        if (dim_in-1)%16 ==0:
            self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, 1, int(dim_out / 2)))
        else:
            self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, int(dim_in/2), int(dim_out / 2)))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.T = nn.Parameter(torch.FloatTensor(window_len))
        self.cnn = CNN(int(dim_out / 2))
    def forward(self, x, x_window, node_embeddings, zigzag_PI):
        #S1: Laplacian construction
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        #S2: Laplacianlink
        for k in range(2, self.link_len):
            support_set.append(torch.mm(supports, support_set[k-1]))
        supports = torch.stack(support_set, dim=0)

        #S3: spatial graph convolution
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #N, link_len, dim_in, dim_out/2
        bias = torch.matmul(node_embeddings, self.bias_pool) #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x) #B, link_len, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3) #B, N, link_len, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) #B, N, dim_out/2

        #S4: temporal feature transformation
        weights_window = torch.einsum('nd,dio->nio', node_embeddings, self.weights_window)  #N, dim_in, dim_out/2
        x_w = torch.einsum('btni,nio->btno', x_window, weights_window)  #B, T, N, dim_out/2
        x_w = x_w.permute(0, 2, 3, 1)  #B, N, dim_out/2, T
        x_wconv = torch.matmul(x_w, self.T)  #B, N, dim_out/2

        #S5: zigzag persistence representation learning
        topo_cnn = self.cnn(zigzag_PI) #B, dim_out/2, dim_out/2
        x_tgconv = torch.einsum('bno,bo->bno',x_gconv, topo_cnn)
        x_twconv = torch.einsum('bno,bo->bno',x_wconv, topo_cnn)

        #S6: combination operation
        x_gwconv = torch.cat([x_tgconv, x_twconv], dim = -1) + bias #B, N, dim_out
        return x_gwconv
