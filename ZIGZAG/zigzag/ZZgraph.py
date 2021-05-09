import dionysus as d
import numpy as np
import zigzagtools as zzt
from math import pi, cos, sin
from random import random
from random import choice
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import os

path = os.getcwd()

#%% Parameters 
nameFolderNet = path + 'File'
NVertices = 6 # Number of vertices
scaleParameter = 0.4 # Scale Parameter (Maximum) # the maximal edge weight #
maxDimHoles = 2 # Maximum Dimension of Holes (It means.. 0 and 1)
sizeWindow = 12 # Number of Graphs 

#%% Open all sets (point-cloud/Graphs)
print("Loading data...") # Beginning
Graphs = []
for i in range(0,sizeWindow):
    #edgesList = np.loadtxt(nameFolderNet+str(i+1)+".txt") # Load data
    edgesList = np.loadtxt(nameFolderNet+str(i)+".csv", delimiter=',') # Load data
    Graphs.append(edgesList)
print("  --- End Loading...") # Ending

#%% Plot Graph
GraphsNetX = []
plt.figure(num=None, figsize=(16, 1.5), dpi=80, facecolor='w', edgecolor='k')
for i in range(0,sizeWindow):
    g = nx.Graph()
    g.add_nodes_from(list(range(1,NVertices+1))) # Add vertices...
    if(Graphs[i].ndim==1 and len(Graphs[i])>0):
        g.add_edge(Graphs[i][0], Graphs[i][1], weight=Graphs[i][2])
    elif(Graphs[i].ndim==2):
        for k in range(0,Graphs[i].shape[0]):
             g.add_edge(Graphs[i][k,0], Graphs[i][k,1], weight=Graphs[i][k,2])
    GraphsNetX.append(g)
    plt.subplot(1, sizeWindow, i+1)
    plt.title(str(i))
    pos = nx.circular_layout(GraphsNetX[i])
    nx.draw(GraphsNetX[i], pos, node_size=15, edge_color='r') 
    #nx.draw_circular(GraphsNetX[i], node_size=15, edge_color='r') 
    labels = nx.get_edge_attributes(GraphsNetX[i], 'weight')
    for lab in labels:
        labels[lab] = round(labels[lab],2)
    nx.draw_networkx_edge_labels(GraphsNetX[i], pos, edge_labels=labels, font_size=5)

plt.savefig('IMGS/Graphs.pdf', bbox_inches='tight')

#%% Building unions and computing distance matrices 
print("Building unions and computing distance matrices...") # Beginning
GUnions = []
MDisGUnions = []
for i in range(0,sizeWindow-1):
    # --- To concatenate graphs
    unionAux = []
    # if(Graphs[i].ndim==1 and len(Graphs[i])==0): # Empty graph
    #     unionAux = Graphs[i+1]
    # elif(Graphs[i+1].ndim==1 and len(Graphs[i+1])==0): # Empty graph
    #     unionAux = Graphs[i]
    # else:
    #     unionAux = np.concatenate((Graphs[i],Graphs[i+1]),axis=0)
    # --- To build the distance matrix
    MDisAux = np.zeros((2*NVertices, 2*NVertices))
    A = nx.adjacency_matrix(GraphsNetX[i]).todense()
    B = nx.adjacency_matrix(GraphsNetX[i+1]).todense()
    
    # ----- Version Original (2)
    C = (A+B)/2
    A[A==0] = 1.0
    A[range(NVertices), range(NVertices)] = 0
    B[B==0] = 1.0
    B[range(NVertices), range(NVertices)] = 0
    MDisAux[0:NVertices,0:NVertices] = A
    C[C==0] = 1.0
    C[range(NVertices), range(NVertices)] = 0
    MDisAux[NVertices:(2*NVertices),NVertices:(2*NVertices)] = B
    MDisAux[0:NVertices,NVertices:(2*NVertices)] = C
    MDisAux[NVertices:(2*NVertices),0:NVertices] = C.transpose()
    
    # Distance in condensed form 
    pDisAux = squareform(MDisAux)
    
    # --- To save unions and distances
    GUnions.append(unionAux) # To save union
    MDisGUnions.append(pDisAux) # To save distance matrix
print("  --- End unions...") # Ending

#%% To perform Ripser computations
print("Computing Vietoris-Rips complexes...") # Beginning
GVRips = []
for i in range(0,sizeWindow-1):
    ripsAux = d.fill_rips(MDisGUnions[i], maxDimHoles, scaleParameter) 
    GVRips.append(ripsAux)
print("  --- End Vietoris-Rips computation") # Ending

#%% Shifting filtrations...
print("Shifting filtrations...") #Beginning
GVRips_shift = []
GVRips_shift.append(GVRips[0]) # Shift 0... original rips01 
for i in range(1,sizeWindow-1):
    shiftAux = zzt.shift_filtration(GVRips[i], NVertices*i)
    GVRips_shift.append(shiftAux)
print("  --- End shifting...") # Ending

#%% To Combine complexes
print("Combining complexes...") # Beginning
completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1]) 
for i in range(2,sizeWindow-1):
    completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[i]) 
print("  --- End combining") # Ending

#%% To compute the time intervals of simplices
print("Determining time intervals...") # Beginning
time_intervals = zzt.build_zigzag_times(completeGVRips, NVertices, sizeWindow)
print("  --- End time") # Beginning

#%% To compute Zigzag persistence
print("Computing Zigzag homology...") # Beginning
G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
print("  --- End Zigzag") # Beginning

#%% To show persistence intervals
print("Persistence intervals:")
print("++++++++++++++++++++++")
print(G_dgms)
for i, dgm in enumerate(G_dgms):
    print(i)
    for p in dgm:
        print(p)
print("++++++++++++++++++++++")
for i,dgm in enumerate(G_dgms):
    print("Dimension:", i)
    if(i<2):
        for p in dgm:
            print(p)
for i,dgm in enumerate(G_dgms):
    print("Dimension:", i)
    if(i<2):
        d.plot.plot_bars(G_dgms[i],show=True)

# %% Personalized plot
for i,dgm in enumerate(G_dgms):
    print("Dimension:", i)
    if(i<2):
        matBarcode = np.zeros((len(dgm), 2)) 
        k = 0
        for p in dgm:
            #print("( "+str(p.birth)+"  "+str(p.death)+" )") 
            matBarcode[k,0] = p.birth
            matBarcode[k,1] = p.death
            k = k + 1
        matBarcode = matBarcode/2   ## final PD ##
        print(matBarcode)
        for j in range(0,matBarcode.shape[0]):
            plt.plot(matBarcode[j], [j,j], 'b')
        #my_xticks = [0,1,2,3,4,5,6,7,8,9,10,11]
        #plt.xticks(x, my_xticks)
        plt.xticks(np.arange(12))
        plt.grid(axis='x', linestyle='-')
        plt.savefig('IMGS/BoxPlot'+str(i)+'.pdf', bbox_inches='tight')
        plt.show() 
        
#%%

#%%
for s in GVRips[0]:
    print(s)

# %%
