import dionysus as d
import math

def shift_filtration(rips,n):
    "Take a Dionysus filtration and increase the name of all of the vertices by n."
    f = d.Filtration()
    for s in rips:
        dim = s.dimension()
        temp = []
        for i in range(0,dim+1):
            temp.append(s[i]+n)
        f.append(d.Simplex(temp,s.data))
    return f

def complex_union(f,g):
    "Takes two filtrations and builds their union simplicial complex."
    union = d.Filtration()
    for s in f:
        union.append(s)
    for s in g:
        union.append(s)
    return union

def build_zigzag_times(rips,n,numbins):
    """ rips should be a union of angle bins as a Dionysus simplicial complex.
      n is the number of data points in each bin.
      numbins is the number of bins of data points, EXCLUDING unions.
      Returns times - the zig-zag birth and death times (list of lists length n)."""
    times = [[] for x in range(0,rips.__len__())]
    i=0
    for x in rips:
       dim = x.dimension()
       t = [];
       for k in range(0,dim+1):
          t.append(x[k])
       xmin = math.floor(min(t)/n)
       xmax = math.floor(max(t)/n)
       if xmax == 0:
          bd = [0,1]
       elif xmin == numbins-1:
          bd = [2*xmin-1,2*xmin]
       elif xmax == xmin:
          bd = [2*xmin-1,2*xmin+1]
       elif xmax > xmin:
          bd = [2*xmax-1,2*xmax-1]
       else:
          print("Something has gone horribly wrong!")
       times[i] = bd
       i = i+1
    return times

def compute_zigzag(f,times):
    zz, dgms, cells = d.zigzag_homology_persistence(f, times)
    return dgms
