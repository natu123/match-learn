"""Guarded sparse doubling candidate generation; lazy optional dependencies."""
import math
from fractions import Fraction
class Unsupported(ValueError): pass

def generate_snapshot(m,n,edges):
    rationals=[Fraction(w) for i,j,w in edges]
    d=math.lcm(*(w.denominator for w in rationals)) if rationals else 1
    zs=[w.numerator*(d//w.denominator) for w in rationals]
    g=(math.gcd(*zs) if zs else 1) or 1
    zs=[z//g for z in zs]; A=max(map(abs,zs),default=0); B=2*min(m,n)*A+1
    if (m+n)*max(B+A,1)>2**45: raise Unsupported('exact integer encoding exceeds conservative 2^45 objective budget')
    # Never imported by the exact checker/constructor, or until solve is called.
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    rows=[];cols=[];data=[]
    for (i,j,w),z in zip(edges,zs):
        rows.extend((i,m+j));cols.extend((j,n+i));data.extend((z-B,z-B))
    for i in range(m): rows.append(i);cols.append(n+i);data.append(1)
    for j in range(n): rows.append(m+j);cols.append(j);data.append(1)
    mat=csr_matrix((np.array(data,dtype=float),(rows,cols)),shape=(m+n,m+n))
    r,c=min_weight_full_bipartite_matching(mat)
    pairs=tuple((int(i),int(j)) for i,j in zip(r,c) if i<m and j<n)
    return pairs,{'scale':str(Fraction(d,g)),'A':A,'B':B,'nnz':mat.nnz,'csr_bytes':mat.data.nbytes+mat.indices.nbytes+mat.indptr.nbytes}
