"""Independent stdlib linear graph-work checker. Does not import the producer.
Input: original graph, candidate matching, untrusted certificate. No Bellman-Ford.
Bit-operation costs of arbitrary-precision integers are additional.
"""
import math
from fractions import Fraction

from ._input import graph,matching,certificate as snapshot_certificate,InputError

class InvalidCertificate(ValueError): pass

def verify(m,n,edges,pairs,certificate):
    try:
        m,n,edges=graph(m,n,edges)
        pairs=matching(m,n,pairs)
        certificate=snapshot_certificate(certificate)
    except InputError as error:
        raise InvalidCertificate(str(error)) from error
    return _verify_snapshot(m,n,edges,pairs,certificate)

def require(condition,why):
    if not condition: raise InvalidCertificate(why)

def _verify_snapshot(m,n,edges,pairs,certificate):
    require(type(m) is int and type(n) is int and m>=0 and n>=0,'dimensions')
    require(certificate.get('schema')=='matching-cut-potentials-v1','schema')
    lookup={}; ratios=[]
    for index,edge in enumerate(edges):
        require(type(edge) in (tuple,list) and len(edge)==3,'edge_shape')
        i,j,w=edge
        require(type(i) is int and type(j) is int and 0<=i<m and 0<=j<n,'coordinate')
        require((i,j) not in lookup,'duplicate_coordinate')
        require(type(w) in (int,float),'weight_type')
        if type(w) is int: num,den=w,1
        else:
            require(math.isfinite(w),'nonfinite_weight')
            num,den=w.as_integer_ratio()
        lookup[i,j]=index; ratios.append((num,den))
    chosen=set(); rused=set(); cused=set()
    for pair in pairs:
        require(type(pair) in (tuple,list) and len(pair)==2,'candidate_shape')
        i,j=pair
        require(type(i) is int and type(j) is int and (i,j) in lookup,'candidate_edge')
        require(i not in rused and j not in cused,'candidate_endpoint')
        chosen.add((i,j)); rused.add(i); cused.add(j)
    for key in ('cardinality','cut_capacity','scale_numerator','scale_denominator','cost_numerator','cost_denominator'):
        require(type(certificate.get(key)) is int,'integer_field:'+key)
    require(certificate['cardinality']==len(chosen),'cardinality')
    sn=certificate['scale_numerator']; sd=certificate['scale_denominator']
    require(sn>0 and sd>0,'positive_scale')
    costs=certificate.get('scaled_costs')
    require(type(costs) is tuple and len(costs)==len(edges),'scaled_costs_length')
    # Independently bind exact scaled integer costs to original values, NOT offsets.
    for (num,den),z in zip(ratios,costs):
        require(type(z) is int and z*den*sd==num*sn,'original_cost_scale_identity')
    exact_sum=sum((Fraction(*ratios[lookup[p]]) for p in chosen),Fraction(0))
    require(certificate['cost_denominator']>0,'cost_denominator')
    require(Fraction(certificate['cost_numerator'],certificate['cost_denominator'])==exact_sum,'original_objective')
    vertex_count=m+n+2; source=0; sink=vertex_count-1
    potentials=certificate.get('potentials'); cut=certificate.get('reachable_cut')
    require(type(potentials) is tuple and len(potentials)==vertex_count and all(type(x) is int for x in potentials),'potentials')
    require(type(cut) is tuple and all(type(x) is int and 0<=x<vertex_count for x in cut),'cut_nodes')
    S=set(cut); require(len(S)==len(cut) and source in S and sink not in S,'cut_separation')
    adj=[[] for _ in range(vertex_count)]; crossing=0; arcs=0; min_reduced=None
    def arc(a,b,c,flow):
        nonlocal crossing,arcs,min_reduced
        crossing+=int(a in S and b not in S)
        tail,head,weight=(b,a,-c) if flow else (a,b,c)
        reduced=weight+potentials[tail]-potentials[head]
        require(reduced>=0,'negative_reduced_cost')
        min_reduced=reduced if min_reduced is None else min(min_reduced,reduced)
        adj[tail].append(head); arcs+=1
    # Rebuild all three arc families independently, including source and sink.
    for i in range(m): arc(source,i+1,0,i in rused)
    for index,(i,j,w) in enumerate(edges): arc(i+1,m+j+1,costs[index],(i,j) in chosen)
    for j in range(n): arc(m+j+1,sink,0,j in cused)
    require(crossing==len(chosen)==certificate['cut_capacity'],'cut_capacity')
    reached={source}; pending=[source]
    for a in pending:
        for b in adj[a]:
            if b not in reached: reached.add(b); pending.append(b)
    require(reached==S,'reachable_cut_mismatch')
    require(sink not in reached,'residual_augmenting_path')
    return {'verified':True,'cardinality':len(chosen),'exact_cost':str(exact_sum),'cut_capacity':crossing,'residual_arcs_checked':arcs,'minimum_reduced_cost':min_reduced}
