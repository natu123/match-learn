"""Standard-library-only exact certificate construction, never repairs a candidate."""
from fractions import Fraction
from collections import deque
import math

from ._input import graph,matching,InputError

class Unverified(ValueError): pass

def construct(m,n,edges,pairs):
    try:
        m,n,edges=graph(m,n,edges)
        pairs=matching(m,n,pairs)
    except InputError as error:
        raise Unverified(str(error)) from error
    return _construct_snapshot(m,n,edges,pairs)

def _construct_snapshot(m,n,edges,pairs):
    if type(m) is not int or type(n) is not int or min(m,n)<0: raise Unverified('invalid_dimensions')
    costs={}; order=[]
    for i,j,w in edges:
        if type(i) is not int or type(j) is not int or not(0<=i<m and 0<=j<n): raise Unverified('invalid_coordinate')
        if (i,j) in costs: raise Unverified('duplicate_coordinate')
        if type(w) not in (int,float) or (type(w) is float and not math.isfinite(w)): raise Unverified('invalid_weight')
        costs[i,j]=Fraction(w); order.append((i,j))
    selected=set(); rows=set(); cols=set()
    for i,j in pairs:
        if type(i) is not int or type(j) is not int or (i,j) not in costs: raise Unverified('candidate_absent_edge')
        if i in rows or j in cols: raise Unverified('candidate_duplicate_endpoint')
        selected.add((i,j)); rows.add(i); cols.add(j)
    denominator=math.lcm(*(v.denominator for v in costs.values())) if costs else 1
    integers=[costs[e].numerator*(denominator//costs[e].denominator) for e in order]
    divisor=(math.gcd(*integers) if integers else 1) or 1
    z=[v//divisor for v in integers]; scale=Fraction(denominator,divisor)
    s=0; t=m+n+1; nodes=t+1
    original=[(s,1+i,0,i in rows) for i in range(m)]
    original.extend((1+i,1+m+j,c,(i,j) in selected) for (i,j),c in zip(order,z))
    original.extend((1+m+j,t,0,j in cols) for j in range(n))
    residual=[(b,a,-c) if used else (a,b,c) for a,b,c,used in original]
    adj=[[] for _ in range(nodes)]
    for a,b,c in residual: adj[a].append(b)
    reached={s}; queue=deque([s])
    while queue:
        a=queue.popleft()
        for b in adj[a]:
            if b not in reached: reached.add(b); queue.append(b)
    if t in reached: raise Unverified('not_maximum_cardinality: residual_s_t_path')
    capacity=sum(a in reached and b not in reached for a,b,c,used in original)
    if capacity!=len(selected): raise Unverified('cut_capacity_mismatch')
    # All-zero labels are equivalent to a zero-cost super-source to EVERY node.
    distances=[0]*nodes; predecessor=[None]*nodes
    for iteration in range(nodes):
        changed=None
        for a,b,c in residual:
            if distances[b]>distances[a]+c:
                distances[b]=distances[a]+c; predecessor[b]=(a,b,c); changed=b
        if changed is None: break
    else:
        x=changed
        for _ in range(nodes): x=predecessor[x][0]
        start=x; cycle=[]
        while True:
            arc=predecessor[x]; cycle.append(arc); x=arc[0]
            if x==start: break
        error=Unverified('not_minimum_cost: negative_residual_cycle')
        error.cycle=list(reversed(cycle)); error.cycle_cost=sum(c for a,b,c in cycle)
        error.reachable=sorted(reached)
        raise error
    if any(c+distances[a]-distances[b]<0 for a,b,c in residual): raise Unverified('potential_construction_failed')
    original_total=sum((costs[e] for e in selected),Fraction(0))
    return {'schema':'matching-cut-potentials-v1','cardinality':len(selected),
            'cost_numerator':original_total.numerator,'cost_denominator':original_total.denominator,
            'scale_numerator':scale.numerator,'scale_denominator':scale.denominator,
            'scaled_costs':z,'reachable_cut':sorted(reached),'cut_capacity':capacity,
            'potentials':distances,'bf_passes':iteration+1,'residual_arc_count':len(residual)}
