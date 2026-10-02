"""Strict shape/type snapshots only; no cost conversion or graph algorithms."""
import math
from types import MappingProxyType
MAX_VERTICES=128  # Original partitions combined; residual network adds s,t.
MAX_EDGES=4096
MAX_INPUT_INT_BITS=2048
MAX_CERT_INT_BITS=4096
class InputError(ValueError): pass

def require(ok,reason):
    if not ok: raise InputError(reason)

def graph(m,n,edges):
    require(type(m) is int and type(n) is int and 0<=m<=MAX_VERTICES and 0<=n<=MAX_VERTICES and m+n<=MAX_VERTICES,'dimension_limit_or_type')
    require(type(edges) in (list,tuple),'edges_must_be_ordinary_list_or_tuple')
    require(len(edges)<=MAX_EDGES,'edge_limit')
    copied=[]; seen=set()
    for edge in edges:
        require(type(edge) in (list,tuple) and len(edge)==3,'edge_record_type_or_length')
        i,j,w=edge
        require(type(i) is int and type(j) is int and 0<=i<m and 0<=j<n,'coordinate')
        require((i,j) not in seen,'duplicate_coordinate')
        require(type(w) in (int,float),'weight_type')
        require((abs(w).bit_length()<=MAX_INPUT_INT_BITS) if type(w) is int else math.isfinite(w),'weight_bits_or_nonfinite')
        seen.add((i,j)); copied.append((i,j,w))
    return m,n,tuple(copied)

def matching(m,n,pairs):
    require(type(pairs) in (list,tuple),'pairs_must_be_ordinary_list_or_tuple')
    require(len(pairs)<=MAX_VERTICES,'candidate_length_limit')
    copied=[]
    for pair in pairs:
        require(type(pair) in (list,tuple) and len(pair)==2,'pair_record_type_or_length')
        i,j=pair
        require(type(i) is int and type(j) is int and 0<=i<m and 0<=j<n,'candidate_coordinate')
        copied.append((i,j))
    return tuple(copied)

def certificate(cert):
    require(type(cert) is dict,'certificate_must_be_ordinary_dict')
    scalar={'cardinality','cut_capacity','scale_numerator','scale_denominator','cost_numerator','cost_denominator','bf_passes','residual_arc_count'}
    arrays={'scaled_costs':MAX_EDGES,'reachable_cut':MAX_VERTICES+2,'potentials':MAX_VERTICES+2}
    required=scalar|set(arrays)|{'schema'}
    require(len(cert)==len(required) and all(type(k) is str for k in cert) and set(cert)==required,'certificate_fields')
    require(type(cert['schema']) is str and cert['schema']=='matching-cut-potentials-v1','certificate_schema')
    copied={'schema':cert['schema']}
    for key in scalar:
        value=cert[key]
        require(type(value) is int and abs(value).bit_length()<=MAX_CERT_INT_BITS,'certificate_integer_limit_or_type:'+key)
        copied[key]=value
    for key,limit in arrays.items():
        value=cert[key]
        require(type(value) in (list,tuple) and len(value)<=limit,'certificate_array_limit_or_type:'+key)
        require(all(type(x) is int and abs(x).bit_length()<=MAX_CERT_INT_BITS for x in value),'certificate_array_integer_limit:'+key)
        copied[key]=tuple(value)
    return MappingProxyType(copied)
