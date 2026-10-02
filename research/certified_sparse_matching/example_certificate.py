"""Dependency-free certificate example: python -B example_certificate.py."""
from matching_prototype import certify_candidate,verify

def main():
    # Both entries are present edges, including the zero-cost edge.
    edges=[[0,0,1],[0,1,0]]
    result=certify_candidate(1,2,edges,[(0,1)])
    assert result['status']=='verified' and result['exact_cost']=='0'
    checked=verify(1,2,edges,result['matching'],result['certificate'])
    assert checked['verified']
    print('Verified cardinality:',checked['cardinality'],'exact cost:',checked['exact_cost'])
    bad=certify_candidate(1,2,edges,[(0,0)])
    assert bad['status']=='unverified' and 'matching' not in bad
    print('Costlier candidate rejected:',bad['reason'])
if __name__=='__main__':main()
