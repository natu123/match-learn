"""Focused stdlib tests; add --with-scipy for optional candidate regressions.
Both dependency-free and optional-SciPy modes passed packaging verification.
"""
import argparse,copy,itertools,json,os,pathlib,subprocess,sys,unittest
from fractions import Fraction
from unittest.mock import patch
import matching_prototype as api
from matching_prototype._input import graph,matching,certificate
import matching_prototype.checker as checker
ROOT=pathlib.Path(__file__).parent
WITH_SCIPY=False

def decode(edges):
    return [(i,j,int(w['value']) if w['type']=='int' else float.fromhex(w['hex'])) for i,j,w in edges]

def oracle(m,n,edges):
    weights={}
    for i,j,w in edges:
        num,den=(w,1) if type(w) is int else w.as_integer_ratio()
        weights[i,j]=Fraction(num,den)
    best=None
    for cols in itertools.product(range(-1,n),repeat=m):
        used=[j for j in cols if j>=0]
        if len(used)!=len(set(used)):continue
        pairs=[(i,j) for i,j in enumerate(cols) if j>=0]
        if any(pair not in weights for pair in pairs):continue
        key=(-len(pairs),sum((weights[p] for p in pairs),Fraction(0)))
        if best is None or key<best:best=key
    return -best[0],str(best[1])

class MatchingTests(unittest.TestCase):
    def test_recorded_fixtures_exactly(self):
        fixtures=json.loads((ROOT/'fixtures.json').read_text())
        verified=0
        for row in fixtures:
            with self.subTest(suite=row['suite'],name=row['name']):
                edges=decode(row['edges']);m,n=row['m'],row['n']
                expected=(row['expected']['cardinality'],row['expected']['exact_cost'])
                self.assertEqual(oracle(m,n,edges),expected)
                if row['recorded_candidate'] is not None:
                    cert=api.construct(m,n,edges,row['recorded_candidate'])
                    result=api.verify(m,n,edges,row['recorded_candidate'],cert)
                    self.assertEqual((result['cardinality'],result['exact_cost']),expected)
                    verified+=1
                if WITH_SCIPY:
                    result=api.solve_verified(m,n,edges)
                    self.assertEqual(result['status'],row['expected_status'])
                    if result['status']=='verified':self.assertEqual((len(result['matching']),result['exact_cost']),expected)
        self.assertEqual((len(fixtures),verified),(138,118))

    def test_wrong_candidates_and_tampered_certificates(self):
        data=json.loads((ROOT/'certificate_regressions.json').read_text())
        self.assertEqual((len(data['wrong_candidates']),len(data['certificate_tampers'])),(7,13))
        for row in data['wrong_candidates']:
            edges=[(i,j,int(w['int']) if 'int' in w else float.fromhex(w['float_hex'])) for i,j,w in row['edges']]
            with self.subTest(name=row['name']):
                with self.assertRaises(api.Unverified) as caught:api.construct(row['m'],row['n'],edges,row['candidate'])
                self.assertEqual(str(caught.exception),row['reason'])
                if hasattr(caught.exception,'cycle'):
                    err=caught.exception;self.assertLess(err.cycle_cost,0)
                    vertices={x for a,b,c in err.cycle for x in (a,b)}
                    if row['name'] in ('cost1_over0_through_t','naive_offset_wrong_candidate'):self.assertIn(row['m']+row['n']+1,vertices)
                    if row['name']=='cost1_over0_through_s':self.assertIn(0,vertices)
                    if row['name']=='unreachable_negative_cycle':self.assertFalse(vertices & set(err.reachable))
        for row in data['certificate_tampers']:
            tiny=row['name']=='rounded_problem_substitution'
            edges=[(0,0,2.**-54),(0,1,0.)] if tiny else [(0,0,1),(0,1,0)]
            with self.subTest(name=row['name']):
                with self.assertRaises(api.InvalidCertificate):api.verify(1,2,edges,[(0,0)] if tiny else [(0,1)],row['tampered_certificate'])

    def test_snapshots_and_mutation_negative_control(self):
        edges=[[0,0,1],[0,1,0]];pairs=[[0,1]];cert=api.construct(1,2,edges,pairs)
        frozen=graph(1,2,edges);fp=matching(1,2,pairs);fc=certificate(cert)
        edges[0][2]=-100;pairs[0][1]=0;cert['scaled_costs'][0]=99
        self.assertEqual(frozen,(1,2,((0,0,1),(0,1,0))))
        self.assertEqual(fp,((0,1),));self.assertEqual(fc['scaled_costs'],(1,0))
        with self.assertRaises(TypeError):fc['cardinality']=0
        live=[[0,0,1],[0,1,0]];ps=[[0,1]];cs=api.construct(1,2,live,ps)
        original=checker._verify_snapshot
        def mutate_after_snapshot(m,n,es,matching_snapshot,certificate_snapshot):
            live[0][2]=-50;ps[0][1]=0;cs['scaled_costs'][0]=99
            return original(m,n,es,matching_snapshot,certificate_snapshot)
        with patch.object(checker,'_verify_snapshot',mutate_after_snapshot):self.assertTrue(api.verify(1,2,live,ps,cs)['verified'])
        with self.assertRaises(api.InvalidCertificate):api.verify(1,2,live,ps,cs)

    def test_strict_types_and_bounds(self):
        hooks=[]
        class Stateful:
            def __iter__(self):hooks.append('iter');return iter([(0,0,0)])
        class ListSubclass(list):pass
        bad=[(1,1,Stateful()),(1,1,ListSubclass([(0,0,0)])),(1,1,[(0,0,True)]),(65,64,[]),(64,64,[(0,0,0)]*4097),(1,1,[(0,0,1<<2048)])]
        for m,n,edges in bad:self.assertEqual(api.certify_candidate(m,n,edges,[])['status'],'unverified')
        self.assertEqual(hooks,[])
        cert=api.construct(1,1,[(0,0,0)],[(0,0)]);cert['potentials'][0]=1<<4096
        with self.assertRaises(api.InvalidCertificate):api.verify(1,1,[(0,0,0)],[(0,0)],cert)

    def test_import_boundary(self):
        script=r'''
import os,sys,importlib.abc
before=dict(os.environ)
try:
 import resource
 limits={k:resource.getrlimit(getattr(resource,k)) for k in dir(resource) if k.startswith('RLIMIT_') and type(getattr(resource,k)) is int}
except ImportError:resource=None
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('numpy','scipy'):raise ImportError('optional dependency blocked')
sys.meta_path.insert(0,Block())
import matching_prototype
import matching_prototype.candidate,matching_prototype.checker,matching_prototype.certify,matching_prototype.api,matching_prototype._input
assert dict(os.environ)==before
if resource:assert {k:resource.getrlimit(getattr(resource,k)) for k in limits}==limits
assert matching_prototype.certify_candidate(1,2,[(0,0,1),(0,1,0)],[(0,1)])['status']=='verified'
assert matching_prototype.solve_verified(1,2,[(0,0,1),(0,1,0)])['status']=='unavailable'
'''
        subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,check=True,timeout=15)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--with-scipy',action='store_true');args=parser.parse_args()
    WITH_SCIPY=args.with_scipy
    unittest.main(argv=[sys.argv[0]])
