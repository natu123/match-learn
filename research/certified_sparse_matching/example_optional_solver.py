"""Optional installed NumPy/SciPy path: python -B example_optional_solver.py."""
from matching_prototype import solve_verified

def main():
    result=solve_verified(1,2,[(0,0,1),(0,1,0)])
    print('Candidate generation:',result['status'])
    if result['status']=='verified':
        print('Matching:',result['matching'],'exact cost:',result['exact_cost'])
    else:
        print(result['reason'])
if __name__=='__main__':main()
