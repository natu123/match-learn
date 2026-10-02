"""Bounded prototype API; snapshots caller inputs once; no import-time settings."""
from ._input import graph,matching,certificate as snapshot_certificate,InputError
from .certify import _construct_snapshot,Unverified
from .checker import _verify_snapshot,InvalidCertificate

def certify_candidate(m,n,edges,pairs):
    """Stdlib-only certification of an externally supplied candidate, no solver.
    The graph limits apply; no candidate-generation guard is bypassed because no
    candidate is generated. Only a successful independently checked result returns.
    """
    try:
        m,n,edges=graph(m,n,edges)
        pairs=matching(m,n,pairs)
        cert=_construct_snapshot(m,n,edges,pairs)
        checked=_verify_snapshot(m,n,edges,pairs,snapshot_certificate(cert))
    except (InputError,Unverified,InvalidCertificate,ValueError,TypeError,OverflowError) as error:
        return {'status':'unverified','reason':str(error)}
    return {'status':'verified','matching':pairs,'exact_cost':checked['exact_cost'],'certificate':cert,'checker_result':checked}

def solve_verified(m,n,edges):
    """Optional SciPy candidate followed by exact certification. No repair."""
    from .candidate import generate_snapshot,Unsupported
    try:
        m,n,edges=graph(m,n,edges)
        pairs,metadata=generate_snapshot(m,n,edges)
        # Solver output is not trusted: matching legality is independently checked.
        cert=_construct_snapshot(m,n,edges,pairs)
        checked=_verify_snapshot(m,n,edges,pairs,snapshot_certificate(cert))
    except Unsupported as error:
        return {'status':'unsupported','reason':str(error)}
    except ImportError as error:
        return {'status':'unavailable','reason':'optional NumPy/SciPy dependency unavailable'}
    except (InputError,Unverified,InvalidCertificate,ValueError,TypeError,OverflowError) as error:
        return {'status':'unverified','reason':str(error)}
    return {'status':'verified','matching':pairs,'exact_cost':checked['exact_cost'],'certificate':cert,'checker_result':checked,'reduction':metadata}
