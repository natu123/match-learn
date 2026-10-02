Sparse matching with exact post-solve certificates
PUBLIC RESEARCH BRANCH EXAMPLE — NOT A RELEASE

This authored prototype targets minimum-cost MAXIMUM-cardinality bipartite
matching with explicit sparse edges. It uses a known sparse doubling reduction
for optional candidate generation, then independently checks an exact cut and
residual-potential certificate. It is wrapper/certification engineering, not a
new matching solver or algorithm-novelty claim.

The authored software and accompanying documentation are licensed under MIT.
Copyright (c) 2026 natu123. See LICENSE and ATTRIBUTION.txt for the license text
and third-party scope. This example is published on the research branch; it is
not merged into the default branch or distributed as a package release.

STANDALONE MODULE INTEGRATION
This directory is an independent Python research example in match-learn. It is
not integrated with the Rust crate, Cargo targets, PyO3/maturin bindings or root
Python package configuration. It does not change their APIs or dependencies.
From a checkout of research/certified-sparse-matching, first run:
  cd research/certified_sparse_matching
Then use the local commands below. No root build or installation is required.
The exact certificate API is dependency-free; only optional candidate generation
uses the separately pinned NumPy/SciPy dependencies.

DEPENDENCY-FREE CERTIFICATE PATH
No third-party packages are required to check an externally supplied candidate:
  python -B example_certificate.py
  python -B test_matching.py

The example certifies the zero-cost edge in a 1x2 graph and rejects its costlier
alternative. It does not generate a candidate or import NumPy/SciPy.

Public entrypoints in matching_prototype:
- certify_candidate(m,n,edges,pairs): verified result or unverified with no matching.
- construct(m,n,edges,pairs): certificate, or Unverified exception.
- verify(m,n,edges,pairs,certificate): independent check, or InvalidCertificate.
- solve_verified(m,n,edges): optional candidate plus exact check; statuses are
  verified, unsupported, unavailable, or unverified. Only verified includes a
  matching. There is no repair, reoptimization fallback or automatic installation.

OPTIONAL CANDIDATE-GENERATION / NUMERICAL REGRESSION PATH
Tested dependency versions are pinned in requirements-optional.txt:
  python -m venv .venv
  . .venv/bin/activate
  python -m pip install --only-binary=:all: -r requirements-optional.txt
  python -B example_optional_solver.py
  python -B test_matching.py --with-scipy

The commands above use POSIX shell syntax. Previously tested: CPython 3.12.14,
NumPy 2.3.5, SciPy 1.17.0, Linux. Windows/macOS operation and wheel availability
were not validated. The resource-limit portion of the import test runs only where
Python's resource module exists. Windows venv activation differs. If compatible
binary wheels are unavailable, stop rather than build SciPy from source.

The package changes no environment, thread or resource settings on import.
Optional dependencies are loaded only when candidate generation is requested.
For comparable Linux numerical runs, the caller can set OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 before starting Python. Caller-owned
wall/CPU limits are appropriate; the package does not install them.

INPUTS / LIMITS
- Ordinary built-in list/tuple containers and nested edge triples/pairs only;
  custom/stateful iterables, generators and subclasses are rejected.
- Built-in integer dimensions/indices only, m+n<=128, at most 4096 edges.
- Weights: built-in int, at most 2048 magnitude bits, or finite built-in float
  interpreted as its exact represented binary64 value, NOT intended decimal
  reals. bool, NumPy scalar, Fraction and Decimal inputs are unsupported.
- Duplicate coordinates, absent selected edges and repeated matching endpoints
  are rejected. An explicit zero or negative zero is an edge; absence is separate.
- Certificate: exact built-in dict with the v1 field set; integer fields <=4096
  magnitude bits, scaled-cost array <=4096, cut/potential arrays <=130.
- Graph/matching inputs are captured once into detached immutable tuples;
  certificate values into an immutable mapping with tuple arrays. Do not mutate
  inputs concurrently during initial capture. Later caller mutations do not
  change the captured problem. Returned certificate data must be rechecked if
  modified. Internal snapshot functions are not public API.

These are prototype bounds, not a production-grade DoS or completion guarantee.
Only returned independently verified results receive the mathematical guarantee.
The candidate's 2^45 numerical guard is heuristic, not a proof of solver-wide
floating-point exactness. Large-scale performance, formal verification and broad
platform support are not established.

CONTENTS / VALIDATION STATUS
methods_results.txt states methods, recorded results and scientific limits.
fixtures.json retains 138 fixed oriented tiny graphs with exact weight encodings,
expected objectives and recorded candidates. test_matching.py includes an
independent exhaustive oracle, certificate corruption, snapshot and import tests.
certificate_regressions.json preserves seven wrong-candidate constructor tests
and thirteen checker tamper tests. They are different rejection categories.

No new scientific cases or benchmarks were added. A clean extraction of the
packaged candidate passed manifest/privacy checks, its dependency-free example and all
five tests in both default and optional-SciPy modes. verification_results.json
contains historical pre-publication commands, output and resource measurements.
packaging_check.json is also historical packaging evidence, not a CI run on this
branch. Publication preparation reruns manifest/content checks only; it does not
claim new scientific results. This validates the existing
packaged fixtures; it does not broaden scientific claims or OS support. The
MIT license applies to the authored package. This branch publication does not
constitute a release. The package contains no upstream source,
third-party binaries, historical archives, credentials or private workflow log.

See references.json for public source URLs and the inspected SciPy source hash;
those reference files are not bundled. manifest.json hashes every payload file
other than itself. The records describe the validated candidate before branch publication; the
manifest reflects the current branch files after documentation updates.
