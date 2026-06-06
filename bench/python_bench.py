"""Cross-language component benchmark: established Python libraries vs match-learn.

Compares, on identical workloads exported by `cargo run --release --example
bench_export`:

  1. Gale-Shapley stable matching: the `matching` library vs Rust. We verify the
     two produce the *same* matchings (so match-learn's from-scratch solver is
     correct against an established library), then compare speed.
  2. UCB1 online bandit: MABWiser vs Rust, per-round decision+update time.

Run inside the bench venv:

    ./bench-venv/bin/python bench/python_bench.py
"""

import time
from pathlib import Path

import numpy as np
from mabwiser.mab import MAB, LearningPolicy
from matching.games import StableMarriage

BENCH = Path(__file__).resolve().parent


def read_instances():
    lines = (BENCH / "instances.txt").read_text().splitlines()
    m, n = map(int, lines[0].split())
    pos = 1
    instances = []
    for _ in range(m):
        prop = [list(map(int, lines[pos + i].split())) for i in range(n)]
        pos += n
        recv = [list(map(int, lines[pos + i].split())) for i in range(n)]
        pos += n
        instances.append((prop, recv))
    return m, n, instances


def read_rust_matchings():
    lines = (BENCH / "rust_matching.txt").read_text().splitlines()
    gs_ms = float(lines[0])
    matchings = [list(map(int, ln.split())) for ln in lines[1:] if ln.strip()]
    return gs_ms, matchings


def solve_with_matching_lib(prop, recv, n):
    # The `matching` library needs distinct, hashable names across the two sides.
    suitor_prefs = {f"s{i}": [f"r{j}" for j in prop[i]] for i in range(n)}
    reviewer_prefs = {f"r{i}": [f"s{j}" for j in recv[i]] for i in range(n)}
    game = StableMarriage.create_from_dictionaries(suitor_prefs, reviewer_prefs)
    matching = game.solve()  # suitor (proposer) optimal
    # Return proposer -> receiver index.
    out = [-1] * n
    for suitor, reviewer in matching.items():
        i = int(str(suitor.name)[1:])
        j = int(str(reviewer.name)[1:])
        out[i] = j
    return out


def bench_matching():
    m, n, instances = read_instances()
    rust_ms, rust_matchings = read_rust_matchings()

    t0 = time.perf_counter()
    py_matchings = [solve_with_matching_lib(p, r, n) for (p, r) in instances]
    py_ms = (time.perf_counter() - t0) * 1000.0

    agree = sum(1 for a, b in zip(py_matchings, rust_matchings) if a == b)

    print("== Gale-Shapley stable matching ==")
    print(f"  instances           : {m} of size {n}x{n}")
    print(f"  matchings identical : {agree}/{m}")
    print(f"  Rust   (match-learn): {rust_ms:8.1f} ms")
    print(f"  Python (matching)   : {py_ms:8.1f} ms")
    print(f"  speedup             : {py_ms / rust_ms:8.1f}x")
    return agree == m


def bench_bandit():
    arms_n, rust_rounds, rust_ms = (
        (BENCH / "rust_bandit.txt").read_text().split()
    )
    arms_n, rust_rounds, rust_ms = int(arms_n), int(rust_rounds), float(rust_ms)
    rust_us_per_round = rust_ms * 1000.0 / rust_rounds

    arms = list(range(arms_n))
    true_means = [a / arms_n for a in arms]
    rng = np.random.default_rng(7)

    mab = MAB(arms, LearningPolicy.UCB1(alpha=0.5))
    mab.fit(arms, [float(rng.normal(true_means[a], 0.3)) for a in arms])

    py_rounds = 5000  # MABWiser's per-round refit is slow; measure per-round.
    t0 = time.perf_counter()
    for _ in range(py_rounds):
        arm = mab.predict()
        reward = float(rng.normal(true_means[arm], 0.3))
        mab.partial_fit([arm], [reward])
    py_us_per_round = (time.perf_counter() - t0) * 1e6 / py_rounds

    print("\n== UCB1 online bandit (per-round decision + update) ==")
    print(f"  arms                : {arms_n}")
    print(f"  Rust   (match-learn): {rust_us_per_round:10.3f} us/round  ({rust_rounds} rounds)")
    print(f"  Python (MABWiser)   : {py_us_per_round:10.3f} us/round  ({py_rounds} rounds)")
    print(f"  speedup             : {py_us_per_round / rust_us_per_round:10.1f}x")


if __name__ == "__main__":
    ok = bench_matching()
    bench_bandit()
    print()
    print("Matchings agree with the established library." if ok else "WARNING: matchings differ!")
