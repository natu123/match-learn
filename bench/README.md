# Cross-language benchmark

`match-learn` vs established Python libraries, on identical workloads. Two
levels:

- **Component** (`python_bench.py`): the matching core vs the [`matching`]
  library, and the bandit core vs [MABWiser]. The matching comparison also
  **verifies the two produce the same matchings**, cross-checking match-learn's
  from-scratch Gale-Shapley against an established implementation.
- **Integrated** (`integrated_bench.py`): the *whole* learn → match → reward →
  update loop. No Python library does this — it is the gap match-learn fills — so
  the reference is a clean, idiomatic NumPy implementation. We check it reaches
  the same sublinear behaviour, then compare throughput.

[`matching`]: https://pypi.org/project/matching/
[MABWiser]: https://pypi.org/project/mabwiser/

## Running

```bash
# one-time: create the venv (needs python3-venv)
python3 -m venv bench-venv
./bench-venv/bin/pip install mabwiser matching

# export instances + Rust timings, then run the Python side
cargo run --release --example bench_export
./bench-venv/bin/python bench/python_bench.py
./bench-venv/bin/python bench/integrated_bench.py
```

## Results

Measured on WSL2 Ubuntu (Rust 1.96, release build). Numbers vary by machine; the
ratios are the point.

### Component

| workload | Rust (match-learn) | Python | speedup |
| --- | --- | --- | --- |
| Gale-Shapley, 2000 × 20×20 instances | 5.7 ms | 2944 ms (`matching`) | **~520×** |
| UCB1 online bandit, per round | 0.14 µs | 85 µs (MABWiser) | **~600×** |

Matchings were **identical on all 2000 instances**, so the from-scratch solver
agrees with the established library.

### Integrated (learn × match loop)

| workload | Rust (match-learn) | Python (NumPy reference) | speedup |
| --- | --- | --- | --- |
| 8×8 market, 4000 rounds | 14 ms (tail stable 0.96) | 158 ms (tail stable 0.99) | **~11×** |

Both implementations reach sublinear regret (`R(2T)/R(T) ≈ 1.0`) and a stable
tail, so the integrated systems **agree in behaviour**. The speedup is smaller
than for the components because NumPy vectorizes the posterior sampling — an
honest result. It is also single-threaded; `simulate_batch` parallelizes across
markets on top of this.

## Notes

- MABWiser is batch-oriented; a tight online loop is not its design point, so its
  per-round number reflects that. We compare it anyway because UCB1-per-round is
  the honest analogue of what the market loop does.
- The integrated reference is intentionally plain NumPy (not MABWiser-in-a-loop),
  to be a *fair* baseline rather than an artificially slow one.
- `bench-venv/` and `bench/*.txt` are generated and git-ignored.
