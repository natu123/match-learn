# Contributing to match-learn

Thanks for your interest. match-learn is built in public, one phase at a time
(see the Roadmap in the README). Issues and PRs are welcome — for anything
larger than a fix, please open an issue to discuss the design first.

## Development

Rust **stable**, edition 2024.

```bash
cargo test --all-targets        # unit + integration tests
cargo test --doc                # doctests
cargo clippy --all-targets -- -D warnings
cargo fmt --all                 # (CI runs --check)
cargo build --target wasm32-unknown-unknown --lib   # wasm builds too
```

Python bindings (optional `python` feature, built with [maturin]):

```bash
pip install maturin
maturin develop --release
python python/example.py
```

Cross-language benchmarks live in [`bench/`](bench/).

[maturin]: https://www.maturin.rs/

## Definition of done

1. Implementation, with tests (green) verifying behaviour on synthetic or real
   data. Gates (e.g. the Phase 1 sublinear-regret gate) must pass on their
   intended bars — don't loosen a gate to make it pass.
2. One logical change per commit, [Conventional Commits](https://www.conventionalcommits.org/).
3. `clippy -D warnings` and `fmt` clean; doctests pass. CI enforces all of this.

## Scope and philosophy

- The **v0 core** (matching + online learning) is built from scratch, with no
  heavy external dependencies — the learning value and the integration are the
  point.
- **Performance and bindings** (parallelism, PyO3, WASM) may add optional,
  feature-gated dependencies.
- Prefer composition: the bandit learners drive both the matching markets and
  the pricing policy. New learners implement `PreferenceLearner` (`Send + Sync`)
  and slot into every market.

## Versioning

Semantic Versioning. During `0.x`, minor releases may break the API; record
user-facing changes in `CHANGELOG.md`.
