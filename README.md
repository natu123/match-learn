# match-learn

[![crates.io](https://img.shields.io/crates/v/match-learn.svg)](https://crates.io/crates/match-learn)
[![docs.rs](https://img.shields.io/docsrs/match-learn)](https://docs.rs/match-learn)
[![license](https://img.shields.io/crates/l/match-learn.svg)](LICENSE)

**Stable matching that learns.** Online preference learning x stable matching, in Rust.

```toml
[dependencies]
match-learn = "0.1"
```

> **Status: v0.1.0 on crates.io. Phases 1–5 done, Phase 7 (dynamic pricing) underway. Built in public.**
> The core is built from scratch, one phase at a time. Verified against an
> established library (identical matchings) and benchmarked across languages.
> Pricing now gates participation and a bandit learns the market-clearing price.

📖 [Tutorial](docs/TUTORIAL.md) · [API docs](https://docs.rs/match-learn) · [Benchmarks](bench/) · [Changelog](CHANGELOG.md)

---

## What it does

Two-sided matching markets where each side\'s preferences are **unknown and learned
online**, while a **stable matching is kept** at every step. As preferences are learned
(Thompson Sampling / UCB), the matching converges toward the stable optimum.

```rust
// API sketch (Phase 1)
let mut market = Market::new(agents, items);
loop {
    let matching = market.stable_match();    // Gale-Shapley on current beliefs
    let reward   = market.observe(&matching); // noisy preference signal
    market.update(reward);                    // online belief update
}
```

---

## From Python

The Rust core is also callable from Python (optional `python` feature, built
with [maturin]):

```python
import match_learn as ml

ml.gale_shapley(proposer_prefs, receiver_prefs)        # stable matching
market = ml.Market.thompson(util_p, receiver_prefs, noise=0.2, seed=42)
report = market.simulate(3000)                          # learn x match loop
report.total_regret(), report.tail_stable_fraction(600)
```

```bash
pip install maturin && maturin develop --release
python python/example.py
```

[maturin]: https://www.maturin.rs/

## Goal

- **Near-term**: a self-built core for **online preference learning x stable matching**.
- **Long-term**: **dynamic pricing x supply-demand matching** (ride-hailing / delivery /
  marketplace), where price is treated as a proxy for preference.

---

## Why

Rust has the pieces separately — Gale-Shapley (basic), bandits (`trashpanda`), Bayesian
inference (`rustmc`) — but **nothing that integrates online learning with stable
matching**. match-learn fills that gap: a self-built core, with performance-specialized
parts borrowed later (clean separation, no premature dependencies).

---

## Roadmap

### Phase 1 — Mechanism proof (v0)
Prove that "learn preferences online while keeping the match stable" actually converges.

- [x] Public repo + scaffold (build in public)
- [x] Gale-Shapley stable matching (from scratch)
- [x] Online preference learning (Thompson Sampling / UCB, from scratch)
- [x] Integration loop (learn -> match -> reward -> update)
- [x] Regret + stability evaluation harness
- [x] Convergence on synthetic data

**Gate**: sublinear regret, and the matching stabilizes. ✅ **passed** — see below.

#### Phase 1 results

Over 40 random 5×5 markets (well-specified Thompson Sampling, `tests/gate.rs`):

| metric | value | meaning |
| --- | --- | --- |
| mean `R(2T)/R(T)` | **1.04** | regret is sublinear (linear would be 2.0); worst market 1.74 |
| tail regret rate | **~0** | per-round regret collapses to zero after learning |
| tail stable fraction | **0.92** | the matching is stable in the true market on most late rounds |
| regret vs no-learning | **~125× lower** | learning massively beats a fixed, information-free policy |

UCB1 also learns and is sublinear on aggregate, but its `ln t` exploration bonus
keeps probing arms it has stopped pulling, leaving heavier regret tails — the
explore/exploit cost that Phase 3 will tune.

```text
cargo test --test gate -- --nocapture   # the gate
cargo run  --example converge           # watch regret flatten
```

### Phase 2 — Matching coverage
From the textbook 1:1 case to real matching shapes.

- [x] Many-to-one matching (capacity / quotas) — Hospital-Residents deferred acceptance
- [x] Incomplete preference lists, tie-breaking
- [x] One-sided vs two-sided unknown preferences — `TwoSidedMarket`, both sides learn
- [x] Top Trading Cycle (TTC) and other mechanisms

### Phase 3 — Learning layer
Make the learning predictive and adaptive.

- [x] Contextual bandit (context-aware preference learning) — `LinearThompson`
- [x] Non-stationary preferences (discounting) — `DiscountedThompson`
- [x] Bayesian preference estimation (posterior uncertainty) — mean / std / credible intervals
- [x] Explore / exploit tuning — `with_exploration(scale)`

### Phase 4 — Real data & benchmarks
From synthetic to real, and against the competition.

- [x] Dataset adapter + correlated market generator (`data` module, text format)
- [x] Benchmarks vs Python (MABWiser / `matching`) — GS identical + ~520×, UCB1 ~600×, integrated ~11× (see [`bench/`](bench/))
- [x] Cross-check against published regret bounds — empirical slope ≈ 0.49 (≈ √T), baseline 1.0
- [x] Visualization of matching and preference evolution — `export_csv` + `benchmark` examples

### Phase 5 — Performance & bindings
Make it the production layer ("research in Python, production in Rust").

- [x] Parallelism — `simulate_batch` over `std::thread` (dependency-free; Rayon could swap in)
- [x] PyO3 Python bindings — optional `python` feature, `import match_learn` (see [`python/`](python/))
- [x] Latency / throughput benchmarks — `benchmark` example
- [x] WASM target — compiles to `wasm32-unknown-unknown` (parallel falls back to sequential)

### Phase 6 — v1.0 stable release
Ship it and become the reference.

- [x] Publish to crates.io — [`match-learn` v0.1.0](https://crates.io/crates/match-learn)
- [ ] Documentation & tutorials
- [ ] Issue / PR workflow
- [ ] Write-up / paper draft

### Phase 7 — Dynamic pricing x matching
Add the price axis: from matching to market.

- [x] Queueing model — `Marketplace` (price-responsive Poisson arrivals, queues, clearing price)
- [x] Dynamic pricing policy — `LearnedPricer` learns the clearing price online (bandit over a price grid)
- [x] Joint pricing x matching optimization — `JointInstance`: price gates entry, Gale-Shapley matches entrants
- [x] Regret-queue tradeoff — `regret_queue` example quantifies exploration's regret vs queue-imbalance cost

### Phase 8 — Productionization
Real platforms.

- [x] Application adapters — `RideHailing`, `Delivery`, `Crowdsourcing` map onto `JointInstance` (proximity / effort / skill fit), with learned pricing
- [ ] Large-scale / production deployment
- [x] Market design extensions — double auction + truthful McAfee mechanism (`auction` module)
- [ ] Price-as-preference, deepened

---

## License

MIT
