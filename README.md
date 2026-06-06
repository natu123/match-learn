# match-learn

**Stable matching that learns.** Online preference learning x stable matching, in Rust.

> **Status: Phases 1–4 done + Phase 5 in progress, built in public.**
> The core is built from scratch, one phase at a time. Verified against an
> established library (identical matchings) and benchmarked across languages.

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
- [ ] PyO3 Python bindings
- [x] Latency / throughput benchmarks — `benchmark` example
- [ ] WASM target

### Phase 6 — v1.0 stable release
Ship it and become the reference.

- [ ] Publish to crates.io
- [ ] Documentation & tutorials
- [ ] Issue / PR workflow
- [ ] Write-up / paper draft

### Phase 7 — Dynamic pricing x matching
Add the price axis: from matching to market.

- [ ] Queueing model
- [ ] Dynamic pricing policy
- [ ] Joint pricing x matching optimization
- [ ] Regret-queue tradeoff

### Phase 8 — Productionization
Real platforms.

- [ ] Application adapters (ride-hailing / delivery / crowdsourcing)
- [ ] Large-scale / production deployment
- [ ] Market design extensions (auctions, mechanisms)
- [ ] Price-as-preference, deepened

---

## License

MIT
