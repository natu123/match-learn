# match-learn

Online preference learning x stable matching, in Rust.

> **Status: Phase 1 (mechanism proof), built in public.** Early and incomplete.

## What

A from-scratch Rust library for **learning-to-match**: two-sided matching markets
where each side's preferences are unknown and learned online (Thompson Sampling /
UCB) while a stable matching (Gale-Shapley) is maintained. As preferences are
learned, the matching converges toward the stable optimum.

## Goal

- **Near-term**: a self-built core for online-preference-learning x stable matching.
- **Long-term**: dynamic pricing x supply-demand matching (ride-hailing / delivery /
  marketplace style), where price is treated as a proxy for preference.

## Why

Rust has Gale-Shapley (basic), bandits (trashpanda), and Bayesian inference (rustmc)
as separate pieces, but no library that **integrates online learning with stable
matching**. match-learn fills that gap: a self-built core, with performance-specialized
parts borrowed later.

## Roadmap

- [ ] **Phase 1 - Mechanism proof (v0)**: Gale-Shapley + online preference learning
      (Thompson Sampling) + integration loop + regret/stability harness.
      Gate: sublinear regret and the matching stabilizes on synthetic data.
- [ ] Phase 2 - Matching coverage (many-to-one, quotas, incomplete lists, TTC)
- [ ] Phase 3 - Learning layer (contextual / non-stationary / Bayesian preference)
- [ ] Phase 4 - Real data and benchmarks
- [ ] Phase 5 - Performance and bindings (Rayon, PyO3, WASM)
- [ ] Phase 6 - v1.0 stable release
- [ ] Phase 7 - Dynamic pricing x matching (queues, pricing policy)
- [ ] Phase 8 - Productionization (ride-hailing / delivery / crowdsourcing)

## License

MIT
