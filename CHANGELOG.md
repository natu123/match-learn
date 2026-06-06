# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/) (during `0.x`, minor releases may
contain breaking changes).

## [Unreleased]

### Added
- Optional `serde` feature deriving `Serialize`/`Deserialize` on the public data
  types (`Matching`, `ManyToOne`, `JointInstance`, `AuctionResult`, `Report`,
  `Demand`/`Supply`/`RoundOutcome`, `Objective`, and the application markets).
- Phase 7 dynamic pricing and Phase 8 applications: `Marketplace`,
  `LearnedPricer`, `JointInstance`, double/McAfee auctions, and ride-hailing /
  delivery / crowdsourcing adapters.

## [0.1.0] - 2026-06-06

First public release: the mechanism-proof core plus the dynamic-pricing
direction, built from scratch in Rust.

### Added
- **Stable matching**: Gale-Shapley deferred acceptance, Hospital-Residents
  (many-to-one with capacities), and Top Trading Cycles, each with a
  brute-force-verified stability/efficiency oracle.
- **Online preference learners**: Thompson Sampling, UCB1, a discounted variant
  for non-stationary preferences, and a linear contextual bandit. Posterior
  uncertainty and an exploration-scale knob are exposed.
- **Learning markets**: one-sided and two-sided unknown preferences on a shared
  `LearningMarket` trait, with a regret + stability evaluation harness. The
  Phase 1 gate proves sublinear regret and stabilization on random markets.
- **Dynamic pricing**: a supply-demand queue `Marketplace`, a `LearnedPricer`
  that learns the clearing price online, and `JointInstance` where a price gates
  participation and Gale-Shapley matches the entrants.
- **Performance & bindings**: dependency-free parallel batch evaluation
  (`simulate_batch`), optional PyO3 Python bindings (`python` feature), and a
  `wasm32` build target.
- **Benchmarks**: cross-language comparison against the `matching` library
  (identical matchings, ~520x faster) and MABWiser, plus an integrated NumPy
  reference.

[Unreleased]: https://github.com/natu123/match-learn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/natu123/match-learn/releases/tag/v0.1.0
