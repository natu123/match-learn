# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/) (during `0.x`, minor releases may
contain breaking changes).

## [Unreleased]

### Added
- `strategyproof` module: brute-force manipulation checks (`proposer_manipulation`,
  `receiver_manipulation`) — verifies Gale-Shapley is strategy-proof for proposers
  but manipulable by receivers.
- `assignment` module: the assignment problem (welfare-optimal matching) via a
  from-scratch O(n^3) Hungarian algorithm — `min_cost_assignment` and
  `max_weight_assignment`, verified against brute force. The welfare-optimal
  counterpart to stable matching.
- `CoordinatedMarket` (experimental) — a live near-tie coordinator that each
  round searches within-near-tie orderings and picks the Gale-Shapley matching
  maximizing belief welfare, with vanishing forced exploration for frozen arms.
  Implements `LearningMarket`; the search is capped (`max_group` + a
  total-combination limit). Public `near_tie_rankings` helper. **Honest caveat:**
  validation (`examples/coordinated_validation.rs`) shows the live coordinator
  does not yet beat plain Thompson on stability — it maximizes belief welfare, so
  it raises proposer welfare but lowers the is-stable fraction. The post-hoc
  cascade cure does not transfer naively to the live loop; a confidence-gated or
  stability-targeting coordinator is open work.

## [0.1.1] - 2026-06-06

### Added
- `ForcedExploreThompson` — Thompson Sampling with vanishing forced exploration
  (`eps_t = min(1, c/t)`) and optional annealing, which beats greedy Thompson's
  frozen-arm matching *stall*; `Market::with_forced_explore` builds a market
  with it. Backed by a research-track analysis in `docs/` (`stall-anatomy.md`,
  `stall-avoidance.md`, `theory-identifiability.md`) that proves near-tie stalls
  resist exploration and yield to coordination.
- `reserves` module: deferred acceptance with diversity reserves
  (minority-reserve choice functions, Hafalir-Yenmez-Yildirim style) for
  distributional constraints like school-choice or residency reserves.
- `fairness` module: rank-cost metrics and egalitarian / sex-equal stable
  matchings (correcting Gale-Shapley's proposer-optimal one-sidedness), plus a
  public `all_stable_matchings` enumerator.
- `online` module: dynamic matching where agents arrive and depart over time
  (`OnlineMarket`, `Policy`), with the greedy-vs-batched timing tradeoff between
  match quality and abandonment. A bandit can learn the net-value-maximizing
  batch interval online when the arrival/abandonment regime is unknown.
- `Report` now records per-round `welfare` (realized proposer-side utility), with
  `tail_mean_welfare`.
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

[Unreleased]: https://github.com/natu123/match-learn/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/natu123/match-learn/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/natu123/match-learn/releases/tag/v0.1.0
