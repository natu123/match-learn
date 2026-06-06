# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/) (during `0.x`, minor releases may
contain breaking changes).

## [Unreleased]

### Added
- `contracts` module: matching with contracts via the cumulative offer process
  (Hatfield-Milgrom). A `Contract` bundles a doctor, a hospital, and `terms`
  (a wage band, a position, a length of service); doctors rank whole contracts
  and hospitals choose *sets* through a substitutable, capacity-limited
  responsive choice function, so the cumulative offer process yields a stable
  allocation (`is_stable_with_contracts`). With one contract per doctor-hospital
  pair it reduces to Hospital-Residents (verified). This is the framework behind
  cadet-branch matching and labor markets with wages; verified stable, against a
  brute-force oracle, and with a cadet-branch terms-competition example.
- `StabilityCoordinatedMarket` — a live near-tie coordinator that fixes the
  *objective* behind the negative finding rather than gating it. The research
  track's controlled A/B (`docs/theory-identifiability.md` §4a) showed that
  maximizing belief welfare is unstable even with accurate beliefs
  (`welfare-max ≠ stable-max`), so the Prop-4 gate can only *bound* the damage
  to `2·eps`-stability. This coordinator instead minimizes the *expected number
  of blocking pairs* over the near-tie orderings (Thompson-sampled profiles),
  targeting stability directly: it has no `2·eps` ceiling and reaches the highest
  tail-stability of all (≈0.96, above plain Thompson's ≈0.92) at the cost of some
  proposer welfare. Validated in `examples/coordinated_validation.rs`; the
  private `count_blocking_pairs` is cross-checked against `is_stable`.
- `lattice` module: the lattice of stable matchings and median stable matchings.
  Conway's lattice operations (`stable_join` / `stable_meet`: each proposer keeps
  the better / worse of its two partners, again a stable matching) and the
  Teo-Sethuraman `generalized_medians` — for each proposer its `i`-th-ranked
  partner across all stable matchings, each itself stable — with
  `median_stable_matching` as the balanced compromise between the proposer- and
  receiver-optimal extremes (a fairness counterpart to `fairness`'s egalitarian
  matchings). Stated for the classic marriage model (complete strict
  preferences); every median is verified stable against the brute-force set.
- `kidney` module: kidney exchange for incompatible patient-donor pairs, the
  market-design problem that has given tens of thousands of patients a transplant.
  Models the pool as a housing market (each patient is endowed with its own
  incompatible donor, ranked last as the no-exchange option) and clears it by
  Top Trading Cycles, so the outcome is individually rational, Pareto efficient,
  and strategy-proof (Roth-Sönmez-Ünver 2004). `kidney_exchange` takes ABO
  blood-typed `Pair`s (`abo_compatible`), `ttc_kidney_exchange` takes explicit
  compatibility lists; both find multi-way cycles, not just pairwise swaps.
  Verified for validity, individual rationality, Pareto efficiency (brute force),
  and strategy-proofness. Altruistic-donor *chains* (w-chains) are a noted
  extension, not yet implemented.
- `boston` module: the Boston (immediate-acceptance) school-choice mechanism — each
  round students apply to their next choice and schools admit by priority up to
  remaining capacity *permanently* (no deferral). Demonstrates, against the
  deferred-acceptance core, why Boston is unstable and manipulable yet
  Pareto-efficient for truthful students (all three verified in tests).
- `GatedCoordinatedMarket` — the Prop-4 confidence-gated cure for the cascade
  stall (research-track theorem, `docs/theory-identifiability.md`). It coordinates
  a near-tie only after the pair's posterior is certified tight
  (`|Δmean| + z·√(s_a²+s_b²) ≤ ε`), so it never reorders an un-converged pair.
  This resolves `CoordinatedMarket`'s negative finding: with a tight band it
  recovers nearly all the lost tail-stability (~0.91 vs plain Thompson's 0.92) at
  slightly better welfare, and `ε` tunes a bounded welfare/stability tradeoff
  (`examples/coordinated_validation.rs`). `PreferenceLearner` now exposes per-arm
  posterior `stds` (default `+inf` = never certified) to support gating.
- `ties` module: stable matching with indifferences. Weak / strong / super
  stability checkers (`is_weakly_stable`, `is_strongly_stable`,
  `is_super_stable`), a `weakly_stable` constructor (tie-break + Gale-Shapley,
  always exists), and brute-force `super_stable` / `strongly_stable` finders.
  Without ties all three collapse to ordinary stability; super ⟹ strong ⟹ weak.
- `allocation` module: one-sided house allocation (no endowments), the companion
  to `ttc`. `serial_dictatorship` (priority order), `random_serial_dictatorship`
  (random priority, fractional), and `probabilistic_serial` — the
  Bogomolnaia-Moulin simultaneous-eating algorithm whose fractional assignment is
  ordinally efficient and envy-free (`sd_envy_free`). `is_pareto_efficient` checks
  a discrete assignment via free-object and trading-cycle improvements, validated
  against a brute-force oracle.
- `many_to_many` module: stable matching where *both* sides have quotas (workers
  hold several firms, firms hold several workers) — worker-proposing deferred
  acceptance with responsive preferences, yielding a pairwise-stable matching
  (`is_pairwise_stable`), verified against a brute-force oracle. Reduces to
  Gale-Shapley at quota 1 and to Hospital-Residents when only one side has a quota.
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
