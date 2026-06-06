# Stall research — handoff

Status of the `research/stall-avoidance` track, written so an **implementer** can
pick up the remaining builds and a **theorist** can pick up the open proofs. The
detailed scientific account is [`stall-anatomy.md`](stall-anatomy.md); the
frozen-arm theory is [`stall-avoidance.md`](stall-avoidance.md). This file is the
short, actionable index.

## 1. What we found (TL;DR)

Greedy Thompson Sampling in a learned stable matching occasionally locks into a
wrong outcome. "The stall" is **three failure modes, all rooted in near-ties**
(true preference gaps below the reward-noise floor `σ`), each with a different
cure:

| # | mode | frequency | mechanism | cure | status |
|---|------|-----------|-----------|------|--------|
| 1 | frozen-arm | rare | an unmatched arm's posterior freezes at an underestimate | **forcing** — `ForcedExploreThompson` (`ε_t=c/t` probe of least-sampled arm), `O(log T)` regret | **implemented + tested** |
| 2 | near-tie churn | dominant | Thompson re-samples near-equal arms forever; never settles | **annealing** — `with_anneal(tau)` cools sample temperature | **implemented + tested** |
| 3 | near-tie cascade | hard residue | an indifferent proposer's order is amplified by Gale-Shapley, hurting others | **coordination** — market-level tie-break maximizing belief welfare | **validated (POC), not yet a library type** |

Key honesty note: the first 40-market gate where forcing made every market
sublinear was **partly seed luck**. The 400-market study is the real picture:
forcing helps only the rare frozen mode; the dominant cure is annealing; the
cascade residue needs coordination.

## 2. What is implemented (pointers)

- `src/learner.rs` — `ForcedExploreThompson` with forcing (`c`) and annealing
  (`with_anneal(tau)`); `c=0, tau=inf` recovers plain Thompson. Unit tests incl.
  `annealing_suppresses_near_tie_churn`, `forced_exploration_keeps_probing_a_frozen_arm`.
- `src/market.rs` — `Market::with_forced_explore(...)`, `belief_means()` diagnostic.
- `src/two_sided.rs` — `TwoSidedMarket::new(...)` (pluggable learners).
- `tests/gate.rs` — frozen-arm gate (forced vs greedy vs UCB).
- Examples (all `cargo run --release --example <name>`): `stall_study`,
  `dissect_stall`, `neartie_analysis`, `anneal_study`, `two_sided_stall`,
  `coordinated_poc`.

All tests green; `cargo fmt`/`clippy` clean. Nothing here touches `master`/`dev`.

## 3. Implementation handoff (delegate these)

### 3a. Live `CoordinatedMarket` (highest value) — de-risked by `coordinated_poc.rs`

Build a market that applies coordinated near-tie tie-breaking **every round**,
not just post-hoc. Spec:

- **Per round**: read each proposer's belief means; form its ranking; partition
  into contiguous **near-tie groups** (adjacent means within `ε`, e.g. `ε≈0.05`).
- **Search**: over the Cartesian product of within-group orderings, run
  Gale-Shapley and pick the matching maximizing **total belief welfare**
  `Σ_p mean_p[partner(p)]` (no true utilities — oracle-free; this is what works in
  the POC). Reference implementation: `coordinated_match()` in `coordinated_poc.rs`.
- **Compose** with annealing + light forcing on the learners (modes 1–2).
- **API shape**: mirror `Market` (a `step()` returning `Matching`, implement
  `LearningMarket`), e.g. `Market::with_coordination(eps)` or a wrapper type.
- **Validate**: a broad-study confirmation that end-to-end (live-loop) stall rate
  drops on the cascade population, comparable to the 9/10 post-hoc coverage.
- **Caveat to solve**: the ordering search is exponential in the **largest
  near-tie group** size. Fine for small `n` / few ties; for large groups, cap the
  group size, sample orderings, or use a local-improvement search. Log any cap
  (no silent truncation).

### 3b. Anytime annealing (small)

`with_anneal(tau)` needs `tau ≈ horizon`. Add a horizon-free schedule (cool by
total pulls or a doubling schedule) so the cure works without knowing `T`.

## 4. Theory agenda — EXPLOIT (solidify; the main track)

1. **Identifiability-aware regret bound.** Formalize §4 of `stall-anatomy.md`: a
   bound `E[R_T] ≲ Σ_p f(Δ_p, σ, T)` where markets with `Δ_p ≪ σ` saturate at an
   **irreducible floor** (you need `≈ σ²/Δ²` pulls to order two arms `Δ` apart).
   This explains why no exploration schedule removes the near-tie modes.
2. **Forced-exploration bound, tighten.** `stall-avoidance.md` gives
   `O(log T)` for `c > 8nσ²/Δ²`. Tighten constants and state the coupled-market
   version under the stable-matching uniqueness assumption.
3. **Annealing convergence.** Prove that `sqrt(tau/(tau+t))` cooling drives the
   churn flip-rate to zero while preserving best-arm identification, and
   characterize the cooling-too-fast lock-in trade-off (when does a fast schedule
   commit to a wrong stable matching?).
4. **Coordination optimality.** Show the belief-welfare tie-break recovers the
   true proposer-optimal stable matching whenever every misordered pair is within
   the indifference band `ε` and all other beliefs are `ε`-accurate.

## 5. New-theory seeds — EXPLORE (sub-track)

- **ε-stability benchmark.** Regret-vs-unique-stable over-charges indifferent
  swaps. Define regret against the *set* of ε-stable matchings; conjecture the
  near-tie modes vanish under it. Could reframe the whole problem.
- **Welfare vs stability under learning.** The coordinator optimizes welfare via
  free tie-breaks; when does that conflict with strategy-proofness / stability?
- **Receiver-informed tie-breaking** (partial result, `examples/receiver_informed.rs`).
  A search-free `O(n log n)` rule — an indifferent proposer takes the receiver that
  prefers it *least* — fixes **5/10** cascades (mean cascade regret `0.359 → 0.129`),
  vs **9/10** for the exponential belief-welfare search. So the known receiver
  preferences alone carry about half the coordination signal; closing the gap to
  the full coordinator cheaply is open.
- **Coupled-exploration lower bound.** Is there an instance-dependent lower bound
  showing the cascade externality is unavoidable for *any* decentralized
  (per-agent) policy, making a coordinator provably necessary?

## 6. Repo / process

- Branch `research/stall-avoidance` (worktree `~/match-learn-research`), pushed to
  `origin`. Do **not** push to `master`/`dev` (other sessions own those).
- Merge decision is the maintainer's; this branch rebases onto `master`'s
  crates.io work when merged (it forked at `ae18cb1`).
