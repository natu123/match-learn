# The anatomy of stalls in learned stable matching

*Research-track note. Companion to `examples/stall_study.rs`,
`examples/dissect_stall.rs`, `examples/neartie_analysis.rs`,
`examples/anneal_study.rs`, and the learner `ForcedExploreThompson` in
`src/learner.rs`. This note supersedes the narrower claims of
[`stall-avoidance.md`](stall-avoidance.md), which analyzes only the frozen-arm
mode.*

## Summary

We set out to kill one bug — greedy Thompson Sampling occasionally locking a
learned matching into a wrong stable outcome — and found that "the stall" is
really **three failure modes with three different cures**, all ultimately caused
by **near-ties in the true preferences relative to the reward noise**.

1. **Frozen-arm stall** (rare): an arm stops being matched, its posterior
   freezes at an underestimate, and the agent never revisits its true partner.
   *Cure: more exploration* — a vanishing forced-exploration schedule
   (`ForcedExploreThompson`, `O(log T)` regret, §3).
2. **Near-tie churn** (dominant): two receivers a proposer values within the
   noise floor cannot be ordered, so Thompson keeps **re-sampling** them forever
   and the matching never settles. *Cure: less tail exploration* — annealing the
   sampling temperature (`with_anneal`, §4). Forcing makes this mode **worse**.
3. **Near-tie cascade** (the hard residue): a near-indifferent proposer's
   arbitrary order is **amplified by Gale-Shapley** into a far matching that costs
   *other* proposers large regret. No per-agent policy fixes this. *Candidate
   cure:* coordination — a market-level tie-break among near-equal arms that
   maximizes total belief welfare (§4.2). Fixes 9/10 cascades **post-hoc on
   converged beliefs**, but ⚠ **does not transfer to the live loop** (with
   mid-learning beliefs, welfare-max ≠ stable, so it *loses* stability — see §4.2).
   A live cure is an open problem.

An honest correction to our first result: the clean 40-market gate
(`tests/gate.rs`) where forced exploration took every market sublinear was partly
**seed luck**. On 400 markets per size the picture is the nuanced one above.

**Sharpening — the stalls are stable, just not proposer-optimal**
(`examples/eps_stability.rs`). The settled "stalled" matchings are *not* unstable:
7/10 are **exactly stable** (no blocking pair at all) and 10/10 are **ε-stable**
(`ε=0.05`). A near-tie stall converges to a *different* stable matching, not the
proposer-**optimal** one. So the regret-vs-proposer-optimal is an **optimality
gap**, not instability — against an any-stable or ε-stable benchmark the stall
vanishes. The three cures below recover proposer-optimality *among* stable
matchings; they do not rescue stability (which is already attained). At scale
(`examples/stability_vs_optimality.rs`, 800 markets) the settled matching is
**ε-stable in 100%** of markets and **exactly stable in 98.1% (5×5) / 93.4%
(8×8)**; the optimality gap is non-trivial on only **1.2% / 4.1%** of markets.

## 1. Setup

One-sided learned matching (`src/market.rs`): `n` proposers each treat the `n`
receivers as a Gaussian bandit (`reward ~ N(μ_{p,r}, σ²)`, `σ = 0.2`), receivers
have fixed known preferences. Each round: proposers submit belief-rankings,
Gale-Shapley matches against receiver preferences, matched proposers observe a
noisy reward and update. Regret is measured against the proposer-optimal stable
matching of the *true* market.

## 2. The phenomenon is real, and has two populations

`examples/stall_study.rs`, 400 random markets per size, horizon `2T = 1500`. A
market "stalls" if its tail mean regret rate stays above `0.01`.

| size | Thompson stall rate | UCB1 stall rate |
| --- | :---: | :---: |
| 3×3 | 9.5% | 15.8% |
| 5×5 | 18.8% | 33.8% |
| 8×8 | 37.0% | 59.0% |
| 12×12 | 49.8% | 67.3% |

Stalls are **not rare** at this horizon, and grow with `n`. But varying the
horizon splits them in two (Thompson, 5×5): the stall rate falls 18.8% → 10.8% →
6.8% as the horizon grows 1500 → 6000 → 24000, while the **worst-case tail regret
stays ≈ 0.80 throughout**. So most "stalls" are **slow convergence** that more
rounds fix; a residual **genuine core** (~7% at 5×5, ~12% at 8×8) never resolves.

## 3. Frozen-arm stall, and forced exploration

The original hypothesis. An arm pulled out of the match freezes; if it is the
true partner and underestimated, the agent never returns. `ForcedExploreThompson`
adds, with probability `ε_t = min(1, c/t)`, a forced pull of the least-sampled
arm. The cumulative forced rounds grow like `c·ln T`, guaranteeing every arm —
including a frozen one — is probed `Ω(log T)` times, while the rate vanishes so
the tail stays calm (unlike UCB's perpetual bonus).

The regret theory (full version in [`stall-avoidance.md`](stall-avoidance.md)):
forced rounds cost `O(Δ_max·c·log T)`; misranking probability decays as
`t^{-cΔ²/(8nσ²)}`, giving **`O(log T)` regret when `c > 8nσ²/Δ²`** and sublinear
for any `c > 0`, with stall probability → 0. This is the right cure **for the
frozen-arm mode**.

But on random markets it under-delivers, because the frozen-arm mode is rare.

## 4. The dominant mode is near-ties, not frozen arms

`examples/dissect_stall.rs` hunts the worst 4×4 lock-in and takes it apart. In the
worst genuine lock-in, **every proposer estimates its true partner accurately** —
nothing is frozen. The cause is a near-tie: one proposer's true utilities for two
receivers differ by `0.001`, far below `σ = 0.2`. It cannot order them, and
Gale-Shapley's discontinuity turns that hair-width error into a far matching that
costs *another* proposer `0.84` regret. (A proposer indifferent at `0.001`
externalizes large regret onto others.)

`examples/neartie_analysis.rs` confirms this across 800 markets at horizon 24000:

- **Every** stalled market has a tightest true-preference gap below the noise
  floor (median `0.0022` vs `0.0067` for settled markets).
- Splitting the stalled set by symptom:
  - **~1/3 cascade**: settle into a *wrong* matching whose displaced proposer is
    near-indifferent (own-loss median `0.0042`, 6/7 below `0.05`).
  - **~2/3 churn**: reach the *correct* belief-mean matching yet still pay tail
    regret, because Thompson keeps re-sampling near-equal arms in different
    orders forever.

Both are the same root cause: **a true gap below the noise floor**.

### Why forcing cannot fix this, and annealing can

More exploration cannot separate two means that differ by `Δ ≪ σ`: that needs
`N ≳ σ²/Δ²` pulls (for `Δ = 0.001, σ = 0.2`, about `40,000` pulls — beyond the
horizon). The beliefs are already correct; the problem is the *policy* keeps
acting on un-separable noise. Forcing adds *more* perturbation, so it worsens the
churn.

Annealing does the opposite. `with_anneal(tau)` scales the Thompson sample's std
by `sqrt(tau/(tau+t))`, cooling from full exploration toward posterior-mean
exploitation. Once cooled, the learner stops flipping near-tie arms and the
matching settles on the (correct) belief-mean matching.

`examples/anneal_study.rs`, 400 markets, horizon 24000, genuine-stall threshold
`0.05`:

| learner (5×5) | stall rate | tail p99 | mean regret |
| --- | :---: | :---: | :---: |
| Thompson | 3.5% | 0.126 | 29.4 |
| forced (c=0.5) | 3.0% | 0.235 | 107.1 |
| **annealed (tau=8000)** | **1.25%** | **0.077** | **−0.7** |
| annealed (tau=500, too fast) | 2.75% | 0.317 | −99.5 |

Slow annealing cuts the genuine-stall rate **2.8×** and turns mean regret from
`29` to `≈0`, exactly where forcing failed. (At 8×8 the stall rate drops 7.25% →
4.5% similarly.)

**Variance caveat (honest).** This aggregate benefit is **high-variance** and does
*not* hold as a clean per-subset inequality. Mean total regret is dominated by a
handful of near-tie outliers and is often *negative* (on a near-tie market the
played matching can beat the true-stable baseline, since the proposer is
indifferent), so on a different 120-market subset Thompson and annealing come out
roughly tied. The robust, low-variance evidence for the churn cure is therefore
the **controlled** unit test `annealing_suppresses_near_tie_churn` (a pure
near-tie bandit, where annealing cuts top-arm flips by >2×), not a market-level
regret inequality — which is why this repository guards the mechanism at the unit
level and merely *documents* the market-level aggregate here rather than asserting
it as a test.

### The annealing trade-off

Cooling too fast (`tau` small) *deepens* the rare lock-in: it commits before the
market settles correctly, so the **worst** single market can get worse even as
the **count** improves. Annealing trades "perpetual mild churn" for "occasional
hard commit." The safe regime is **slow** cooling (`tau` on the order of the
horizon), with a small forcing constant as frozen-arm insurance. Forcing and
annealing compose in one learner (`c > 0` and finite `tau`).

## 4.2 The cascade cure: coordinated near-tie tie-breaking

Annealing settles the *churn*, but the **cascade** residue is different: a
near-indifferent proposer's order, once amplified by Gale-Shapley, lands on a
matching that hurts *others*. No per-agent exploration or annealing schedule can
fix it, because the proposer's beliefs are already correct — it is genuinely
indifferent, and the damage is an externality on the rest of the market.

The cure is **coordination**. Since the proposer is indifferent among its
near-tie arms (belief means within `ε`), reordering them is free for it, so a
market-level coordinator may choose the order. The practical objective is **total
belief welfare**: enumerate the near-tie orderings and pick the matching
maximizing `Σ_p mean_p[partner(p)]` — no true utilities needed. Crucially this is
not the indifferent proposer's own preference (which slightly favors the wrong
arm); it is the *total*, in which the indifferent proposer's `0.005` loss is
dwarfed by another proposer's `0.84` gain.

`examples/coordinated_poc.rs`: on the dissected worst lock-in it recovers the
exact true stable matching (regret `0.93 → 0.00`), matching the true-regret-
optimal (oracle) choice. Over 800 markets it **fully fixes 9/10 settled cascades**
(mean settled regret `0.359 → 0.012`); the one it misses is a frozen-arm case
(wrong beliefs) — forcing's job, not the coordinator's.

> **⚠ Negative result — this is a *post-hoc* result and does NOT transfer to the
> live loop.** The POC above runs on **converged** belief means. When the
> implementation team built the live `CoordinatedMarket` (coordinating every round
> on *current* beliefs), it **lost stability to plain Thompson** (tail-stable
> `0.699` vs `0.919`; tail regret `−0.096` vs `0.0011`). The reason: maximizing
> belief welfare with *inaccurate, mid-learning* beliefs picks a welfare-optimal
> matching that is **not stable** — welfare-max `≠` stable-max once beliefs are
> wrong. So coordination raises proposer welfare (regret goes negative) but
> *lowers* `is_stable`. The live coordinator is shipped **experimental** with this
> caveat. Fixing it is an open research problem: gate coordination by belief
> *confidence* (only reorder near-ties once the relevant posteriors are tight), or
> make **stability** (not welfare) the coordinator's objective. The post-hoc
> success shows the *information* is there at convergence; converting it to a live
> algorithm is unsolved.

### Generality: two-sided markets

The same picture holds when *both* sides learn (`examples/two_sided_stall.rs`,
`TwoSidedMarket`, 300 random 5×5 markets, horizon 20000): plain Thompson on both
sides stalls 9/300 with mean total regret 51, while annealed Thompson on both
sides stalls 4–5/300 with mean total regret ~29. Here the improvement is *cleaner*
than one-sided — both the stall count and the (positive) regret drop consistently
across cooling rates — so the near-tie phenomenon and the annealing cure are not
artifacts of the one-sided setting.

## 5. Recommendations

- **Default**: annealed Thompson with slow cooling (`tau ≈ horizon`) and light
  forcing (`c ≈ 0.25–0.5`). Annealing handles the dominant near-tie churn;
  forcing insures the rare frozen arm. Coordinated near-tie tie-breaking (§4.2)
  is **not** recommended live yet — it lowers stability with mid-learning beliefs;
  it is shipped experimental pending a confidence-gated or stability-objective fix.
- **Report regret honestly**: the `is_stable` flag and regret-vs-unique-stable
  both punish near-tie swaps that the proposer is indifferent to. An ε-stability
  or welfare-based benchmark would not charge a proposer for a swap below its own
  resolution.

## 6. Open problems

- **Live coordinated market (now a known-hard problem).** The implementation
  team's live `CoordinatedMarket` *lost* stability to plain Thompson (§4.2):
  belief-welfare maximization on mid-learning beliefs picks welfare-optimal but
  unstable matchings. The open problem is a coordinator that helps *live*:
  (i) **confidence-gating** — only reorder a near-tie group once its posteriors
  are tight enough that welfare-max ≈ stable-max; (ii) a **stability-direct
  objective** — minimize blocking pairs rather than maximize belief welfare;
  (iii) characterize when the post-hoc information becomes usable as beliefs
  converge. Plus the original scaling issue (search exponential in the largest
  near-tie group).
- **Anytime annealing.** `tau ≈ horizon` needs the horizon. A pull-count-based or
  doubling schedule would be horizon-free.
- **Identifiability-aware regret bound.** A bound of the form
  `regret ≲ Σ_p f(Δ_p, σ, T)` that explicitly shows the `Δ ≪ σ` markets
  saturating at an irreducible floor would formalize §4.

## References

- P. Auer, N. Cesa-Bianchi, P. Fischer. *Finite-time Analysis of the Multiarmed
  Bandit Problem.* Machine Learning, 2002.
- L. T. Liu, H. Mania, M. I. Jordan. *Competing Bandits in Matching Markets.*
  AISTATS 2020.
- A. Sankararaman, S. Basu, K. A. Sankararaman. *Dominate or Delete:
  Decentralized Competing Bandits in Serial Dictatorship.* AISTATS 2021.
