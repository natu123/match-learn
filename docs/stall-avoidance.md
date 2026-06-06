# Forced-Exploration Thompson Sampling: beating the matching stall

*Research-track note. Companion to `src/learner.rs` (`ForcedExploreThompson`) and
`tests/gate.rs` (`phase1_gate_forced_explore_beats_thompson_stall`).*

## 1. The phenomenon

In the Phase 1 market each proposer treats the receivers as a multi-armed
bandit: pulling arm `a` (being matched to receiver `a`) returns a noisy reward
whose mean is the proposer's unknown utility `μ_{p,a}`. Each round the proposers
submit belief-rankings, Gale–Shapley computes a stable matching of those beliefs
against the *known* receiver preferences, and only the **matched** proposer pulls
its partner and updates.

Plain Thompson Sampling (`GaussianThompson`) explores **only** through posterior
variance. The coupling with matching creates a failure mode that ordinary
bandits do not have:

> **Frozen-arm stall.** When a proposer stops being matched to an arm, that arm
> is never pulled again, so its posterior count, mean, and variance *freeze*. If
> the frozen mean underestimates the proposer's true stable partner `r*`, the
> proposer never proposes to `r*` again, and the whole market can settle into a
> *wrong* stable matching with no path out.

Empirically (see §5) this hits 2 of 40 random `5×5` markets, where greedy
Thompson accumulates regret through the entire tail.

UCB1 avoids freezing — its `√(ln t / N_a)` bonus grows for un-pulled arms, so it
never stops probing — but that bonus never vanishes, so in the coupled loop it
perturbs the match *forever*, leaving heavier regret tails.

## 2. The learner

`ForcedExploreThompson` keeps Thompson Sampling's calm tail while adding a
vanishing exploration floor. Each round `t` (per proposer):

- with probability `ε_t = min(1, c/t)` — a **forced round** — rank the
  **least-sampled** arm first (the rest fall back to posterior means);
- otherwise — a **Thompson round** — draw one posterior sample per arm and rank
  by the samples.

`c ≥ 0` is the forced-exploration constant; `c = 0` recovers greedy Thompson.
The `ε_t = O(1/t)` schedule is the classic Auer–Cesa-Bianchi–Fischer (2002)
ε-greedy rule; targeting the *least-sampled* arm (rather than a uniform random
one) spends the budget precisely on the frozen arm, which is by definition the
least-pulled.

## 3. Notation

- `n` proposers and `n` receivers; reward noise `N(0, σ²)`.
- `Δ` = the minimum preference gap: the smallest difference, over all proposers,
  between the true utilities of two consecutively-ranked receivers. `Δ > 0` means
  every proposer has a strict true preference order.
- `Δ_max` = the largest possible per-round regret (bounded by the utility span
  times `n`).
- `N_a(t)` = number of times arm `a` has been pulled by round `t`.
- `F(t)` = number of forced rounds up to `t`.
- Regret is measured against the **agent (proposer)-optimal stable matching of
  the true market**, summed over proposers and rounds.

## 4. Analysis

The clean statements live in the **single-agent reduction**: fix a proposer and
the set of arms available to it (those no higher-priority proposer claims). This
is an ordinary MAB, and in it forcing the least-sampled arm to the top *does*
result in pulling it. §4.4 lifts the argument to the coupled market and is honest
about what that step assumes.

### 4.1 Lemma 1 — no arm stays frozen: `min_a N_a(t) = Ω(log t)`

The forced rounds are independent Bernoulli`(ε_s)` trials, so

```
E[F(t)] = Σ_{s=1}^{t} min(1, c/s) = c·ln t + O(1),
```

and by a Chernoff bound `F(t) ≥ (1−δ)·c·ln t` with probability `1 − t^{−Ω(δ²c)}`.
Each forced round pulls the *current* least-sampled arm, so forced pulls are
distributed greedily onto the minimum-count arm; after `F` such pulls,

```
min_a N_a(t) ≥ ⌊F(t)/n⌋ ≥ (c/n)·ln t − O(1)   (w.h.p.).
```

So **every arm is pulled `Ω(log t)` times**. The frozen-arm pathology is
impossible for any `c > 0`: no posterior can stay frozen.

### 4.2 Lemma 2 — posteriors concentrate, rankings become correct

With `N_a(t) ≥ k` Gaussian observations, the posterior mean `μ̂_a` obeys

```
P(|μ̂_a − μ_a| ≥ Δ/2) ≤ 2·exp(−k·Δ² / (8σ²)).
```

Substituting `k ≥ (c/n)·ln t` from Lemma 1,

```
P(|μ̂_a − μ_a| ≥ Δ/2) ≤ 2·t^{−α},   α := c·Δ² / (8 n σ²).
```

If *every* arm of a proposer is estimated within `Δ/2` of truth, its
belief-ranking equals its true ranking. By a union bound over the `n` arms and
`n` proposers, the probability that **any** proposer misranks at round `t` is

```
P(misranking at t) ≤ 2 n² · t^{−α}.                       (★)
```

When all proposers rank correctly, Gale–Shapley returns the true stable matching
(under the uniqueness assumption of §4.4), so that round has **zero regret**.

### 4.3 Theorem — sublinear regret, and `O(log T)` above a gap threshold

Split the cumulative regret into three parts:

1. **Forced rounds.** At most `F(T)` of them, each costing `≤ Δ_max`:
   `E[forced regret] ≤ Δ_max·E[F(T)] = O(Δ_max·c·log T)`.
2. **Correct Thompson rounds.** Zero regret.
3. **Misranked Thompson rounds.** Expected count `Σ_{t≤T} 2n²·t^{−α}` by (★),
   each costing `≤ Δ_max`.

Therefore

```
E[R_T] ≤ O(Δ_max·c·log T) + Δ_max·Σ_{t=1}^{T} 2n²·t^{−α}.
```

- If `α > 1`, i.e. **`c > 8 n σ² / Δ²`**, the tail sum converges to `O(1)` and

  ```
  E[R_T] = O( (n σ² / Δ²)·Δ_max·log T ) = O(log T).
  ```

  This matches Auer et al.'s ε-greedy `O(log T)` and Liu et al.'s
  `O(log T / Δ²)` stable regret for matching bandits.

- For **any** `c > 0`, the tail sum is `O(T^{1−α})`, so

  ```
  E[R_T] = O(c·log T) + O(T^{1−α}) = o(T)   (sublinear).
  ```

### 4.4 Corollary — stall probability → 0

By (★), the per-round probability that the realized matching differs from the
true stable matching is `≤ 2n²·t^{−α} → 0`. Greedy Thompson has no such
guarantee: a frozen underestimate gives a *fixed*, non-vanishing misranking
probability, so its stall persists. Forcing replaces that fixed floor with a
`t^{−α}` decay.

### 4.5 Honesty about the coupled market

The single-agent reduction makes two idealizations; both are standard in the
matching-bandit literature, and the gate (§5) is what validates the coupled
behavior directly.

- **Forced ranking ≠ guaranteed pull.** Ranking an arm first only makes the
  proposer *propose* there; a receiver may still reject it. But a persistently
  rejecting receiver is, by definition of the proposer-optimal stable matching,
  *not* that proposer's stable partner — so the pulls it denies do not create
  regret against the agent-optimal benchmark. The pulls that matter (of the true
  stable partner `r*`) do land once the market is near the true configuration,
  because `r*` accepts `p` whenever the proposers `r*` prefers over `p` are
  matched to someone they prefer over `r*` — which holds at the true stable
  matching.
- **Uniqueness condition.** Lifting "all proposers rank correctly ⟹ GS returns
  the true stable matching" to the coupled dynamics needs the true stable
  matching to be unique (or an `α`-reducibility condition ruling out cycling), as
  in Liu et al. (2020) and Sankararaman–Basu. We adopt the same assumption. Our
  contribution is the **forced schedule that guarantees the concentration
  premise** (Lemma 1), which greedy Thompson lacks.

## 5. Empirical corroboration

`tests/gate.rs`, 40 random `5×5` markets, `T = 750`, well-specified noise,
`c = 0.25`. Identical markets for both learners (same seed stream).

| metric (lower is better unless noted)        | greedy Thompson | forced-explore (`c=0.25`) |
|----------------------------------------------|:---------------:|:-------------------------:|
| sublinear markets (of 40, higher better)     |      38/40      |         **40/40**         |
| worst-case doubling ratio `R(2T)/R(T)`       |      2.521      |         **1.465**         |
| worst-case tail regret rate                  |      0.315      |        **0.046** (~7×)    |
| mean tail regret rate                        |     −0.014      |          −0.014           |
| mean total regret vs no-learning (×1488)     |     −5.11       |          −18.09           |

Every market becomes sublinear, the worst-case doubling ratio falls below 1.5,
and the worst-case tail regret rate — the direct signature of the stall — drops
about sevenfold, with no cost to the aggregate metrics.

**A note on the `is_stable` flag.** The gate scores stability by *regret*, not by
the boolean `is_stable` check. That check counts a near-tie swap (two receivers a
proposer values almost equally) as "unstable" even when the regret is ~zero, so a
converged market can show a low tail-stable fraction at negligible regret. The
regret metrics above are the honest ones, and forced exploration improves all of
them.

## 6. Choosing `c`

- The `O(log T)` rate needs `c > 8 n σ² / Δ²`: harder markets (small gap `Δ`,
  large noise `σ`, many arms `n`) want more forcing.
- In practice a *small* constant suffices when gaps are not pathological — the
  sweep in `tests/gate.rs::sweep_c` shows `c = 0.25` already lifts all 40 markets
  to sublinear while forcing only `~0.25·ln T ≈ 2` probes over the whole run.
- Larger `c` (e.g. 3) marginally improves the worst case but forces ~12× more;
  *intermediate* values can transiently over-perturb near-tie markets (their
  doubling ratio is dominated by a near-zero `R(T)` denominator), so prefer the
  smallest `c` that clears the stall.

## References

- P. Auer, N. Cesa-Bianchi, P. Fischer. *Finite-time Analysis of the Multiarmed
  Bandit Problem.* Machine Learning, 2002. (ε-greedy `ε_t = O(1/t)` ⟹ `O(log T)`.)
- L. T. Liu, H. Mania, M. I. Jordan. *Competing Bandits in Matching Markets.*
  AISTATS 2020. (UCB + Deferred Acceptance ⟹ `O(log T / Δ²)` stable regret.)
- A. Sankararaman, S. Basu, K. A. Sankararaman. *Dominate or Delete:
  Decentralized Competing Bandits in Serial Dictatorship.* AISTATS 2021.
  (Phased / forced exploration in decentralized matching markets.)
