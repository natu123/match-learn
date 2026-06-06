# The stability-direct objective for near-tie coordination

*Theory note (EXPLORE sub-track, confirmed live). Explains why a coordinator that
**minimizes blocking pairs** beats one that **maximizes belief welfare** — with no
`2ε` ceiling — and what limit even it cannot beat. Companion to the dev side's
`StabilityCoordinatedMarket` and to [`stall-anatomy.md`](stall-anatomy.md) §4 /
[`theory-identifiability.md`](theory-identifiability.md) §4a. The rigorous parts
and the empirical witnesses are labelled.*

## Setup

A near-tie coordinator (§4a) reorders arms within each near-tie group (belief-mean
gap `< ε`) and runs Gale-Shapley. Two objectives over the reachable reorderings:

- **welfare** — pick the matching maximizing `W = Σ_p mean_p[partner(p)]`
  (the rule that failed live: `ungated`/`gated` belief-welfare).
- **stability** — pick the matching minimizing the number of **blocking pairs**
  (estimated from the current belief profile).

Notation as in `theory-identifiability.md`: `M*` is the proposer-optimal stable
matching of the true market; a near-tie pair has true gap `Δ ≪ σ`; `ε` is the
coordinator's indifference band.

## 1. The welfare objective is stability-*biased*; the stability objective is not

> **Proposition 5.** Even at **perfect information** (beliefs `=` true utilities):
> 1. the stability coordinator attains a blocking-pair-minimal matching, which is
>    **exactly stable** — because `M*` (Gale-Shapley on the true ranking) has zero
>    blocking pairs and is always the *un-reordered base candidate*, so the minimum
>    is `0`;
> 2. the welfare coordinator can select a **strictly unstable** matching: the
>    welfare-maximal reordering is the proposer-welfare-maximal matching within the
>    near-tie freedom, which need not be stable.

*Proof of (1).* The base candidate of every proposer is its true (mean) ranking;
running Gale-Shapley on the base profile yields `M*`, which is stable, hence has
`0` blocking pairs. The stability objective minimizes blocking pairs over a
candidate set that contains the base, so its optimum is `0`. ∎

*(2) is an existence claim, witnessed empirically.* `examples/stability_objective.rs`
(4000 random `5×5` markets, `ε=0.05`, perfect information):

| objective at perfect info | unstable fraction |
|---|---|
| welfare (max `W`) | **27.4 %** |
| stability (min blocks) | **0.0 %** |

The two objectives disagree on exactly those `27.4 %`, with stability strictly
fewer blocks each time. Concrete witness (seed `758695475`): the welfare matching
has welfare `3.250` and **1** blocking pair; the stability matching has welfare
`3.121` and **0**. Welfare buys `+0.129` proposer welfare by going unstable.

**Why welfare deviates (structural).** A near-tie reorder that raises total
proposer welfare does so by moving some receiver to a proposer it ranks *lower*
(to free a better receiver for someone else). That displaced receiver and its
preferred proposer then form a blocking pair. The near-tie agent is indifferent
(its loss `≤ ε`), so welfare-max happily makes this trade — it optimizes proposer
welfare, which is *not* a stability-aligned quantity. The stability objective
scores the blocking pair directly and refuses the trade.

> **Corollary (no `2ε` ceiling).** The gated belief-welfare coordinator (Prop. 4)
> is capped at `2ε`-stability because its **objective** carries the bias of
> Prop. 5(2): even with `ε`-accurate beliefs it keeps the ~27%-type instability,
> so gating can only *bound* the damage. The stability objective has no such bias
> (Prop. 5(1)); its only error is belief noise in the blocking-pair estimate,
> which vanishes as beliefs concentrate. This is the structural reason behind the
> dev's live numbers: belief-welfare `0.699`/`0.909` vs stability **`0.961`**.

## 2. The limit even stability cannot beat — and the regret-sign dichotomy

Removing the bias does not remove the **identifiability** floor.

> **Proposition 6 (dichotomy).** On a near-tie instance with gap `Δ ≪ σ`:
> 1. **Stability is unbiased but still noise-limited.** The stability objective's
>    residual instability is symmetric belief noise: at perfect info it is `0`
>    (Prop. 5(1)), and it vanishes as beliefs become exact. It is *not* a
>    persistent bias. (Contrast: welfare's instability persists at perfect info.)
> 2. **Proposer-optimality is identifiability-bound for any objective.** Selecting
>    the proposer-*optimal* stable matching requires ranking the near-tie pair in
>    `M*`'s order, which by Lemma 1 needs `Θ(σ²/Δ²)` pulls — unbounded as `Δ → 0`.
>    So the Prop. 2 floor stands for every coordinator.
> 3. **The regret sign diagnoses the objective.** Both coordinators pay the
>    Prop. 2 near-tie floor, but on opposite sides: the **welfare** coordinator
>    pays it as **negative** regret (proposer-favoring, *unstable* — proposers gain
>    beyond `M*`); the **stability** coordinator pays it as **positive** regret
>    (it stays stable, so proposers get `≤ M*`, landing on a proposer-suboptimal
>    stable matching).

*Proof.* (1) Prop. 5(1) gives `0` at perfect info; as beliefs → exact, the
estimated blocking-pair counts → true, so the minimizer → a true-stable matching.
There is no perfect-info bias to leave behind. (2) Is Lemma 1 applied to the pair
deciding `M*`'s order (as in Prop. 2/2′). (3) A stable matching is weakly
`M*`-dominated for proposers (proposer-optimality of `M*`), so a *stable*
coordinator has regret `≥ 0`; an *unstable* proposer-favoring matching can give
proposers strictly more than any stable matching, so regret `< 0`. ∎

**Empirical confirmation (dev live loop).** `regret/round`: `ungated −0.096`,
`gated ε=.05 −0.084`, `gated ε=.02 −0.000`, **`stability +0.074`** — exactly the
predicted sign flip, with stability the only positive-regret (genuinely stable)
coordinator, at the highest tail-stability `0.961`.

## 3. Consequences and open directions

- **Recommended live coordinator = stability-targeting** (minimize estimated
  blocking pairs over near-tie reorderings), confirmed dominant; the gated
  belief-welfare coordinator (Prop. 4) is its `2ε`-bounded sibling for settings
  that explicitly want proposer welfare.
- **What is proved:** the welfare objective is stability-biased at perfect info
  (Prop. 5, witnessed `27.4 %`); the stability objective is unbiased (`0 %`); the
  regret-sign dichotomy (Prop. 6) holds and matches the live loop.
- **Open:** a quantitative *rate* — how fast the stability coordinator's residual
  instability decays with the pull budget on coupled near-ties (it inherits the
  `Θ(σ²/Δ²)` resolution barrier only for the *coupled* pairs, `Θ(σ²/ε²)` for the
  rest); and whether a single objective can trade the two sides of Prop. 6
  smoothly (a `λ·welfare − blocks` family).
