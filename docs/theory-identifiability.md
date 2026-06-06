# An identifiability floor for learned stable matching

*Theory note (EXPLOIT track). Formalizes why the **near-tie** stall modes of
[`stall-anatomy.md`](stall-anatomy.md) cannot be removed by any amount of
exploration, and why a coordinator can. Rigorous parts and proof sketches are
labelled as such.*

## Setup and notation

One-sided learned matching: `n` proposers, `n` receivers. Proposer `p` has true
utilities `μ_{p,·}`; a pull of arm `r` returns `N(μ_{p,r}, σ²)`. Receivers have
fixed known preferences. Each round a stable matching is computed from beliefs and
matched pairs update. Regret is against the proposer-optimal stable matching `M*`
of the true market. `N_{p,r}(t)` = pulls of arm `r` by `p` up to `t`.

Write the **decision gap** of an adjacent pair `(a,b)` in `p`'s true order as
`Δ_p(a,b) = |μ_{p,a} − μ_{p,b}|`.

## 1. The resolution lemma (rigorous)

> **Lemma 1.** Fix proposer `p` and two arms `a,b` with gap `Δ = Δ_p(a,b)`. Any
> estimator using `N` i.i.d. pulls of each arm orders them correctly with
> probability at most `1 − Φ(−Δ√N / (σ√2))`, where `Φ` is the standard normal
> CDF. Equivalently, to achieve error `≤ δ` one needs
> `N ≥ (2σ² / Δ²) · (Φ^{-1}(1−δ))²`.

*Proof.* The difference of the two empirical means is
`N(μ_a − μ_b, 2σ²/N)`. Ordering is correct iff this difference has the right sign;
its probability is `Φ(Δ√N/(σ√2))`. The Bayes-optimal rule (sign of the difference)
maximizes this, so no estimator does better. Rearranging gives the sample bound. ∎

**Reading.** Resolving a gap `Δ` to confidence `1−δ` costs `Θ(σ²/Δ²)` pulls. For
`Δ = 0.001, σ = 0.2` (the dissected lock-in) and `δ = 0.05`, that is
`N ≳ (2·0.04/10^{-6})·(1.64)² ≈ 2.2×10^5` pulls of *each* arm — beyond any
practical horizon. The order is then effectively a fair coin **for every policy**,
because Lemma 1 bounds *all* estimators, not a particular learner.

## 2. Why exploration does not help the near-tie modes

Forcing and annealing change *which* arms are pulled and *how* samples are turned
into a ranking; neither changes the information content of `N` pulls, so neither
escapes Lemma 1. Concretely:

- **Churn (mode 2).** When `Δ_p(a,b) ≲ σ/√N`, the posterior means of `a,b` stay
  within sampling noise, so Thompson's per-round order is a coin flip — perpetual
  re-sampling. Annealing does not *resolve* the pair; it *stops asking*, freezing
  the (arbitrary) current order. That removes the churn regret but is a tie-break
  choice, not an identification.
- **Cascade (mode 3).** Same un-resolvability, but now the arbitrary order is
  amplified by Gale-Shapley. This is the expensive case, bounded next.

## 3. A per-agent regret floor from cascades (proposition + sketch)

The cascade cost is driven by *another* agent's gap, gated by the near-tie agent's
un-resolvability. We state it for the clean two-decision instance the dissection
exhibits and sketch the general claim.

> **Proposition 2.** There is a family of markets, parameterized by `(Δ_p, Δ_q)`
> with `Δ_p ≪ σ ≪ Δ_q = Θ(1)`, in which: proposer `p` is near-indifferent between
> two receivers `a,b` (gap `Δ_p`); the proposer-optimal stable matching `M*`
> requires `p` to rank `a` before `b`; and if `p` instead ranks `b` before `a`,
> Gale-Shapley yields a matching in which proposer `q` loses its `M*` partner, a
> per-round regret `≥ Δ_q`. Consequently, for **any decentralized policy** (each
> agent rankings depending only on its own reward history) run for `T` rounds,
> ```
> E[R_T] ≥ Δ_q · Σ_{t≤T} P(p ranks b before a at round t) ≥ c · Δ_q · (T − T₀),
> ```
> where `T₀ = Θ(σ²/Δ_p²)` is the horizon before `p` could even in principle
> resolve the pair, and `c > 0` is an absolute constant. With `Δ_p` fixed below
> the noise floor and finite `T`, the bracket is `Θ(T)`: a **linear** regret floor
> that no exploration schedule removes.

*Proof sketch.* (i) The instance is the dissected `4×4` market (seed 235418470)
made parametric: scale `p`'s two top utilities to differ by `Δ_p` and `q`'s by
`Δ_q`; receiver preferences fixed so that `p`'s order is the unique swing variable
deciding `q`'s partner (verified for the base instance in `dissect_stall.rs`).
(ii) By Lemma 1, until `T₀ = Θ(σ²/Δ_p²)` pulls, *any* estimator of `p` orders
`a,b` correctly with probability `≤ 1/2 + o(1)`; a decentralized `p` sees only its
own rewards, so this applies regardless of the other agents. Hence
`P(p ranks b before a) ≥ c` for `t ≤ T₀`, and (since the pair never resolves for
`Δ_p` below the floor) for the whole horizon when `Δ_p ≲ σ/√T`. (iii) Each such
round costs `≥ Δ_q` by construction. Summing gives the bound. ∎

**Gaps to close (for full rigor).** The base-instance "unique swing variable"
claim is checked numerically, not proved in general; a fully general lower bound
needs an instance family with a proved GS sensitivity. The constant `c` and the
coupling between `p`'s pull count and the matching dynamics are stated for the
single-swing instance; the multi-pair case is conjectured to add over pairs.

## 4. Why coordination escapes the floor (rigorous, given the band)

> **Proposition 3.** Suppose at round `t` every proposer's belief means are
> `ε`-accurate on all arms except within near-tie groups of true width `≤ ε`, and
> the coordinator picks, among all within-group orderings, the matching maximizing
> total belief welfare `W = Σ_p mean_p[partner(p)]`. Then the chosen matching's
> true total welfare is within `2nε` of the proposer-optimal stable matching `M*`.

*Proof.* `M*` is achievable by *some* within-group ordering (reorder each near-tie
group to its `M*` order; this only permutes arms whose means are within `ε`, so it
is a legal candidate). Belief welfare and true welfare differ by at most `nε`
(each of `n` proposers is matched to an arm whose mean is `ε`-accurate, except
near-tie arms which are within `ε` of each other and of truth). The maximizer of
`W` therefore has true welfare `≥ W(M*) − nε ≥ trueW(M*) − 2nε`. ∎

**Reading.** The floor of Prop. 2 is `Θ(Δ_q)` per round; coordination drops it to
`O(nε)`, with `ε` the indifference band the coordinator may set near the noise
floor. The near-tie agent is indifferent within `ε`, so this costs it nothing —
the cure is *information-free re-coordination*, exactly what exploration cannot do
and a market-level mechanism can. This is the theory companion to the 9/10
empirical coverage in `coordinated_poc.rs`.

## 5. Consequences

- **No-go for decentralized policies (Prop. 2):** forcing, annealing, UCB,
  Thompson — any per-agent rule — suffer the `Θ(Δ_q)` cascade floor on near-tie
  instances. This is *why* the 400-market study showed exploration tweaks moving
  the dominant modes only modestly.
- **Coordination is both sufficient (Prop. 3) and, on these instances, necessary.**
  It motivates the live `CoordinatedMarket` (handoff §3a) as the principled fix.
- **Annealing's role is sharpened:** it is the right cure for *churn* (it stops a
  coin-flip that costs the agent itself), but it cannot help *cascade* (the cost
  is an externality the agent is indifferent to).

## References

- Lemma 1 is the standard two-Gaussian hypothesis-testing bound (e.g. best-arm
  identification lower bounds, Mannor–Tsitsiklis 2004; Kaufmann–Cappé–Garivier 2016).
- Matching-bandit regret context: Liu–Mania–Jordan 2020; Sankararaman–Basu–
  Sankararaman 2021 (see `stall-avoidance.md`).
