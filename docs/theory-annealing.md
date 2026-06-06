# Annealing convergence: why cooling cures near-tie churn

*Theory note (EXPLOIT track). Formalizes the churn cure of
[`stall-anatomy.md`](stall-anatomy.md) §4 and the cooling-too-fast trade-off.
Numerically validated by `examples/anneal_convergence.rs`.*

## The churn model

Near-tie **churn** is the regime where a proposer's belief **means already give
the right order**, but Thompson Sampling keeps re-drawing two near-equal arms in
different orders, so the matching never settles. It is sustained by a **frozen
posterior**: in the matching loop the currently-unmatched competitor stops being
pulled, so its posterior stays wide while its mean is (by hypothesis) roughly
correct.

Model one such pair as two fixed posteriors, `a ~ N(m_a, s²)` and
`b ~ N(m_b, s²)`, with the correct order `m_a > m_b` at a near-tie gap
`δ = m_a − m_b` with `δ ≪ s`. Annealed Thompson draws `X_a ~ N(m_a, (α_t s)²)`,
`X_b ~ N(m_b, (α_t s)²)` and ranks by the samples, with cooling factor
`α_t = sqrt(tau / (tau + t))` (`tau = ∞` is plain Thompson). A **churn round** is
one where `X_b > X_a` (the order flips off the truth).

## 1. Per-round churn probability (rigorous)

> **Lemma A.** The churn probability at round `t` is
> `q_t = Φ( −δ / (α_t · s·√2) )`, where `Φ` is the standard normal CDF.

*Proof.* `X_b − X_a ~ N(−δ, 2 α_t² s²)`; a churn round is `X_b − X_a > 0`, with
probability `Φ(−δ / (α_t s√2))`. ∎

`examples/anneal_convergence.rs` confirms Lemma A to three digits (empirical tail
rate vs the formula): e.g. `tau=2000 → 0.376` vs `0.373`, `tau=100 → 0.0845` vs
`0.0784`, `tau=20 → 0.0010` vs `0.0008`.

## 2. Convergence theorem (cumulative churn is finite)

> **Theorem B.** Under `α_t = sqrt(tau/(tau+t))` with any finite `tau > 0` and
> `δ > 0`, the expected cumulative churn `Σ_{t≥1} q_t` is **finite**, so the tail
> churn rate `q_t → 0`. Without annealing (`α_t ≡ 1`), `q_t ≡ Φ(−δ/(s√2)) > 0` is
> constant, so cumulative churn grows **linearly**.

*Proof.* With `α_t² = tau/(tau+t)`,
`δ/(α_t s√2) = (δ/(s√2)) · sqrt((tau+t)/tau) ≥ (δ/(s√(2 tau))) · √t`. Writing
`c = δ/(s√(2 tau)) > 0`, monotonicity of `Φ` gives `q_t ≤ Φ(−c√t)`. The Gaussian
tail bound `Φ(−u) ≤ ½ e^{−u²/2}` yields `q_t ≤ ½ e^{−c² t/2}`, a convergent
geometric-type series: `Σ_t q_t ≤ ½ Σ_t e^{−c² t/2} = O(1/c²) = O(s² tau/δ²) < ∞`.
The plain case is immediate since `q_t` is constant. ∎

So annealing converts a **linear** churn cost `Θ(T)·Φ(−δ/(s√2))` into a **bounded**
one `O(s² tau / δ²)`. The numerics match: cumulative churn falls from `18,837`
(plain) to `1,974` (`tau=20`) over `40,000` rounds, and the tail rate from `0.47`
to `0.001`.

## 3. The cooling-too-fast trade-off (rigorous statement)

Cooling does not *resolve* the pair — it *commits* to whichever order the belief
means currently encode. The committed order is `sign(m_a − m_b)`, which is correct
only if the belief means have the true order. By the identifiability bound
(`theory-identifiability.md`, Lemma 1), with `N` pulls of the frozen arm the
belief order is correct only with probability `Φ(Δ_true √N / (σ√2))`, where
`Δ_true` is the true gap.

> **Corollary C.** A schedule that has effectively committed by round `t`
> (i.e. `α_t s ≲ δ`) locks in the belief order as of `t`. If the frozen arm has
> received only `N_t` pulls by then, the probability of committing to the *wrong*
> order is `≈ Φ(−Δ_true √N_t / (σ√2))`. Faster cooling (smaller `tau`, hence
> smaller `t` and `N_t` at commit) **increases** this wrong-lock-in probability.

This is exactly the empirical trade-off: small `tau` collapses churn but can
deepen a rare lock-in (it commits before the frozen arm is corrected — which is
the **frozen-arm** mode, where the cure is *forcing*, not annealing). Hence the
composed recommendation: **slow cooling** (`tau` on the order of the horizon, so
commitment happens after enough pulls) **plus light forcing** as frozen-arm
insurance. Annealing handles churn (`δ` real, order correct); forcing handles the
frozen overestimate (order wrong); coordination handles the cascade externality
(`theory-identifiability.md` Prop. 3).

## 4. Scope

Lemma A and Theorem B are exact for the fixed-posterior churn model. In the live
matching loop the posteriors drift (the incumbent tightens, the rival re-freezes
at varying widths), so `s` and `δ` are effective values, not constants; the
qualitative conclusion (finite vs linear churn) carries over and is what
`anneal_study.rs` measures at the market level (5×5 genuine-stall rate
`3.5% → 1.25%`). A fully dynamic proof would track the coupled pull counts and is
left to the agenda.
