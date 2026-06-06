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
un-resolvability. We prove the **individual** floor rigorously on an explicit
parametric family (GS computed by hand on both branches), then separate out the
harder *net*-floor claim.

> **Proposition 2 (individual cascade floor — rigorous).** There is an explicit
> family of `3×3` markets, parameterized by `(Δ_p, Δ_q)` with
> `Δ_p ≪ σ ≪ Δ_q = Θ(1)`, in which proposer `q`'s per-round regret is **exactly
> `Δ_q`** on every round that proposer `p` mis-orders its near-tie pair. Hence for
> **any decentralized policy** (each agent's ranking depends only on its own reward
> history) run for `T` rounds,
> ```
> E[R_T^q] = Δ_q · Σ_{t≤T} P(p ranks b before a at t) ≥ c · Δ_q · (T − T₀),
> ```
> with `T₀ = Θ(σ²/Δ_p²)` and an absolute constant `c > 0`. For `Δ_p` below the
> noise floor and finite `T` the bracket is `Θ(T)`: a **linear** floor on the
> victim's regret that no exploration schedule removes.

**The instance** (`σ = 0.2`, `Δ_q = 0.6`; proposers `p,q,s`, receivers `A,B,C`):

| util | A | B | C | true order |
|------|----|----|----|-----------|
| `p` | `0.80` | `0.80−Δ_p` | `0.00` | `A ≻ B ≻ C` (top gap `Δ_p`) |
| `q` | `0.05` | `0.70` | `0.70−Δ_q` | `B ≻ C ≻ A` (gap `Δ_q` at `B,C`) |
| `s` | `0.90` | `0.50` | `0.30` | `A ≻ B ≻ C` |

Receiver preferences (known, exact): `A: p≻s≻q`, `B: p≻q≻s`, `C: q≻s≻p`.
(This is the parametric form of `examples/cascade_lower_bound.rs`.)

*Proof.* Both branches are finite GS runs, computed by hand.

- **Correct branch** (`p` reports `A≻B≻C`). Propose: `p→A`, `q→B`, `s→A`. `A` holds
  `p` (`p≻s`), rejects `s`. `s→B`; `B` holds `q` (`q≻s`), rejects `s`. `s→C`; `C`
  holds `s`. Result `M* = {p-A, q-B, s-C}` — the proposer-optimal stable matching
  (GS with proposers proposing). Here `q` gets `B` (utility `0.70`).
- **Mis-order branch** (`p` reports `B≻A≻C`, free to `p` since `A,B` differ by
  `Δ_p`). Propose: `p→B`, `q→B`, `s→A`. `B` holds `p` (`p≻q`), rejects `q`. `q→C`;
  `C` holds `q`. `A` holds `s`. Result `{p-B, q-C, s-A}`. Now `q` gets `C` (utility
  `0.70−Δ_q`).

So `q`'s loss is exactly `0.70 − (0.70−Δ_q) = Δ_q` on every mis-order round, and
`p`'s order of `A,B` is **provably the only swing variable**: it is the sole input
that changes between the two branches, and the two hand-computed GS runs give the
two different partners for `q`. By Lemma 1, until `T₀ = Θ(σ²/Δ_p²)` pulls any
estimator orders `p`'s pair correctly with probability `≤ ½ + o(1)`, and since a
decentralized `p` sees only its own rewards this is unconditional on the others;
for `Δ_p ≲ σ/√T` the pair never resolves, so `P(p mis-orders) ≥ c` throughout.
Multiplying by the per-round cost `Δ_q` and summing gives the bound. ∎

**What kind of floor (sharpening).** The mis-order branch is **not exactly
stable** but it **is `Δ_p`-stable**: its only true blocking pair is `(p, A)` —
`p` truly prefers `A` over its match `B` by `Δ_p`, and `A` prefers `p` over `s`.
That gain is the *near-tie gap* `Δ_p ≪ ε`, so the matching is ε-stable for any
`ε ≥ Δ_p`. The victim `q` cannot block: it covets `B`, but `B` prefers its holder
`p` to `q`. So the floor of Prop. 2 is on **proposer-optimality-gap regret**, not
on (ε-)instability — consistent with `eps_stability.rs` finding the settled stalls
ε-stable. ("Regret" here = distance from the proposer-optimal stable matching.)

**Individual vs net.** In the `3×3` family the cascade is a **redistribution**:
while `q` loses `Δ_q`, proposer `s` *gains* (`C→A`, `+0.60`), so the *net*
proposer regret is `Δ_p + Δ_q − 0.60 < 0` — a per-victim floor, not a net one. The
gain is possible only because the mis-order matching is `Δ_p`-stable (not exactly
stable): a *true*-stable matching is weakly `M*`-dominated for all proposers by the
proposer-optimality theorem, so no one could gain. To get a **net** floor we make
the cascade a pure **descending chain** in which even the last displaced proposer
is downgraded — proved next.

> **Proposition 2′ (net cascade floor — rigorous).** There is an explicit family
> of `4×4` markets, parameterized by `Δ_p ≪ σ`, in which a single near-tie
> mis-order by proposer `p` makes **every** proposer strictly worse off, so the
> per-round **net** proposer regret is `≥ 1.20 = Θ(1)` (no proposer absorbs the
> loss). Hence, by the same Lemma 1 argument, any decentralized policy has
> `E[R_T] ≥ c · 1.20 · (T − T₀)` — a **linear net** floor.

**The instance** (proposers `p,q,r,s`, receivers `A,B,C,D`; `examples/net_floor_4x4.rs`):

| util | A | B | C | D | true order |
|------|----|----|----|----|-----------|
| `p` | `1.00` | `1.00−Δ_p` | `0.10` | `0.00` | `A ≻ B ≻ C ≻ D` (top gap `Δ_p`) |
| `q` | `0.00` | `0.90` | `0.40` | `0.05` | `B ≻ C ≻ D ≻ A` |
| `r` | `0.00` | `0.10` | `0.80` | `0.50` | `C ≻ D ≻ B ≻ A` |
| `s` | `0.30` | `0.20` | `0.10` | `0.70` | `D ≻ A ≻ B ≻ C` |

Receivers (known, exact): `A: p≻q≻r≻s`, `B: p≻q≻r≻s`, `C: q≻r≻p≻s`, `D: r≻s≻p≻q`.

*Proof.* In the correct branch every proposer proposes to its rank-1 receiver and
they are **all distinct** (`p→A, q→B, r→C, s→D`), so `M* = {p-A, q-B, r-C, s-D}`
forms at once; since every proposer holds its top choice, `M*` is trivially stable
**and** proposer-optimal. In the mis-order branch (`p` reports `B≻A`, legal since
`A,B` differ by `Δ_p`) Gale-Shapley is a single rejection chain:
`p→B` displaces `q` (`B: p≻q`); `q→C` displaces `r` (`C: q≻r`); `r→D` displaces `s`
(`D: r≻s`); `s→A`, which is now free (`p` left it), and `A` accepts. Result
`M' = {p-B, q-C, r-D, s-A}`, stable w.r.t. the misreport profile (each proposer's
strictly-preferred receivers all prefer their current holders). True per-proposer
losses: `p: A→B = Δ_p`, `q: B→C = 0.50`, `r: C→D = 0.30`, `s: D→A = 0.40`. The
sum is `Δ_p + 1.20`; **every term is `≥ 0`** (no beneficiary — the freed receiver
`A` is a downgrade even for `s`, its taker), so the net floor is `1.20 = Θ(1)` as
`Δ_p → 0`. The decentralized lower-bound argument of Prop. 2 then applies verbatim
with per-round cost `1.20` in place of `Δ_q`. ∎

The two Gale-Shapley runs and the net-regret arithmetic are reproduced exactly by
`examples/net_floor_4x4.rs` (net `1.210` at `Δ_p = 0.01`, min individual loss
`+0.010`). The seed-235418470 `4×4` (`dissect_stall.rs`) is a *random* witness of
the same mechanism; this family is its clean parametric form.

**Multi-pair additivity.** The floor adds over near-tie swings whose rejection
chains are **vertex-disjoint** (share no proposer or receiver).

> **Proposition 2″ (additivity over disjoint swings).** For any `k`, there is a
> market with `k` near-tie agents whose swings are vertex-disjoint, in which the
> per-round **net** floor is `≥ k · 1.20 = Θ(k)`. Hence the linear lower bound
> scales as `E[R_T] ≥ c · k · 1.20 · (T − T₀)`.

*Proof.* Tile `k` copies of the Prop. 2′ block on disjoint proposer/receiver
index sets and make every cross-block receiver unacceptable (cross utilities `0`,
omitted from rankings). Then no proposer ever proposes outside its block and no
receiver holds a cross-block proposer, so Gale-Shapley runs independently on each
connected component; `M*` and every mis-order outcome decompose blockwise. Each
block contributes its Prop. 2′ net floor `1.20`, and each block's near-tie agent
sees only its own rewards, so by Lemma 1 the `k` mis-order events are independent
with probability `≥ c` each. Summing the per-block floors gives `k · 1.20`. ∎

`examples/multipair_floor.rs` confirms it: net floor `= k · 1.21` for `k = 1..4`,
and mis-ordering a single block adds exactly one block's `1.21` (independence).

**Remaining open.** Only **vertex-sharing** swings are uncovered: when two
near-tie chains route through a common proposer or receiver, GS couples them and
the floors need not add (they can even partially cancel, as the `3×3`
redistribution shows). A general bound for overlapping swings is instance-
dependent and remains open; the disjoint case (Prop. 2″), the single-swing net
floor (Prop. 2′), and the individual floor (Prop. 2) are rigorous.

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

> **⚠ The premise is the catch (live-transfer failure).** Prop. 3 assumes belief
> means are `ε`-accurate outside the near-tie groups — a **converged-belief**
> condition. *During* learning this fails, and then belief-welfare-max picks an
> *unstable* matching: the implementation team's live `CoordinatedMarket` lost
> stability to plain Thompson (tail-stable `0.699` vs `0.919`). So Prop. 3
> characterizes the *target once beliefs are accurate*, not a live algorithm. A
> live coordinator must restrict coordination to groups whose posteriors are
> already `ε`-tight (confidence-gating) so the premise holds — or optimize
> stability directly. **Section 4a (Prop. 4) closes this**, turning the gating
> idea into a posterior-width test with safety and eventual-optimality guarantees.

## 4a. Confidence-gating: a safe online coordinator (Prop. 4)

The ⚠ above says Prop. 3 describes a *target*, not an algorithm: its
`ε`-accuracy premise fails mid-learning, and the naive live coordinator that
ignored this lost stability. We now turn Prop. 3 into an online rule by **gating
coordination on posterior width** — coordinate a near-tie group only once its
posteriors are tight enough to *certify* the premise. This is the spec the
implementation team needs and the guarantee the live failure was missing.

**Posterior width.** With a `N(m₀, τ₀²)` prior and `N_r` observations of variance
`σ²`, arm `r`'s posterior is `N(m̂_r, s_r²)` with
`s_r² = (1/τ₀² + N_r/σ²)^{-1} ≈ σ²/N_r`. Write `s_r` for the posterior std.

**The certification test.** For an adjacent pair `(a,b)` in a proposer's belief
ranking, the posterior on the gap `δ = μ_a − μ_b` is `N(m̂_a − m̂_b, s_a²+s_b²)`.
Call the pair **certified `ε`-tied** when the whole credible interval for `δ` lies
in the indifference band:
```
|m̂_a − m̂_b| + z·√(s_a² + s_b²) ≤ ε,     z = Φ^{-1}(1 − η).
```
By construction this guarantees `P(|δ| > ε) ≤ η`: with confidence `1−η` the pair
is genuinely within the band, so reordering it is information-free for the agent.

> **Lemma 2 (gate ⇒ tightness).** The test can pass only if
> `√(s_a²+s_b²) ≤ ε/z`, hence each arm's posterior std satisfies `s_r ≤ ε/z`
> (and `s_r ≤ ε/(z√2)` in the symmetric case). Define the **gating threshold**
> ```
> g(ε) := ε / (z√2),     equivalently     N_r > 2z²σ²/ε² pulls.
> ```
> The required pull count is `Θ(σ²/ε²)` — finite, and set by the band `ε` the
> coordinator chooses, **not** by the (possibly sub-floor) true gap `Δ`.

*Proof.* Since `|m̂_a − m̂_b| ≥ 0`, the test forces `z√(s_a²+s_b²) ≤ ε`, i.e.
`√(s_a²+s_b²) ≤ ε/z`; each term is bounded by the sum, giving `s_r ≤ ε/z`, with
equality split symmetrically at `ε/(z√2)`. Substituting `s_r ≈ σ/√N_r` and solving
for `N_r` gives `N_r ≥ 2z²σ²/ε²`. ∎

> **Lemma 3 (belief-stability ⇒ approximate true-stability).** If matching `M` is
> stable w.r.t. belief utilities `m̂` with `|m̂_{p,r} − μ_{p,r}| ≤ ε` on every
> `(p,r)` it compares, then `M` is `2ε`-stable w.r.t. the true `μ`: no pair `(p,r)`
> has `μ_{p,r} − μ_{p,M(p)} > 2ε` while `r` also prefers `p` to its match.

*Proof.* A true `2ε`-blocking pair `(p,r)` has belief gain
`m̂_{p,r} − m̂_{p,M(p)} ≥ (μ_{p,r}−ε) − (μ_{p,M(p)}+ε) = (μ_{p,r}−μ_{p,M(p)}) − 2ε > 0`;
receiver preferences are known/exact, so `(p,r)` would block under beliefs too,
contradicting belief-stability. ∎

> **Proposition 4 (gated coordination is safe and eventually optimal).** Run the
> coordinator of Prop. 3 but restrict it to reorder only **certified `ε`-tied**
> groups (test above), leaving every other pair in its belief order. Then:
> 1. **(Safety — resolves the live failure.)** Each reorder is, w.p. `≥ 1−η` per
>    pair, within the true `ε`-band; non-certified pairs are untouched, so the
>    output coincides with the plain belief-GS matching except inside certified
>    bands. By Lemma 3 it is `2ε`-stable w.r.t. truth wherever the acted beliefs
>    are `ε`-accurate — the coordinator can no longer convert a belief-stable
>    matching into an unstable one (the naive version's failure mode).
> 2. **(Eventual activation.)** Under forced exploration (mode 1) `min_r N_r → ∞`,
>    so `max_r s_r → 0` and every true-`ε`-tied group passes the gate after
>    `Θ(σ²/ε²)` pulls/arm — independent of the unresolvable `Δ`.
> 3. **(Optimality once active.)** After activation Prop. 3's premise holds by
>    construction, so the matching is within `O(nε)` of the proposer-optimal `M*`.

*Proof sketch.* (1) The gate passes only when `P(|δ|>ε) ≤ η` for the reordered
pair, so within-band w.p. `≥ 1−η`; the output is GS-stable w.r.t. a belief profile
that is `ε`-accurate on every acted pair (post-gate), and Lemma 3 lifts this to
`2ε`-true-stability. The decisive point is that *non*-certified pairs are left in
belief order, so the coordinator never performs the welfare-chasing reorder of an
un-converged pair that sank the naive build. (2) Forcing's `ε_t = c/t`
least-sampled probe drives `min_r N_r(t) → ∞`, hence `max_r s_r(t) → 0`; Lemma 2
gives the `Θ(σ²/ε²)` activation horizon. (3) Immediate from Prop. 3. ∎

**The spec this hands the implementer.** Plumb the posterior std `s_r` (or `N_r`)
through `Market` / the learner trait, exposed beside `belief_means()`. In the
coordinator, replace the unconditional near-tie grouping with the certification
test `|m̂_a−m̂_b| + z√(s_a²+s_b²) ≤ ε`. Compose with forcing (guarantees eventual
activation) and annealing (churn). Re-validate on **both** tail-stability and
regret: Prop. 4 predicts tail-stability `≥` plain Thompson (deviations are
certified-safe) while regret falls toward the `O(nε)` floor.

**Gaps to close.** `η` is per-pair; a union bound over the `≤ n` near-tie pairs
per proposer gives per-round failure `≤ n²η`, so pick `η = δ/n²` for a global `δ`.
The `Θ(σ²/ε²)` activation assumes forcing reaches every arm at the Auer `c/t`
rate; the exact constant couples to the matching dynamics (which arm is pulled
depends on the current matching) and is stated here for the forced-uniform regime.

**Empirical check — same-belief A/B (`examples/prop4_gating_study.rs`).** To
isolate the gate from the learning-feedback loop, one Thompson loop generates the
beliefs and each measured round forms the `plain` / `ungated` / `gated` decisions
on the *same* posterior means/stds (300 random `5×5` markets, early vs late
regime). It confirms the safety claim and sharpens its scope:

- **Ungated coordination loses stability**, exact-stable `0.74–0.76` vs `plain`'s
  `0.91–0.95` — the live negative finding, reproduced in isolation. Its regret goes
  *negative* (proposers gain beyond `M*`) precisely *because* belief-welfare-max
  picks proposer-favoring **unstable** matchings.
- **Gating restores most of the stability** (`gated` `0.87–0.94`, and `≈ plain` at
  the tight band `ε=0.02`): Prop. 4(1). The `gated` reorder-rate rises early→late
  (`0.00→0.02` at `ε=0.02`, `0.05→0.11` at `ε=0.05`) — the Prop. 4(2) activation
  curve; tighter `ε` ⇒ stricter gate ⇒ closer to `plain` (a safety/coverage knob).
- **Honest limit.** Even with *accurate* beliefs `ungated` stays at `0.76`
  stability, so belief-welfare-max is **not** the right objective (welfare-max ≠
  stable-max even when beliefs are good). The gate *caps the damage* but does not
  fix the objective; this is concrete evidence for the "optimize stability
  directly" alternative above. Consistent with Prop. 4 guaranteeing `2ε`-stability,
  `gated` does give up a little exact stability for coordination.

## 5. Consequences

- **No-go for decentralized policies (Prop. 2, 2′, 2″):** forcing, annealing,
  UCB, Thompson — any per-agent rule — suffer a cascade floor on near-tie
  instances: `Θ(Δ_q)` on the victim (Prop. 2), `Θ(1)` on *net* welfare in the
  descending-chain family (Prop. 2′), and `Θ(k)` over `k` disjoint swings
  (Prop. 2″), all linear in `T`. This is *why* the
  400-market study showed exploration tweaks moving the dominant modes only
  modestly.
- **Coordination is both sufficient (Prop. 3) and, on these instances, necessary.**
  The *naive* live coordinator failed (welfare-max on un-converged beliefs is
  unstable); **Prop. 4 fixes it** by gating coordination on posterior width
  (`s_r < g(ε)`), giving a safe online rule that recovers Prop. 3's `O(nε)` once
  every near-tie group is certified. This is the principled `CoordinatedMarket`
  spec for handoff §3a.
- **Annealing's role is sharpened:** it is the right cure for *churn* (it stops a
  coin-flip that costs the agent itself), but it cannot help *cascade* (the cost
  is an externality the agent is indifferent to).

## References

- Lemma 1 is the standard two-Gaussian hypothesis-testing bound (e.g. best-arm
  identification lower bounds, Mannor–Tsitsiklis 2004; Kaufmann–Cappé–Garivier 2016).
- Matching-bandit regret context: Liu–Mania–Jordan 2020; Sankararaman–Basu–
  Sankararaman 2021 (see `stall-avoidance.md`).
