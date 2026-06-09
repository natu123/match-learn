# Theory verification — paper ① core (2026-06-10, dev math-verification track)

Adversarial audit of the paper-① theory core, per the command-center handoff
(vault `02_Loa/Messages/others/2026-06-10_dev-theory-verification-handoff.md`).
Claims audited as recorded in memory `projects/irreversible-online-learning.md`
(the 正本; line references below are to its 2026-06-10 state).

Companion experiment: `examples/theory_verification.rs` on branch
`research/theory-verification` (worktree off `master` @ 192d4a0). Reproduce with
`cargo run --release --example theory_verification`. Its output is quoted in
each section.

## Verdict summary

| # | target | verdict |
|---|--------|---------|
| 1 | Theorem B lower bound (Bretagnolle–Huber) | **修正要 (修正案つき)** — the `Θ(σ² log T /(k Δ²))` rate stands; five rigor fixes, corrected proof in §1 |
| 2 | Theorem A key lemma (pre-commit instance-blindness) | **検証済み (論文可)** — formal coupling lemma supplied (§2); two model assumptions must be stated explicitly |
| 3 | Rigid core / embedding uniqueness | **検証済み (論文可)** — full proof in §3 + 72-case exhaustive check; one spec typo found |
| 4 | Two-Floor Dichotomy single statement | **修正要 → 起草済み** — unified statement drafted in §4; two-layer presentation recommended |
| 5 | Basu σ normalization | **検証済み (条項つき)** — noise is 1-subgaussian; exact-match claims need a "σ = 1 normalization" clause |

Nothing collapsed. All three load-bearing lower-bound/uniqueness claims survive
adversarial reading; every issue found is a fixable rigor gap, and each fix is
supplied below.

---

## 1. Theorem B lower bound — audit and corrected proof

**Claim as recorded** (正本 L140): Bretagnolle–Huber gives
`P_I(Aᶜ)+P_II(A) ≥ ½ exp(−KL)`, `KL ≈ E[N]·Δ²/(2σ²)`; error `δ` needs
`E[N] ≥ (2σ²/Δ²) log(1/4δ)`; `regret ≥ (u*/k)·E[N] + ¼ΔT·exp(−E[N]Δ²/2σ²)`;
minimizing over `E[N]` gives `Θ(log T/Δ²)`.

### Findings (issues, in decreasing weight)

- **F1 — mixed units.** The displayed bound is in *reward* units (`u*`, `Δ`
  multiply the cost terms) while the paper's main metric is *binary stable
  regret* (正本 L119). In binary units both the pre-commit opportunity cost and
  the wrong-commit cost are `1`/round, so `u*` and `Δ` must *not* appear in the
  cost terms; `Δ` enters only through the KL. State one version (binary
  recommended) and derive the other as a remark, not interchangeably.
- **F2 — `E[N]` under which measure?** The divergence decomposition yields
  `KL(P_I|_{F_{τ∧T}} ‖ P_II|_{F_{τ∧T}}) = E_I[N]·Δ²/(2σ²)` — the expectation is
  under instance I. The interview-cost term must then also be `E_I[N]`, which
  forces the bound through `max(R_I,R_II) ≥ ½(R_I+R_II)`. Writing a bare `E[N]`
  in both terms is not wrong but hides this; make it explicit.
- **F3 — remaining horizon and the never-commit case.** The wrong-commit cost
  is `(T−τ)⁺`, not `T`, and "never commit" must be charged separately. Both are
  absorbed at once by the pointwise identity used below
  (`loss = T` exactly on {wrong commit} ∪ {no commit}), which also removes the
  正本's coefficient-comparison step ("`Δ ≤ 1`") entirely.
- **F4 — lower bounds need a concrete noise law.** σ-subgaussian is an
  upper-bound-side class (it caps, not pins, the KL). Instantiate the lower
  bound with Gaussian `N(·, σ²)` noise: per-sample `KL = Δ²/(2σ²)` where `Δ` is
  the mean shift of each pivot pair across I/II. (Implementation footnote: the
  simulator's noise is Bernoulli; near mean ½,
  `KL(Bern(½+Δ/2)‖Bern(½−Δ/2)) ≈ 2Δ²`, i.e. effectively Gaussian with
  `σ² = ¼` — consistent.)
- **F5 — stopping-time KL.** `N` is a random count up to the stopping time
  `τ∧T`, so the per-sample KL multiplies `E_I[N]` only via the divergence
  decomposition at stopping times (Wald-type): Kaufmann–Cappé–Garivier 2016,
  Lemma 1, or Lattimore–Szepesvári, *Bandit Algorithms*, Ch. 15. Also state
  that `N` counts interviews **on the two differing pairs only** — all other
  observation coordinates have identical laws under I/II and contribute zero
  KL — and that `N ≤ k·(τ∧T)` because interviews exist only pre-commit.

### Corrected proof (binary stable regret)

Setup: embedding pair I/II (§3 gives each a unique benchmark `μ*_I ≠ μ*_II`,
differing in `a*`'s slot), Gaussian noise `σ²`, `k` interviews per agent per
round, absorbing commits. Let `τ` = `a*`'s commit round (∞ if none), `N` = #
interviews on `{(a*,f₁),(a*,f₂)}` before `τ∧T`, `κ = Δ²/(2σ²)`,
`A = {τ ≤ T, target = f₂} ∪ {τ > T} ∪ {target ∉ {f₁,f₂}}` (so
`Aᶜ = {τ ≤ T, target = f₁}`).

1. Pre-commit rounds leave `a*` unmatched, so the realized matching misses
   `μ*` and pays 1: `R ≥ τ∧T ≥ N/k`. A wrong (or core) commit pays 1 for every
   remaining round: on {wrong-or-no-commit}, `R ≥ τ∧T + (T−τ)⁺ = T`.
2. Hence `R_I + R_II ≥ E_I[N]/k + T·(P_I(A) + P_II(Aᶜ))` (the bad event under
   I is exactly `A`; the bad event under II contains `Aᶜ`).
3. Bretagnolle–Huber + divergence decomposition (F5):
   `P_I(A) + P_II(Aᶜ) ≥ ½ exp(−E_I[N]·κ)`.
4. So `max(R_I,R_II) ≥ ½(R_I+R_II) ≥ ½ min_{m≥0} g(m)`,
   `g(m) = m/k + (T/2)e^{−κm}`. For `kκT/2 ≥ 1` the minimum is at
   `m* = (1/κ)·ln(kκT/2)` with value `(1/(kκ))(ln(kκT/2) + 1)`, giving

   `max(R_I, R_II) ≥ (σ²/(kΔ²))·(ln(kΔ²T/(4σ²)) + 1)  = Ω(σ² log T /(k Δ²))`. ∎

Two by-products worth keeping: the **`1/k` prefactor is part of the floor**
(more interview budget per round buys regret down linearly), and the bound
needs no case split (the `loss = T` identity covers late, never, and core
commits at once).

### Numerical confirmation (`theory_verification`, V1)

```text
T 4k/16k/64k: 1799 / 1997 / 2218; increments 198 vs 221  (log T ⇒ equal)
k 1/2/4:      3917 / 1997 / 999;  ratios 1.96, 2.00      (1/k ⇒ 2.00)
Δ_A 0.2→0.1:  1182 → 4563;        ratio 3.86             (1/Δ² ⇒ 4.00)
algebra: grid argmin m=146.2 g=84.2 vs closed form m*=146.2 g*=84.2
```

The interview-then-commit policy's measured stable regret has exactly the
`(σ²/(kΔ_A²))·log T` shape — including the newly explicit `1/k` — and the
minimization algebra checks numerically to the digit.

**Verdict: 修正要 (修正案つき).** The rate `Θ(σ² log T/(k Δ_A²))` survives;
F1–F5 are presentation/rigor fixes with no substantive damage. The upper-bound
side (interview to confidence `δ = 1/T`, `conf = ln(2m/δ)`) already matches the
implementation (`src/embedding.rs`, `delta`/`conf`/`resolved`).

---

## 2. Theorem A key lemma — formalization

**Claim as recorded** (正本 L138): without interviews the pre-commit history is
instance-independent; hence the law of (commit time, target) is the same under
I/II, and the averaged regret is `Ω(T)`.

### Formal lemma (instance-blindness by coupling)

*Model assumptions to state explicitly in the paper:*

- **(A1) Observation channels.** A pair's reward stream is observed only while
  that pair is matched; interviews are the only pre-match observation channel.
  No other signal (public or private) depends on the instance.
- **(A2) Localized difference.** Instances I and II differ only in the mean
  vector of the channels `{(a*,f₁), (a*,f₂)}` (true for the embedding pair:
  only `a*`'s two means swap). Consequently all *other* agents' observations
  have identical laws under I/II — so even fully public rewards leak nothing.

**Lemma.** Under (A1)–(A2), no interviews, and absorbing commits, every
(randomized) algorithm's pre-commit trajectory, commit time `τ`, and commit
target have the same joint law under I and II.

*Proof.* Couple the two runs: same algorithm randomness `ω`, same noise
realizations on every channel whose law agrees (all channels except the two
pivot pairs, by A2). Induct on rounds: before `a*` commits, the only ways to
sample a pivot channel are to match `a*` with `f₁/f₂` (that *is* the commit,
absorbing) or to interview (disabled). So every observation made so far comes
from law-identical channels and is pointwise equal under the coupling; the
algorithm's next action is a measurable function of `(ω, history)`, hence
equal. The trajectories coincide up to and including the commit decision. ∎

**Corollary (Ω(T)).** Let `p = P(commit f₁ by T)`, `q = P(commit f₂)`,
`c = P(commit elsewhere)`, `r = P(no commit)` — instance-independent by the
lemma. In binary stable regret, a wrong/elsewhere/no commit pays `T` (as in
§1 step 1): `R_I ≥ T(q+c+r)`, `R_II ≥ T(p+c+r)`, so
`max(R_I,R_II) ≥ ½ T (p+q+2c+2r) ≥ T/2`. ∎

This replaces the 正本's coefficient comparison (`½u* ≥ Δ/2 ⟺ Δ ≤ 1`): in
binary units no condition is needed, and the constant is a clean `½`.

### Model–code consistency

`src/embedding.rs::simulate_market` provides observations *only* at the
interview branch (active iff `Policy::Interview` and not committed) and the
recoverable-bandit branch (active iff `reversible && NoInterview`). In the
irreversible no-interview regime the commit at `t = 0` is computed from seeded
noise alone — the simulator literally cannot read `prop_utils` before its
first observation. (A1) holds by construction; (A2) holds for `embed`'s I/II.

### Numerical confirmation (`theory_verification`, V2)

```text
400 seeds: R_I + R_II ≥ T always; ended-stable I: 0.193, II: 0.193
```

Same-seed coupled runs share their realized trajectory, so the two regrets sum
to ≥ T on every seed (no exception in 400), at most one instance ends stable
(never both), and the ended-stable frequency is instance-symmetric — the
executable face of the lemma. The 80.7% of seeds that commit off-benchmark pay
linearly forever: `E[R] ≥ 0.8·T` here, the measured face of `Ω(T)`.

**Verdict: 検証済み (論文可)** once (A1)–(A2) are stated as model assumptions
and the lemma is written as above (the 正本's one-line version is correct in
substance; the unit slip `½u*·E[τ_c]` disappears in binary units).

---

## 3. Rigid core / embedding uniqueness — proof

**Claim** (正本 L154, ① spec): the embedding market has a unique stable
matching; the core resolves rigidly; the pivot alone swings it.

**Theorem.** For `n ≥ 2`, `0 < Δ_A < Δ_big`, `Δ_big·(n−1) ≤ 0.9`, and (only
needed when `n = 2`) `Δ_A < 0.8`, each instance of `embed(Δ_A, Δ_big, n, ·)`
has **exactly one** stable matching, which is also its **unique super-stable**
matching: `μ* = {a* → pivot-better firm, a_s → the other, a_i → f_i (i ≥ 2)}`
— and instances I/II swap exactly the two gadget slots.

*Proof.* Unmatched utility is 0; blocking = both strictly improve;
weakly-blocking = both weakly, one strictly (its absence = super-stability).

**Step 1 (mutual-top forcing).** If `a` and `f` are each other's strict global
optima, every stable matching pairs them — otherwise `(a,f)` blocks, each side
strictly preferring the other to its current partner or to being unmatched.

- Core pairs `(a_i, f_i)`, `i ≥ 2`: `u(a_i,f_i) = 0.9`, every alternative
  `≤ 0.9 − Δ_big`; symmetrically for `f_i`. Mutual-top, so `μ(a_i) = f_i` for
  all `i ≥ 2` *simultaneously* (no induction needed).
- Pivot pair `(a*, f_best)`: `a*`'s top is `f_best`
  (`½+Δ_A/2 > ½−Δ_A/2 > 0.1`; the second inequality needs `Δ_A < 0.8`, which
  for `n ≥ 3` follows from `Δ_A < Δ_big ≤ 0.45`); `f_best`'s top is `a*`
  (`0.9 > 0.9−Δ_big > 0.1`). So `μ(a*) = f_best`.

**Step 2 (last pair).** Only `a_s` and `f_other` remain. If either were
unmatched, `(a_s, f_other)` blocks: `u(a_s, f_other) ∈ {0.8, 0.2} > 0` and
`u(f_other, a_s) = 0.9 − Δ_big > 0`. So `μ(a_s) = f_other`, and `μ = μ*` is
the unique stable matching.

**Step 3 (super-stability).** Uniqueness gives at most one; it remains to
check `μ*` has no weakly-blocking pair. Every off-matching pair has at least
one side *strictly* worse than its `μ*`-partner: `(a*, f_other)` — `a*` strictly
worse (gap `Δ_A`); `(a*, core)` — strictly worse (`0.1 < ½−Δ_A/2`);
`(a_s, f_best)` — firm strictly worse (`0.9−Δ_big < 0.9`); `(a_s, core)` — `a_s`
strictly worse (`0.1 < 0.2`); `(a_i, f_j), j ≠ i` — `a_i` strictly worse
(`≤ 0.9−Δ_big < 0.9`); `(a_i, f₁/f₂)` — firm strictly worse (`0.1 < 0.9−Δ_big`).
So `μ*` is super-stable. ∎

Notes:

- The utilities contain ties (e.g. `a*`'s flat `0.1` over core firms), but the
  forcing argument never appeals to those comparisons — which is *why* unique
  stable and unique super-stable coincide here. This matters because the
  benchmark used by `admissible_gap` and `simulate_market` is the
  super-stable/GS matching.
- **Spec typo (non-load-bearing):** 正本 L154 says the core agent's top is
  `f_{i+1}`; the implementation, the tests, and this proof use the aligned
  identity diagonal `f_i`. Fix the spec line.
- Conditions to carry into the paper statement: `Δ_A < Δ_big`,
  `Δ_big (n−1) ≤ 0.9`, and the `n = 2` proviso `Δ_A < 0.8`.

### Numerical confirmation (`theory_verification`, V3)

```text
unique stable = unique super-stable = predicted, 72 parameter cases
```

Exhaustive enumeration over **all partial matchings** (unmatched slots
included, so individual rationality is genuinely tested), for
`n ∈ {2,3,4,5} × Δ_A ∈ {0.02, 0.1, 0.149} × Δ_big ∈ {0.15, 0.2, 0.225} ×`
both instances.

**Verdict: 検証済み (論文可)** — proof above can be lifted verbatim.

---

## 4. Two-Floor Dichotomy — proposed single statement

**Recommended structure: two layers.** Layer 1 states the dichotomy
self-contained on the embedding family (everything either proven here or
proven by the command center); Layer 2 lifts to general markets via the
admissible gap, importing Basu/Mirfakhar with explicit model-translation
remarks. This keeps the theorem honest: cells (b),(c) are ours; cells (a),(d)
are imported.

> **Theorem (Two-Floor Dichotomy).** Fix `σ > 0`. Consider a two-sided market
> with admissible gap `Δ_A > 0` (Basu 2506.15926, Def. 7), rewards = mean +
> σ-subgaussian noise (lower bounds instantiated with Gaussian `N(·,σ²)`),
> benchmark = its unique super-stable matching, `R(T)` = binary stable regret,
> and an A-channel of `k` safe interviews per agent per round (observations
> that pay no reward and commit nothing). As match dynamics and channel vary
> over the 2×2 `{recoverable, absorbing} × {k = 0, k ≥ 1}`:
>
> (a) **recoverable, k = 0**: `R(T) = Θ(σ² log T / Δ_A²)` — the
>     *identifiability floor* (finite rate). [Basu Thm 3.2 + §5]
>
> (b) **absorbing, k = 0**: for every algorithm there is an instance with
>     `R(T) ≥ T/2` — the *irreversibility floor* (impossibility). [Thm A; §2]
>
> (c) **absorbing, k ≥ 1**: `R(T) = Θ(σ² log T / (k Δ_A²))` — the A-channel
>     restores the recoverable rate; the absorbing commit's `1/T` error demand
>     is what keeps the `log T`. [Thm B; §1]
>
> (d) **recoverable, k ≥ 1**: `R(T) = O(poly(n,m) · σ²/Δ_A²)`,
>     horizon-independent. [Mirfakhar 2602.12224]
>
> In particular, safe interviewing **substitutes for reversibility**
> ((b) → (c)), while under recoverability it is an accelerator, not a
> necessity ((a) → (d)).

*Un-samplability remark (the unifying axis):* the decision-relevant
uncertainty shrinks at (information per safe sample) × (number of payable safe
samples). The identifiability floor caps the first factor at `Δ_A²/σ²`; the
irreversibility floor sets the second to zero; the A-channel restores the
sample supply, which is why one mechanism dissolves both floors — at the rate
in (c), not for free.

*Honesty clauses to attach:*

1. (a)/(d) are imported; their models differ in detail (Basu: 1-subgaussian,
   decentralized variants; Mirfakhar: recoverable with strategic deferral).
   State the translation: σ-normalization (§5), centralized benchmark, binary
   regret in all four cells (Basu's `R_{0/1}` is exactly this).
2. The `Θ` in (c) hides a `poly(n, m)` prefactor on the upper side (round-robin
   interviewing over `m` firms; multiple simultaneous near-ties open). This is
   the known "prefactor poly(n,m)" enhancement, not a gap in the dichotomy.
3. Lower bounds in (a),(b),(c) are witnessed by the embedding family
   `embed(Δ_A, Δ_big, n)` — §3 makes the benchmark well-defined and pivot-only;
   so a single instance family carries all four cells (Layer 1), and the
   admissible gap carries the statement to general markets (Layer 2).

**Verdict: 修正要 → 起草済み.** Decisions the command center should bless:
(i) the two-layer structure; (ii) Gaussian instantiation for lower bounds;
(iii) the now-explicit `1/k` in (c) (measured: ratios 1.96 / 2.00).

---

## 5. Basu σ normalization (余力 target)

Fetched from `arxiv.org/html/2506.15926v1` (2026-06-10):

- **Noise model:** "η_{i,j}(t) and η_{j,i'}(t) are independent **1-subgaussian**
  noise" — σ = 1 convention; means unbounded, no [0,1]/Bernoulli restriction.
- **Thm 3.2 (centralized upper):**
  `E[R_{0/1}(T)] ≤ K + NKπ²/3 + 96·K·log T / Δ_A²(μ,γ)` — no explicit σ.
- **Def. 7 (admissible gap):** largest minimum gap over admissible partial
  rankings (compatible with the true ranks, non-empty super-stable set) — the
  same object `admissible_gap` computes (cross-checked 2026-06-08).
- **§5 lower bound:** `Ω(K_eff log T / Δ_eff²)`; the full general form was
  truncated in the HTML render — re-derive/quote from the PDF at submission.

**Conclusion.** The exact-match claim "identifiability floor `σ²/Δ²` ≅ Basu's
`Δ_A^{-2}` rate" is valid **under σ = 1 normalization**; for general σ, divide
utilities by σ (the admissible gap is scale-equivariant), i.e. replace `Δ_A`
by `Δ_A/σ`. State the clause once and the correspondence is exact, not just
rate-level. Implementation footnote: the simulator's Bernoulli noise is
effectively σ² = ¼ (near-½ KL `≈ 2Δ²`), consistent with the measured
prefactors.

**ID confirmations (fetch, not memory):** 2506.15926 = Basu, "Competing
Bandits in Matching Markets via Super Stability" ✓; 2502.14043 = Plaut,
Liévano-Karim, Zhu, Russell, "Safe Learning Under Irreversible Dynamics via
Asking for Help" ✓; 2602.12224 = Mirfakhar, Wang, Xu, Beyhaghi, Hajiesmaili,
"Two-Sided Time-Independent Regret for Matching Markets with Limited
Interviews" ✓.

⚠ **Citation caveat for Thm A's framing:** the Plaut abstract foregrounds the
*positive* (mentor) result; the Heaven-or-Hell impossibility we cite lives in
the paper body (正本: Fig. 2). Before submission, quote the exact impossibility
statement from the body — if it is informal there, our §2 lemma is the formal
statement and should be cited as such (self-contained), with Plaut as
inspiration rather than as the source of the theorem.

**Verdict: 検証済み (条項つき).**

---

## Reproduction

```bash
git fetch origin research/theory-verification
git worktree add ../match-learn-verify research/theory-verification
cd ../match-learn-verify && cargo run --release --example theory_verification
```

All assertions pass as of `master` @ 192d4a0 (2026-06-10).
