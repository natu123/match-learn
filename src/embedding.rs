//! Embedding a pivotal near-tie into a general `n × n` market — the bridge that
//! makes the *whole market's* admissible gap equal to one deciding gap.
//!
//! match-learn paper (1) needs the learning difficulty of a full market to be
//! captured by a single number, [`admissible_gap`](crate::admissible). [`embed`]
//! constructs a market where that is provably true:
//!
//! - a **pivotal gadget** `{a*, a_s} × {f₁, f₂}`: `a*` near-ties `f₁, f₂` at gap
//!   `Δ_A` (utilities `½ ± Δ_A/2`); both firms rank `a*` top and the spare `a_s`
//!   second; `a_s` prefers `f₁` by a wide margin but is outranked by `a*`. So the
//!   stable matching gives `a*` its truly-better firm and `a_s` the other, and
//!   swapping `a*`'s `f₁/f₂` order (the only near-tie) flips the stable matching —
//!   a pivotal, gap-`Δ_A` decision.
//! - a **rigid core** `{a₃..} × {f₃..}`: an aligned diagonal with every gap
//!   `≥ Δ_big ≫ Δ_A`, so it has a unique, easily-resolved stable matching and
//!   contributes nothing to the difficulty.
//!
//! Every *other* gap that could change the matching — the firms' `a*`-over-`a_s`
//! margin, the core's gaps — is set to `Δ_big`, so the **smallest gap whose
//! blurring breaks (super-)stability is `a*`'s `Δ_A`**. Hence
//! `admissible_gap(market) == Δ_A`, *independent of `Δ_big`*: the wide core gaps
//! are "free" (outcome-relative — Basu's admissibility), and the value the
//! `admissible` utility computes on the whole market is exactly the pivotal gap.
//! This is the structural half of the theory-parameter = computed-object =
//! measured-driver trinity; [`simulate_market`] is the regret (measured-driver)
//! half, lifting the four-regime 2×2 onto this market in *stable regret* — and
//! showing the regret is driven by the pivot `Δ_A` and invariant to the core.

use crate::irreversible::Policy;
use crate::matching::gale_shapley;
use crate::prefs::rank_by_scores;
use crate::rng::Rng;

/// Build the embedding market for the given pivotal gap `delta_a`, core gap
/// `delta_big`, and size `n` (agents = firms = `n`, with `n ≥ 2`; `n ≥ 3` gives a
/// non-empty rigid core). `instance` swaps `a*`'s `f₁/f₂` ordering. Returns
/// `(agent_utilities, firm_utilities)`, both `n × n` cardinal in `[0, 1]`.
///
/// Requires `delta_a < delta_big` and `delta_big ≤ 0.9 / (n - 1)` (so the core's
/// descending utilities stay in range).
pub fn embed(
    delta_a: f64,
    delta_big: f64,
    n: usize,
    instance: bool,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    assert!(n >= 2, "the gadget needs two agents and two firms");
    assert!(
        delta_a < delta_big,
        "the pivot must be the narrowest deciding gap"
    );
    assert!(
        delta_big * (n as f64 - 1.0) <= 0.9 + 1e-12,
        "delta_big too large for n: core utilities would go negative"
    );

    let mut prop = vec![vec![0.0; n]; n];
    let mut recv = vec![vec![0.0; n]; n];

    // a* (agent 0): a near-tie between f₁ (firm 0) and f₂ (firm 1).
    let (u1, u2) = if instance {
        (0.5 - delta_a / 2.0, 0.5 + delta_a / 2.0)
    } else {
        (0.5 + delta_a / 2.0, 0.5 - delta_a / 2.0)
    };
    prop[0][0] = u1;
    prop[0][1] = u2;
    for f in prop[0].iter_mut().take(n).skip(2) {
        *f = 0.1; // a* does not want core firms
    }

    // a_s (agent 1): a wide f₁ ≻ f₂ preference, but outranked by a* at both.
    prop[1][0] = 0.8;
    prop[1][1] = 0.2;
    for f in prop[1].iter_mut().take(n).skip(2) {
        *f = 0.1;
    }

    // Core agent i (≥ 2): strictly tops firm i, every gap ≥ delta_big.
    for (i, row) in prop.iter_mut().enumerate().skip(2) {
        let mut rank = 0.0;
        for (f, val) in row.iter_mut().enumerate() {
            *val = if f == i {
                0.9
            } else {
                rank += 1.0;
                0.9 - delta_big * rank
            };
        }
    }

    // Gadget firms f₁, f₂: a* top (0.9), a_s second (gap delta_big), core low.
    for row in recv.iter_mut().take(2) {
        row[0] = 0.9;
        row[1] = 0.9 - delta_big;
        for a in row.iter_mut().skip(2) {
            *a = 0.1;
        }
    }

    // Core firm i (≥ 2): strictly tops agent i, every gap ≥ delta_big.
    for (i, row) in recv.iter_mut().enumerate().skip(2) {
        let mut rank = 0.0;
        for (a, val) in row.iter_mut().enumerate() {
            *val = if a == i {
                0.9
            } else {
                rank += 1.0;
                0.9 - delta_big * rank
            };
        }
    }

    (prop, recv)
}

/// The outcome of an embedding-market learning run.
#[derive(Debug, Clone)]
pub struct MarketOutcome {
    /// Cumulative **stable regret** after each round: a round counts `1` whenever
    /// the realized matching differs from the true stable matching.
    pub stable_regret: Vec<f64>,
    /// Whether the final round's matching is the true stable matching.
    pub ended_stable: bool,
}

impl MarketOutcome {
    /// The final cumulative stable regret.
    pub fn final_regret(&self) -> f64 {
        self.stable_regret.last().copied().unwrap_or(0.0)
    }
}

/// Run the four-regime learning experiment on a market — agents learn their firm
/// utilities `prop_utils`; firms have fixed known rankings `recv_ranks` — and
/// measure **stable regret** against the true stable matching: each round the
/// realized matching is compared to the benchmark, counting `1` when they differ.
///
/// `reversible` and `policy` select the regime exactly as in
/// [`crate::irreversible`], but here the decision is the *whole market's*
/// matching, formed by Gale-Shapley on current beliefs (recoverable: recomputed
/// each round; irreversible: committed once and absorbing). On the [`embed`]
/// market this lifts the 2×2 to a general market, with the regret driven by the
/// single pivotal gap and independent of the rigid core.
pub fn simulate_market(
    prop_utils: &[Vec<f64>],
    recv_ranks: &[Vec<usize>],
    reversible: bool,
    policy: Policy,
    horizon: usize,
    seed: u64,
) -> MarketOutcome {
    let mut rng = Rng::new(seed);
    let n = prop_utils.len();
    let m = recv_ranks.len();
    let true_ranks: Vec<Vec<usize>> = prop_utils.iter().map(|u| rank_by_scores(u)).collect();
    let benchmark = gale_shapley(&true_ranks, recv_ranks).proposer;

    let mut sum = vec![vec![0.0; m]; n];
    let mut cnt = vec![vec![0usize; m]; n];
    // Tiny per-pair noise so an uninformed ranking is effectively random by seed.
    let noise: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..m).map(|_| rng.uniform() * 1e-6).collect())
        .collect();

    // Confidence the interviewer demands before committing: constant when matches
    // are recoverable, but ~1/T under irreversibility (an absorbing commit must be
    // near-certain), which is what makes the irreversible interviewer pay log T.
    let delta = if reversible {
        0.05
    } else {
        1.0 / horizon as f64
    };
    let conf = (2.0 * m as f64 / delta).ln();

    let mut committed: Option<Vec<Option<usize>>> = None;
    let mut regret = Vec::with_capacity(horizon);
    let mut cum = 0.0;
    let mut ended_stable = false;

    for t in 0..horizon {
        // Interview: safe round-robin samples while still unmatched.
        if let (Policy::Interview { per_round }, None) = (policy, &committed) {
            for a in 0..n {
                for _ in 0..per_round {
                    let f = least_sampled(&cnt[a]);
                    sum[a][f] += bernoulli(&mut rng, prop_utils[a][f]);
                    cnt[a][f] += 1;
                }
            }
        }

        // Form this round's realized matching.
        let realized: Vec<Option<usize>> = if let Some(mat) = &committed {
            mat.clone()
        } else if reversible {
            // Recompute Gale-Shapley on current beliefs every round.
            let ranks = belief_ranks(&sum, &cnt, &noise, policy, t);
            gale_shapley(&ranks, recv_ranks).proposer
        } else {
            match policy {
                // No safe signal before an absorbing commit: commit blind now.
                Policy::NoInterview => {
                    let ranks: Vec<Vec<usize>> =
                        noise.iter().map(|row| rank_by_scores(row)).collect();
                    let mat = gale_shapley(&ranks, recv_ranks).proposer;
                    committed = Some(mat.clone());
                    mat
                }
                // Interview until every agent's top is certified, then commit once.
                Policy::Interview { .. } => {
                    if (0..n).all(|a| resolved(&sum[a], &cnt[a], conf)) {
                        let ranks = belief_ranks(&sum, &cnt, &noise, policy, t);
                        let mat = gale_shapley(&ranks, recv_ranks).proposer;
                        committed = Some(mat.clone());
                        mat
                    } else {
                        vec![None; n] // still unmatched
                    }
                }
            }
        };

        // Recoverable bandit: learn from the firm you were matched to this round.
        if reversible && matches!(policy, Policy::NoInterview) {
            for (a, &slot) in realized.iter().enumerate() {
                if let Some(f) = slot {
                    sum[a][f] += bernoulli(&mut rng, prop_utils[a][f]);
                    cnt[a][f] += 1;
                }
            }
        }

        ended_stable = realized == benchmark;
        cum += if ended_stable { 0.0 } else { 1.0 };
        regret.push(cum);
    }

    MarketOutcome {
        stable_regret: regret,
        ended_stable,
    }
}

fn bernoulli(rng: &mut Rng, p: f64) -> f64 {
    if rng.uniform() < p { 1.0 } else { 0.0 }
}

fn least_sampled(cnt: &[usize]) -> usize {
    (0..cnt.len()).min_by_key(|&f| cnt[f]).unwrap()
}

/// Per-agent belief rankings for Gale-Shapley: empirical means (interviews) or
/// UCB scores (the recoverable bandit), with unsampled firms broken by `noise`.
fn belief_ranks(
    sum: &[Vec<f64>],
    cnt: &[Vec<usize>],
    noise: &[Vec<f64>],
    policy: Policy,
    t: usize,
) -> Vec<Vec<usize>> {
    sum.iter()
        .zip(cnt)
        .zip(noise)
        .map(|((srow, crow), nrow)| {
            let score: Vec<f64> = srow
                .iter()
                .zip(crow)
                .zip(nrow)
                .map(|((&s, &c), &noise)| {
                    if c == 0 {
                        return match policy {
                            Policy::NoInterview => f64::INFINITY, // pull unsampled first
                            Policy::Interview { .. } => noise,
                        };
                    }
                    let mean = s / c as f64;
                    match policy {
                        Policy::NoInterview => {
                            mean + (2.0 * ((t + 1) as f64).ln() / c as f64).sqrt()
                        }
                        Policy::Interview { .. } => mean,
                    }
                })
                .collect();
            rank_by_scores(&score)
        })
        .collect()
}

/// Is the agent's best firm certified above every other at confidence `conf`?
fn resolved(sum: &[f64], cnt: &[usize], conf: f64) -> bool {
    if cnt.contains(&0) {
        return false;
    }
    let mean = |f: usize| sum[f] / cnt[f] as f64;
    let radius = |f: usize| (conf / (2.0 * cnt[f] as f64)).sqrt();
    let b = (0..cnt.len())
        .max_by(|&i, &j| mean(i).partial_cmp(&mean(j)).unwrap())
        .unwrap();
    let lcb = mean(b) - radius(b);
    let ucb_rest = (0..cnt.len())
        .filter(|&f| f != b)
        .map(|f| mean(f) + radius(f))
        .fold(f64::NEG_INFINITY, f64::max);
    lcb >= ucb_rest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admissible::admissible_gap;
    use crate::irreversible::Policy;
    use crate::matching::{all_stable_matchings, gale_shapley};
    use crate::prefs::rank_by_scores;

    fn rankings(utils: &[Vec<f64>]) -> Vec<Vec<usize>> {
        utils.iter().map(|row| rank_by_scores(row)).collect()
    }

    /// Mean final stable regret over `seeds` runs of one regime on a 4×4 embedding.
    fn mean_final(
        reversible: bool,
        policy: Policy,
        delta_a: f64,
        delta_big: f64,
        horizon: usize,
        seeds: u64,
    ) -> f64 {
        let (prop, recv) = embed(delta_a, delta_big, 4, false);
        let recv_ranks = rankings(&recv);
        (1..=seeds)
            .map(|s| {
                simulate_market(&prop, &recv_ranks, reversible, policy, horizon, s).final_regret()
            })
            .sum::<f64>()
            / seeds as f64
    }

    /// The headline link: the whole market's admissible gap is exactly the
    /// pivotal near-tie `Δ_A` — for every size, instance, and core gap.
    #[test]
    fn admissible_gap_equals_the_pivotal_gap() {
        for &n in &[2usize, 3, 4, 5] {
            for &delta_a in &[0.02, 0.05, 0.1] {
                for &delta_big in &[0.2, 0.25] {
                    if delta_big * (n as f64 - 1.0) > 0.9 {
                        continue;
                    }
                    for instance in [false, true] {
                        let (prop, recv) = embed(delta_a, delta_big, n, instance);
                        let da = admissible_gap(&prop, &recv);
                        assert!(
                            (da - delta_a).abs() < 1e-9,
                            "n={n} Δ_A={delta_a} Δ_big={delta_big} inst={instance}: got {da}"
                        );
                    }
                }
            }
        }
    }

    /// Outcome-relativity: widening the rigid core (any `Δ_big ≫ Δ_A`) leaves the
    /// admissible gap pinned at `Δ_A`. The core gaps are free.
    #[test]
    fn admissible_gap_is_invariant_to_the_core_gap() {
        let delta_a = 0.05;
        let mut seen = Vec::new();
        for &delta_big in &[0.15, 0.2, 0.25, 0.3] {
            let (prop, recv) = embed(delta_a, delta_big, 4, false);
            seen.push(admissible_gap(&prop, &recv));
        }
        for da in &seen {
            assert!((da - delta_a).abs() < 1e-9, "not invariant: {seen:?}");
        }
    }

    /// The market has a unique stable matching, and swapping the pivot (instance
    /// I vs II) flips it — but only inside the gadget; the rigid core is untouched.
    #[test]
    fn stable_matching_is_unique_and_swings_on_the_pivot() {
        let n = 5;
        let (p1, r1) = embed(0.05, 0.2, n, false);
        let (p2, r2) = embed(0.05, 0.2, n, true);
        let (pr1, rr1) = (rankings(&p1), rankings(&r1));
        let (pr2, rr2) = (rankings(&p2), rankings(&r2));

        // Uniqueness.
        assert_eq!(all_stable_matchings(&pr1, &rr1).len(), 1);
        assert_eq!(all_stable_matchings(&pr2, &rr2).len(), 1);

        let m1 = gale_shapley(&pr1, &rr1);
        let m2 = gale_shapley(&pr2, &rr2);

        // a* (0) and a_s (1) swap their firms between the two instances.
        assert_eq!(m1.proposer[0], Some(0)); // a* → f₁
        assert_eq!(m1.proposer[1], Some(1)); // a_s → f₂
        assert_eq!(m2.proposer[0], Some(1)); // a* → f₂ (swung)
        assert_eq!(m2.proposer[1], Some(0)); // a_s → f₁

        // The rigid core (agents 2..n) is identical in both instances.
        for i in 2..n {
            assert_eq!(m1.proposer[i], Some(i));
            assert_eq!(m2.proposer[i], Some(i));
        }
    }

    /// The 2×2 lifts to the general market in stable regret: only irreversible +
    /// no-interview is linear in `T`; the other three regimes are sublinear, and
    /// recoverable + interview is a horizon-free constant far below the log-`T`
    /// irreversible interviewer.
    #[test]
    fn lifts_the_2x2_only_irreversible_no_interview_is_linear() {
        let iv = Policy::Interview { per_round: 2 };
        let growth = |rev, pol| {
            let r1 = mean_final(rev, pol, 0.15, 0.25, 4000, 12);
            let r2 = mean_final(rev, pol, 0.15, 0.25, 16000, 12);
            (r2, r2 / r1)
        };
        let (_, irrev_noint) = growth(false, Policy::NoInterview);
        let (irrev_int_reg, irrev_int) = growth(false, iv);
        let (_, rev_noint) = growth(true, Policy::NoInterview);
        let (rev_int_reg, rev_int) = growth(true, iv);

        assert!(
            irrev_noint > 3.0,
            "irrev/no-int should be linear: {irrev_noint}"
        );
        assert!(
            irrev_int < 2.5,
            "irrev/int should be sublinear: {irrev_int}"
        );
        assert!(
            rev_noint < 2.5,
            "rev/no-int should be sublinear: {rev_noint}"
        );
        assert!(rev_int < 1.2, "rev/int should be flat: {rev_int}");
        // Recoverable + interview is O(1): far below the log-T irreversible one.
        assert!(
            rev_int_reg * 5.0 < irrev_int_reg,
            "rev/int (O(1)) should be far below irrev/int (log T): {rev_int_reg} vs {irrev_int_reg}"
        );
    }

    /// The irreversible interviewer's stable regret scales like `1/Δ_A²`: halving
    /// the pivot roughly quadruples it.
    #[test]
    fn irreversible_interview_regret_scales_inverse_square_in_the_pivot() {
        let iv = Policy::Interview { per_round: 2 };
        let wide = mean_final(false, iv, 0.2, 0.25, 8000, 12);
        let narrow = mean_final(false, iv, 0.1, 0.25, 8000, 12);
        let ratio = narrow / wide;
        assert!((2.8..5.0).contains(&ratio), "inverse-square off: {ratio}");
    }

    /// Outcome-relativity in regret: widening the rigid core leaves the regret
    /// unchanged — the difficulty is the pivot `Δ_A`, not the core gap `Δ_big`.
    #[test]
    fn irreversible_interview_regret_is_invariant_to_the_core_gap() {
        let iv = Policy::Interview { per_round: 2 };
        let narrow_core = mean_final(false, iv, 0.15, 0.2, 6000, 12);
        let wide_core = mean_final(false, iv, 0.15, 0.3, 6000, 12);
        let ratio = wide_core / narrow_core;
        assert!(
            (0.85..1.18).contains(&ratio),
            "core gap changed the regret: {ratio}"
        );
    }
}
