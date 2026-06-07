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
//! measured-driver trinity (the regret half is the irreversible simulation).

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admissible::admissible_gap;
    use crate::matching::{all_stable_matchings, gale_shapley};
    use crate::prefs::rank_by_scores;

    fn rankings(utils: &[Vec<f64>]) -> Vec<Vec<usize>> {
        utils.iter().map(|row| rank_by_scores(row)).collect()
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
}
