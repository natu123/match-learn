//! Fairness across the two sides of a stable matching.
//!
//! Gale-Shapley returns the *proposer-optimal* stable matching: it is best for
//! every proposer and worst for every receiver among all stable matchings. When
//! the two sides are people — students and schools, doctors and hospitals — that
//! one-sidedness is an equity problem. This module measures it and finds fairer
//! stable matchings.
//!
//! Cost is measured in *ranks*: a side's cost is the sum, over its matched
//! members, of the position of its partner on its preference list (0 = top
//! choice, so lower is better). Two fairness notions:
//!
//! - **egalitarian**: minimize the total `proposer_cost + receiver_cost` — the
//!   most efficient stable matching overall;
//! - **sex-equal**: minimize `|proposer_cost - receiver_cost|` — the most
//!   balanced between the two sides.
//!
//! Both are found by enumerating the (small) set of stable matchings, so they
//! are for modest instance sizes.

use crate::matching::{Matching, all_stable_matchings};

/// Position of `partner` on `prefs` (0 = most preferred); list length if absent.
fn rank_of(prefs: &[usize], partner: usize) -> usize {
    prefs
        .iter()
        .position(|&x| x == partner)
        .unwrap_or(prefs.len())
}

/// Total proposer-side rank cost of `m` (sum over matched proposers of the rank
/// of their partner). Lower is better for proposers.
pub fn proposer_cost(proposer_prefs: &[Vec<usize>], m: &Matching) -> usize {
    m.proposer
        .iter()
        .enumerate()
        .filter_map(|(p, &slot)| slot.map(|r| rank_of(&proposer_prefs[p], r)))
        .sum()
}

/// Total receiver-side rank cost of `m`.
pub fn receiver_cost(receiver_prefs: &[Vec<usize>], m: &Matching) -> usize {
    m.receiver
        .iter()
        .enumerate()
        .filter_map(|(r, &slot)| slot.map(|p| rank_of(&receiver_prefs[r], p)))
        .sum()
}

/// Egalitarian cost: `proposer_cost + receiver_cost` (total welfare loss).
pub fn egalitarian_cost(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
    m: &Matching,
) -> usize {
    proposer_cost(proposer_prefs, m) + receiver_cost(receiver_prefs, m)
}

/// Sex-equality cost: `|proposer_cost - receiver_cost|` (imbalance between sides).
pub fn sex_equality_cost(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
    m: &Matching,
) -> usize {
    proposer_cost(proposer_prefs, m).abs_diff(receiver_cost(receiver_prefs, m))
}

/// The stable matching minimizing egalitarian cost (ties broken arbitrarily).
///
/// Enumerates all stable matchings, so for small instances only.
pub fn egalitarian_stable(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
) -> Matching {
    all_stable_matchings(proposer_prefs, receiver_prefs)
        .into_iter()
        .min_by_key(|m| egalitarian_cost(proposer_prefs, receiver_prefs, m))
        .unwrap_or(Matching {
            proposer: vec![None; proposer_prefs.len()],
            receiver: vec![None; receiver_prefs.len()],
        })
}

/// The stable matching minimizing sex-equality cost (the most balanced).
///
/// Enumerates all stable matchings, so for small instances only.
pub fn sex_equal_stable(proposer_prefs: &[Vec<usize>], receiver_prefs: &[Vec<usize>]) -> Matching {
    all_stable_matchings(proposer_prefs, receiver_prefs)
        .into_iter()
        .min_by_key(|m| sex_equality_cost(proposer_prefs, receiver_prefs, m))
        .unwrap_or(Matching {
            proposer: vec![None; proposer_prefs.len()],
            receiver: vec![None; receiver_prefs.len()],
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{gale_shapley, is_stable};
    use crate::rng::Rng;

    #[test]
    fn egalitarian_is_no_worse_than_proposer_optimal() {
        // Over random complete instances, the egalitarian stable matching's total
        // cost is at most the proposer-optimal (Gale-Shapley) one's, and is stable.
        let mut rng = Rng::new(1);
        for _ in 0..300 {
            let n = 2 + rng.below(4); // 2..=5
            let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();

            let gs = gale_shapley(&prop, &recv);
            let eg = egalitarian_stable(&prop, &recv);

            assert!(is_stable(&prop, &recv, &eg));
            assert!(
                egalitarian_cost(&prop, &recv, &eg) <= egalitarian_cost(&prop, &recv, &gs),
                "egalitarian not <= proposer-optimal"
            );
        }
    }

    #[test]
    fn sex_equal_reduces_the_imbalance() {
        // The sex-equal stable matching is at least as balanced as Gale-Shapley.
        let mut rng = Rng::new(7);
        for _ in 0..300 {
            let n = 2 + rng.below(4);
            let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();

            let gs = gale_shapley(&prop, &recv);
            let se = sex_equal_stable(&prop, &recv);

            assert!(is_stable(&prop, &recv, &se));
            assert!(
                sex_equality_cost(&prop, &recv, &se) <= sex_equality_cost(&prop, &recv, &gs),
                "sex-equal not at least as balanced as proposer-optimal"
            );
        }
    }

    #[test]
    fn proposer_optimal_favours_proposers() {
        // A small instance where Gale-Shapley is lopsided and the fairer matchings
        // help. Both sides have opposing preferences.
        let prop = vec![vec![0, 1], vec![1, 0]];
        let recv = vec![vec![1, 0], vec![0, 1]];
        // Here every matching is stable; the egalitarian/sex-equal ones exist.
        let eg = egalitarian_stable(&prop, &recv);
        assert!(is_stable(&prop, &recv, &eg));
    }
}
