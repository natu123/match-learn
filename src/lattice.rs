//! The lattice of stable matchings, and median stable matchings.
//!
//! The stable matchings of a market are not an unstructured pile: ordered by
//! "every proposer is at least as happy", they form a **distributive lattice**
//! (Conway). Its top is the proposer-optimal matching ([Gale-Shapley](crate::gale_shapley)),
//! its bottom the receiver-optimal one, and any two stable matchings have a
//! **join** (each proposer keeps the better of its two partners) and a **meet**
//! (each keeps the worse) that are *themselves* stable.
//!
//! That structure is what lets matching escape Gale-Shapley's one-sidedness. The
//! proposer-optimal matching is best possible for every proposer *simultaneously*
//! and worst for every receiver — a systematic bias toward whoever proposes. The
//! **median stable matching** (Teo-Sethuraman) is the principled compromise: take
//! each proposer's *median*-ranked partner across all stable matchings, and the
//! result is, remarkably, a stable matching. This module builds the lattice
//! operations and the generalized medians that interpolate between the two
//! extremes, complementing the egalitarian and sex-equal matchings in
//! [`fairness`](crate::fairness).
//!
//! These results are stated for the classic marriage model — **complete strict
//! preferences with equally many proposers and receivers**, where every stable
//! matching is perfect — and the functions here assume that setting.

use crate::matching::{Matching, all_stable_matchings, rank_table};

/// Rank of `partner` for a proposer with rank row `prop_rank_p` (lower = more
/// preferred). Being unmatched is worst of all.
fn partner_rank(prop_rank_p: &[Option<usize>], partner: Option<usize>) -> usize {
    match partner {
        Some(r) => prop_rank_p[r].unwrap_or(usize::MAX - 1),
        None => usize::MAX,
    }
}

/// Build the receiver view from a proposer assignment, or `None` if two
/// proposers claim the same receiver (an invalid matching). For genuine lattice
/// operations on stable matchings this never fails.
fn matching_from_proposer(proposer: Vec<Option<usize>>, n_r: usize) -> Option<Matching> {
    let mut receiver = vec![None; n_r];
    for (p, &slot) in proposer.iter().enumerate() {
        if let Some(r) = slot {
            if receiver[r].is_some() {
                return None;
            }
            receiver[r] = Some(p);
        }
    }
    Some(Matching { proposer, receiver })
}

/// Combine two matchings proposer-by-proposer, taking the better partner of each
/// when `take_better`, the worse otherwise.
fn combine(
    a: &Matching,
    b: &Matching,
    proposer_prefs: &[Vec<usize>],
    take_better: bool,
) -> Option<Matching> {
    let n_r = a.receiver.len();
    let prop_rank = rank_table(proposer_prefs, n_r);
    let proposer: Vec<Option<usize>> = (0..proposer_prefs.len())
        .map(|p| {
            let ra = partner_rank(&prop_rank[p], a.proposer[p]);
            let rb = partner_rank(&prop_rank[p], b.proposer[p]);
            let pick_a = if take_better { ra <= rb } else { ra >= rb };
            if pick_a { a.proposer[p] } else { b.proposer[p] }
        })
        .collect();
    matching_from_proposer(proposer, n_r)
}

/// The lattice **join** of two stable matchings: every proposer keeps the better
/// of its two partners. By Conway's theorem the result is again a stable matching
/// (and the *receiver*-pessimal of the two). `None` only if the inputs are not
/// two matchings of the same market.
///
/// ```
/// use match_learn::{gale_shapley, lattice::stable_join};
///
/// // Two proposers, both prefer receiver 0; receiver 0 prefers proposer 1.
/// let prop = vec![vec![0, 1], vec![0, 1]];
/// let recv = vec![vec![1, 0], vec![1, 0]];
/// let gs = gale_shapley(&prop, &recv);
/// // Join with itself is itself.
/// assert_eq!(stable_join(&gs, &gs, &prop), Some(gs));
/// ```
pub fn stable_join(a: &Matching, b: &Matching, proposer_prefs: &[Vec<usize>]) -> Option<Matching> {
    combine(a, b, proposer_prefs, true)
}

/// The lattice **meet** of two stable matchings: every proposer keeps the worse
/// of its two partners. Again a stable matching, and the proposer-pessimal of the
/// two. See [`stable_join`].
pub fn stable_meet(a: &Matching, b: &Matching, proposer_prefs: &[Vec<usize>]) -> Option<Matching> {
    combine(a, b, proposer_prefs, false)
}

/// The `k` generalized median stable matchings (Teo-Sethuraman), given the full
/// set of `k` stable matchings.
///
/// For each proposer, list its partners across the `k` stable matchings and sort
/// them from most to least preferred; the `i`-th returned matching assigns every
/// proposer its `i`-th entry. The `i = 0` matching is proposer-optimal, the
/// `i = k-1` receiver-optimal, and each one in between is itself a stable
/// matching — a fairness dial from one side's optimum to the other's.
///
/// Returns `k` matchings (some may coincide). Assumes the classic marriage model
/// (complete strict preferences, equal sides); the stability of every median is
/// the Teo-Sethuraman theorem, checked exhaustively against the brute-force
/// stable set in this crate's tests.
pub fn generalized_medians(stable: &[Matching], proposer_prefs: &[Vec<usize>]) -> Vec<Matching> {
    let k = stable.len();
    if k == 0 {
        return Vec::new();
    }
    let n_p = proposer_prefs.len();
    let n_r = stable[0].receiver.len();
    let prop_rank = rank_table(proposer_prefs, n_r);

    // Each proposer's partners across the stable set, sorted best-first.
    let sorted: Vec<Vec<Option<usize>>> = (0..n_p)
        .map(|p| {
            let mut ps: Vec<Option<usize>> = stable.iter().map(|m| m.proposer[p]).collect();
            ps.sort_by_key(|&partner| partner_rank(&prop_rank[p], partner));
            ps
        })
        .collect();

    (0..k)
        .map(|i| {
            let proposer: Vec<Option<usize>> = (0..n_p).map(|p| sorted[p][i]).collect();
            matching_from_proposer(proposer, n_r)
                .expect("a generalized median is a valid matching (Teo-Sethuraman)")
        })
        .collect()
}

/// The median stable matching: the middle [generalized median](generalized_medians),
/// the balanced compromise between the proposer- and receiver-optimal extremes.
/// `None` only if no stable matching exists (never for the marriage model).
///
/// ```
/// use match_learn::lattice::median_stable_matching;
///
/// // A symmetric market with a unique stable matching: the median is it.
/// let prop = vec![vec![0, 1], vec![1, 0]];
/// let recv = vec![vec![0, 1], vec![1, 0]];
/// let m = median_stable_matching(&prop, &recv).unwrap();
/// assert_eq!(m.proposer, vec![Some(0), Some(1)]);
/// ```
pub fn median_stable_matching(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
) -> Option<Matching> {
    let stable = all_stable_matchings(proposer_prefs, receiver_prefs);
    let medians = generalized_medians(&stable, proposer_prefs);
    if medians.is_empty() {
        return None;
    }
    let mid = (medians.len() - 1) / 2;
    Some(medians[mid].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{gale_shapley, is_stable};
    use crate::rng::Rng;

    /// A random small classic-marriage-model market (complete strict preferences,
    /// equal sides), the domain where the lattice and median theorems hold.
    fn random_market(rng: &mut Rng) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let n = 1 + rng.below(4);
        let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
        let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
        (prop, recv)
    }

    #[test]
    fn join_and_meet_of_stable_pairs_are_stable() {
        let mut rng = Rng::new(0x1A77);
        for _ in 0..1000 {
            let (prop, recv) = random_market(&mut rng);
            let stable = all_stable_matchings(&prop, &recv);
            for a in &stable {
                for b in &stable {
                    let j = stable_join(a, b, &prop).expect("join is a matching");
                    let m = stable_meet(a, b, &prop).expect("meet is a matching");
                    assert!(is_stable(&prop, &recv, &j), "join not stable");
                    assert!(is_stable(&prop, &recv, &m), "meet not stable");
                    // Closure: the lattice operations stay inside the stable set.
                    assert!(stable.contains(&j) && stable.contains(&m));
                }
            }
        }
    }

    #[test]
    fn folding_join_recovers_gale_shapley() {
        // The join over the whole stable set is the proposer-optimal matching.
        let mut rng = Rng::new(0x2B88);
        for _ in 0..500 {
            let (prop, recv) = random_market(&mut rng);
            let stable = all_stable_matchings(&prop, &recv);
            let top = stable.iter().skip(1).fold(stable[0].clone(), |acc, m| {
                stable_join(&acc, m, &prop).unwrap()
            });
            assert_eq!(top, gale_shapley(&prop, &recv));
        }
    }

    #[test]
    fn generalized_medians_are_valid_and_stable() {
        let mut rng = Rng::new(0x3C99);
        for _ in 0..1500 {
            let (prop, recv) = random_market(&mut rng);
            let stable = all_stable_matchings(&prop, &recv);
            if stable.is_empty() {
                continue;
            }
            let medians = generalized_medians(&stable, &prop);
            assert_eq!(medians.len(), stable.len());
            for m in &medians {
                assert!(
                    is_stable(&prop, &recv, m),
                    "median not stable: prop={prop:?} recv={recv:?} m={m:?}"
                );
            }
        }
    }

    #[test]
    fn medians_interpolate_between_the_two_optima() {
        // First median = proposer-optimal (= GS); last = receiver-optimal.
        let mut rng = Rng::new(0x4D10);
        for _ in 0..500 {
            let (prop, recv) = random_market(&mut rng);
            let stable = all_stable_matchings(&prop, &recv);
            let medians = generalized_medians(&stable, &prop);
            assert_eq!(medians[0], gale_shapley(&prop, &recv));
            // The receiver-optimal matching is GS with the sides swapped.
            let swapped = gale_shapley(&recv, &prop);
            let recv_opt = Matching {
                proposer: swapped.receiver,
                receiver: swapped.proposer,
            };
            assert_eq!(*medians.last().unwrap(), recv_opt);
        }
    }

    #[test]
    fn finds_a_genuine_interior_median() {
        // Search for an instance with >= 3 stable matchings and confirm the
        // median is stable and strictly between the two optima for someone.
        let mut rng = Rng::new(0x5E21);
        let mut seen_interior = false;
        for _ in 0..4000 {
            let (prop, recv) = random_market(&mut rng);
            let stable = all_stable_matchings(&prop, &recv);
            if stable.len() < 3 {
                continue;
            }
            let med = median_stable_matching(&prop, &recv).unwrap();
            assert!(is_stable(&prop, &recv, &med));
            let top = gale_shapley(&prop, &recv);
            let swapped = gale_shapley(&recv, &prop);
            let bottom = Matching {
                proposer: swapped.receiver,
                receiver: swapped.proposer,
            };
            if med != top && med != bottom {
                seen_interior = true;
            }
        }
        assert!(
            seen_interior,
            "expected at least one instance with a strict interior median"
        );
    }

    #[test]
    fn single_stable_matching_is_its_own_median() {
        let prop = vec![vec![0, 1], vec![1, 0]];
        let recv = vec![vec![0, 1], vec![1, 0]];
        let med = median_stable_matching(&prop, &recv).unwrap();
        assert_eq!(med, gale_shapley(&prop, &recv));
    }
}
