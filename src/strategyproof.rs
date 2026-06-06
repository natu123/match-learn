//! Strategy-proofness: can an agent gain by lying about its preferences?
//!
//! Gale-Shapley is **strategy-proof for the proposing side** (Dubins-Freedman,
//! Roth): no proposer can ever do better — by its *true* preferences — by
//! reporting a different list. It is **not** strategy-proof for receivers: a
//! receiver can sometimes manipulate the proposing side into a partner it prefers.
//! This module checks both, by brute-forcing all misreports (small instances).

/// Position of `partner` on `prefs` (0 = best); `prefs.len()` if unmatched.
fn rank(prefs: &[usize], partner: Option<usize>) -> usize {
    match partner {
        Some(p) => prefs.iter().position(|&x| x == p).unwrap_or(prefs.len()),
        None => prefs.len(),
    }
}

/// All permutations of `items` (small slices only).
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

/// A profitable misreport for proposer `manipulator`, if one exists.
///
/// Returns a reported preference list that yields a partner the manipulator
/// *truly* prefers to its truthful Gale-Shapley partner, or `None` if none does.
/// For the proposing side this is always `None` (the theorem); the function is
/// the empirical witness.
pub fn proposer_manipulation(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
    manipulator: usize,
) -> Option<Vec<usize>> {
    let truthful = crate::gale_shapley(proposer_prefs, receiver_prefs);
    let true_rank = rank(&proposer_prefs[manipulator], truthful.proposer[manipulator]);

    for report in permutations(&proposer_prefs[manipulator]) {
        let mut prefs = proposer_prefs.to_vec();
        prefs[manipulator] = report.clone();
        let m = crate::gale_shapley(&prefs, receiver_prefs);
        // Judge the new partner by the manipulator's TRUE preferences.
        let new_rank = rank(&proposer_prefs[manipulator], m.proposer[manipulator]);
        if new_rank < true_rank {
            return Some(report);
        }
    }
    None
}

/// A profitable misreport for receiver `manipulator`, if one exists.
///
/// Returns a reported preference list that yields a partner the receiver *truly*
/// prefers to its truthful partner, or `None`. Unlike proposers, receivers can
/// sometimes manipulate proposer-proposing Gale-Shapley.
pub fn receiver_manipulation(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
    manipulator: usize,
) -> Option<Vec<usize>> {
    let truthful = crate::gale_shapley(proposer_prefs, receiver_prefs);
    let true_rank = rank(&receiver_prefs[manipulator], truthful.receiver[manipulator]);

    for report in permutations(&receiver_prefs[manipulator]) {
        let mut prefs = receiver_prefs.to_vec();
        prefs[manipulator] = report.clone();
        let m = crate::gale_shapley(proposer_prefs, &prefs);
        let new_rank = rank(&receiver_prefs[manipulator], m.receiver[manipulator]);
        if new_rank < true_rank {
            return Some(report);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn proposers_can_never_manipulate() {
        // The Dubins-Freedman / Roth theorem, verified empirically: across random
        // complete instances, no proposer has a profitable misreport.
        let mut rng = Rng::new(2026);
        for _ in 0..300 {
            let n = 2 + rng.below(4); // 2..=5
            let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            for p in 0..n {
                assert!(
                    proposer_manipulation(&prop, &recv, p).is_none(),
                    "proposer {p} could manipulate: prop={prop:?} recv={recv:?}"
                );
            }
        }
    }

    #[test]
    fn a_found_receiver_manipulation_is_genuine() {
        // Find an instance where a receiver can manipulate, then verify the
        // reported misorder really does yield a strictly better true partner.
        let mut rng = Rng::new(99);
        for _ in 0..500 {
            let n = 3 + rng.below(3);
            let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            for r in 0..n {
                if let Some(report) = receiver_manipulation(&prop, &recv, r) {
                    let truthful = crate::gale_shapley(&prop, &recv);
                    let true_rank = rank(&recv[r], truthful.receiver[r]);
                    let mut lied = recv.clone();
                    lied[r] = report;
                    let m = crate::gale_shapley(&prop, &lied);
                    let new_rank = rank(&recv[r], m.receiver[r]);
                    assert!(new_rank < true_rank, "claimed manipulation did not improve");
                    return; // found and verified one
                }
            }
        }
        panic!("no receiver manipulation found in 500 instances");
    }

    #[test]
    fn receivers_can_manipulate_more_often_than_proposers() {
        // Over random instances, receivers find manipulations and proposers never
        // do — the asymmetry of proposer-proposing Gale-Shapley.
        let mut rng = Rng::new(7);
        let mut receiver_manips = 0;
        let mut proposer_manips = 0;
        for _ in 0..200 {
            let n = 3 + rng.below(3); // 3..=5
            let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            for r in 0..n {
                if receiver_manipulation(&prop, &recv, r).is_some() {
                    receiver_manips += 1;
                }
                if proposer_manipulation(&prop, &recv, r).is_some() {
                    proposer_manips += 1;
                }
            }
        }
        assert_eq!(proposer_manips, 0, "proposers should never manipulate");
        assert!(receiver_manips > 0, "receivers should sometimes manipulate");
    }
}
