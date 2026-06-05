//! Preference utilities: turning scores or weak (tied) orders into the strict,
//! possibly-incomplete preference lists the matching algorithms consume.
//!
//! Gale-Shapley and Hospital-Residents need *strict* rankings, and they treat a
//! candidate absent from a list as unacceptable. Real preferences often arrive
//! as scores (with exact ties) or as weak orders (tiers of equal candidates),
//! and some candidates are simply unacceptable. These helpers bridge that gap.

use crate::rng::Rng;
use std::cmp::Ordering;

/// Strict ranking from `scores`, most preferred (highest) first; exact ties are
/// broken by index, so the result is deterministic.
pub fn rank_by_scores(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// Strict ranking from `scores`, most preferred first; exact ties are broken
/// uniformly at random using `rng`.
pub fn rank_by_scores_random(scores: &[f64], rng: &mut Rng) -> Vec<usize> {
    // A random tie-break key per index; only consulted when scores are equal.
    let mut keyed: Vec<(usize, f64, f64)> = scores
        .iter()
        .map(|&s| (s, rng.uniform()))
        .enumerate()
        .map(|(i, (s, k))| (i, s, k))
        .collect();
    keyed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
    });
    keyed.into_iter().map(|(i, _, _)| i).collect()
}

/// Flatten a weak order — `tiers` from most to least preferred, each tier a set
/// of equally-preferred candidates — into a strict order, shuffling within each
/// tier. Candidates in an earlier tier always precede those in a later one.
pub fn break_ties(tiers: &[Vec<usize>], rng: &mut Rng) -> Vec<usize> {
    let mut out = Vec::new();
    for tier in tiers {
        let mut t = tier.clone();
        rng.shuffle(&mut t);
        out.extend(t);
    }
    out
}

/// Keep only acceptable candidates, preserving order. `acceptable[c]` says
/// whether candidate `c` is acceptable; the resulting (possibly shorter) list is
/// an *incomplete* preference list.
pub fn restrict_to_acceptable(ranking: &[usize], acceptable: &[bool]) -> Vec<usize> {
    ranking
        .iter()
        .copied()
        .filter(|&c| acceptable.get(c).copied().unwrap_or(false))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::gale_shapley;

    fn is_permutation_of(v: &[usize], n: usize) -> bool {
        let mut s = v.to_vec();
        s.sort_unstable();
        s == (0..n).collect::<Vec<_>>()
    }

    #[test]
    fn rank_by_scores_descending_with_index_ties() {
        let scores = [0.2, 0.9, 0.9, 0.1];
        // 1 and 2 tie at 0.9; index breaks it so 1 before 2.
        assert_eq!(rank_by_scores(&scores), vec![1, 2, 0, 3]);
    }

    #[test]
    fn rank_by_scores_random_is_a_permutation_and_respects_strict_order() {
        let mut rng = Rng::new(3);
        let scores = [0.5, 0.5, 0.5, 0.5];
        let r = rank_by_scores_random(&scores, &mut rng);
        assert!(is_permutation_of(&r, 4));

        // With distinct scores, randomization must not disturb the strict order.
        let scores = [0.1, 0.4, 0.2, 0.9];
        let r = rank_by_scores_random(&scores, &mut rng);
        assert_eq!(r, vec![3, 1, 2, 0]);
    }

    #[test]
    fn random_tie_break_actually_varies() {
        // All-equal scores: different seeds should sometimes produce different
        // top elements (so the tie-break is genuinely random, not fixed).
        let scores = [0.0; 6];
        let mut tops = std::collections::HashSet::new();
        for seed in 0..50 {
            let mut rng = Rng::new(seed);
            tops.insert(rank_by_scores_random(&scores, &mut rng)[0]);
        }
        assert!(tops.len() > 1, "tie-break never varied: {tops:?}");
    }

    #[test]
    fn break_ties_respects_tiers_and_is_a_permutation() {
        let tiers = vec![vec![3, 1], vec![0], vec![2, 4]];
        let mut rng = Rng::new(11);
        let order = break_ties(&tiers, &mut rng);
        assert!(is_permutation_of(&order, 5));
        // Tier 0 members precede tier 1, which precedes tier 2.
        let pos = |x: usize| order.iter().position(|&y| y == x).unwrap();
        assert!(pos(3).max(pos(1)) < pos(0));
        assert!(pos(0) < pos(2).min(pos(4)));
    }

    #[test]
    fn restrict_drops_unacceptable_and_preserves_order() {
        let ranking = vec![2, 0, 3, 1];
        let acceptable = vec![true, false, true, false]; // 0 and 2 acceptable
        assert_eq!(restrict_to_acceptable(&ranking, &acceptable), vec![2, 0]);
    }

    #[test]
    fn incomplete_lists_feed_gale_shapley() {
        // Proposer 0 finds only receiver 1 acceptable; matching must respect it.
        let full = [vec![0, 1], vec![0, 1]];
        let acceptable_p0 = [false, true];
        let p0 = restrict_to_acceptable(&full[0], &acceptable_p0);
        let prop = [p0, full[1].clone()];
        let recv = [vec![0, 1], vec![0, 1]];
        let m = gale_shapley(&prop, &recv);
        // Proposer 0 can only be with receiver 1 (or unmatched), never receiver 0.
        assert_ne!(m.proposer[0], Some(0));
    }
}
