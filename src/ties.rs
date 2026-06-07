//! Stable matching with **ties** (indifferences) in the preference lists.
//!
//! Real markets rarely have strict preferences: a school is indifferent between
//! applicants in the same priority band, an employer between equally-qualified
//! candidates. With ties, "stable" splits into three nested notions (Irving):
//!
//! - **weakly stable** — no pair *both strictly* prefer each other to their
//!   partners. Always exists; obtained by breaking ties arbitrarily and running
//!   Gale-Shapley ([`weakly_stable`]).
//! - **strongly stable** — no pair where one strictly prefers the other and the
//!   other is indifferent-or-better. May not exist.
//! - **super stable** — no pair where each is indifferent-or-better off together.
//!   The strongest; may not exist.
//!
//! Super ⟹ strong ⟹ weak. With no ties all three collapse to ordinary
//! [`is_stable`](crate::matching::is_stable). Preferences are given as tiers:
//! `prefs[a]` is a list of indifference classes, better classes first.

use crate::matching::Matching;

/// `class[a][b] = Some(k)` if `b` is in `a`'s `k`-th indifference class (0 =
/// best), else `None` (unacceptable).
fn class_table(prefs: &[Vec<Vec<usize>>], n_other: usize) -> Vec<Vec<Option<usize>>> {
    let mut table = vec![vec![None; n_other]; prefs.len()];
    for (a, tiers) in prefs.iter().enumerate() {
        for (k, tier) in tiers.iter().enumerate() {
            for &b in tier {
                if b < n_other {
                    table[a][b] = Some(k);
                }
            }
        }
    }
    table
}

/// Does `a` *strictly* prefer `other` to its current partner `cur`?
fn strict(row: &[Option<usize>], other: usize, cur: Option<usize>) -> bool {
    match row[other] {
        None => false, // unacceptable: never preferred
        Some(rk) => match cur {
            None => true, // unmatched: any acceptable partner is strictly better
            Some(c) => row[c].is_none_or(|ck| rk < ck),
        },
    }
}

/// Does `a` *weakly* prefer `other` to `cur` (strictly better or indifferent)?
fn weak(row: &[Option<usize>], other: usize, cur: Option<usize>) -> bool {
    match row[other] {
        None => false,
        Some(rk) => match cur {
            None => true,
            Some(c) => row[c].is_none_or(|ck| rk <= ck),
        },
    }
}

/// Whether `m` is **weakly stable**: no blocking pair where *both* sides
/// strictly prefer each other to their assigned partners.
pub fn is_weakly_stable(
    proposer_prefs: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<Vec<usize>>],
    m: &Matching,
) -> bool {
    let (pc, rc) = (
        class_table(proposer_prefs, receiver_prefs.len()),
        class_table(receiver_prefs, proposer_prefs.len()),
    );
    for (p, prow) in pc.iter().enumerate() {
        for (r, rrow) in rc.iter().enumerate() {
            if prow[r].is_none() || rrow[p].is_none() {
                continue; // not mutually acceptable
            }
            if strict(prow, r, m.proposer[p]) && strict(rrow, p, m.receiver[r]) {
                return false;
            }
        }
    }
    true
}

/// Whether `m` is **strongly stable**: no pair where one side strictly prefers
/// the other while that other is indifferent-or-better off.
pub fn is_strongly_stable(
    proposer_prefs: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<Vec<usize>>],
    m: &Matching,
) -> bool {
    let (pc, rc) = (
        class_table(proposer_prefs, receiver_prefs.len()),
        class_table(receiver_prefs, proposer_prefs.len()),
    );
    for (p, prow) in pc.iter().enumerate() {
        for (r, rrow) in rc.iter().enumerate() {
            if prow[r].is_none() || rrow[p].is_none() {
                continue;
            }
            let (ps, pw) = (strict(prow, r, m.proposer[p]), weak(prow, r, m.proposer[p]));
            let (rs, rw) = (strict(rrow, p, m.receiver[r]), weak(rrow, p, m.receiver[r]));
            if (ps && rw) || (pw && rs) {
                return false;
            }
        }
    }
    true
}

/// Whether `m` is **super stable**: no pair where *each* side is
/// indifferent-or-better off together than with its assigned partner.
pub fn is_super_stable(
    proposer_prefs: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<Vec<usize>>],
    m: &Matching,
) -> bool {
    let (pc, rc) = (
        class_table(proposer_prefs, receiver_prefs.len()),
        class_table(receiver_prefs, proposer_prefs.len()),
    );
    for (p, prow) in pc.iter().enumerate() {
        for (r, rrow) in rc.iter().enumerate() {
            if prow[r].is_none() || rrow[p].is_none() || m.proposer[p] == Some(r) {
                continue; // unacceptable, or already matched (equal, not blocking)
            }
            if weak(prow, r, m.proposer[p]) && weak(rrow, p, m.receiver[r]) {
                return false;
            }
        }
    }
    true
}

/// A weakly-stable matching, always obtainable: break every tie at random and
/// run Gale-Shapley on the resulting strict lists.
///
/// ```
/// use match_learn::ties::{weakly_stable, is_weakly_stable};
/// use match_learn::rng::Rng;
///
/// // Each proposer is indifferent between both receivers (one tied class).
/// let p = vec![vec![vec![0, 1]], vec![vec![0, 1]]];
/// let r = vec![vec![vec![0], vec![1]], vec![vec![1], vec![0]]];
/// let mut rng = Rng::new(1);
/// let m = weakly_stable(&p, &r, &mut rng);
/// assert!(is_weakly_stable(&p, &r, &m));
/// ```
pub fn weakly_stable(
    proposer_prefs: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<Vec<usize>>],
    rng: &mut crate::rng::Rng,
) -> Matching {
    let prop: Vec<Vec<usize>> = proposer_prefs
        .iter()
        .map(|tiers| crate::prefs::break_ties(tiers, rng))
        .collect();
    let recv: Vec<Vec<usize>> = receiver_prefs
        .iter()
        .map(|tiers| crate::prefs::break_ties(tiers, rng))
        .collect();
    crate::matching::gale_shapley(&prop, &recv)
}

/// A super-stable matching if one exists, by brute-force search.
///
/// Exponential — `O((R+1)^P)` — and intended for small instances and as the
/// reference oracle for [`super_stable_irving`], the polynomial algorithm.
/// Returns the first super-stable matching found, or `None` if none exists
/// (super stability can fail).
pub fn super_stable(
    proposer_prefs: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<Vec<usize>>],
) -> Option<Matching> {
    find_matching(proposer_prefs.len(), receiver_prefs.len(), |m| {
        is_super_stable(proposer_prefs, receiver_prefs, m)
    })
}

/// A super-stable matching if one exists, via **Irving's algorithm** (1994) for
/// stable marriage with indifference — the polynomial counterpart to the
/// brute-force [`super_stable`].
///
/// Each free man proposes to his *entire* most-preferred indifference class; a
/// woman deletes every man she ranks *strictly* below a current proposer; a woman
/// left engaged to two or more men breaks all those engagements and deletes her
/// tail (worst) class. When the proposals settle, an emptied list means no
/// super-stable matching exists; otherwise the engagement relation is the
/// proposer-optimal super-stable matching. Runs in `O((P·R)²)` on the tiered
/// lists. With strict lists it reduces to ordinary Gale-Shapley.
///
/// Returns `None` when no super-stable matching exists. The candidate is checked
/// with [`is_super_stable`] before it is returned, so a non-`None` result is
/// always genuinely super-stable; existence agreement with [`super_stable`] is
/// verified by test on complete tiered markets.
pub fn super_stable_irving(
    proposer_prefs: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<Vec<usize>>],
) -> Option<Matching> {
    let n_p = proposer_prefs.len();
    let n_r = receiver_prefs.len();
    let pc = class_table(proposer_prefs, n_r);
    let rc = class_table(receiver_prefs, n_p);

    // `present[p][r]`: the pair is still on both lists (mutually acceptable and
    // not yet deleted). Deleting a pair is symmetric — it leaves both lists.
    let mut present = vec![vec![false; n_r]; n_p];
    for (p, prow) in present.iter_mut().enumerate() {
        for (r, cell) in prow.iter_mut().enumerate() {
            *cell = pc[p][r].is_some() && rc[r][p].is_some();
        }
    }
    // `engaged[p][r]`: a provisional, many-to-many engagement during the run.
    let mut engaged = vec![vec![false; n_r]; n_p];

    loop {
        // Proposal phase: every free man proposes to his whole head class.
        while let Some(m) =
            (0..n_p).find(|&p| engaged[p].iter().all(|&e| !e) && present[p].contains(&true))
        {
            let head_k = (0..n_r)
                .filter(|&r| present[m][r])
                .map(|r| pc[m][r].unwrap())
                .min()
                .unwrap();
            let heads: Vec<usize> = (0..n_r)
                .filter(|&r| present[m][r] && pc[m][r] == Some(head_k))
                .collect();
            for r in heads {
                engaged[m][r] = true;
                let m_rank = rc[r][m].unwrap();
                for m2 in 0..n_p {
                    if m2 != m && present[m2][r] && rc[r][m2].is_some_and(|k| k > m_rank) {
                        present[m2][r] = false;
                        engaged[m2][r] = false;
                    }
                }
            }
        }

        // Resolve every multiply-engaged woman: break her engagements and delete
        // her tail (worst) class — the super-blocking risk the proposals exposed.
        let mut changed = false;
        for r in 0..n_r {
            if (0..n_p).filter(|&p| engaged[p][r]).count() <= 1 {
                continue;
            }
            for prow in engaged.iter_mut() {
                prow[r] = false;
            }
            let tail_k = (0..n_p)
                .filter(|&p| present[p][r])
                .map(|p| rc[r][p].unwrap())
                .max()
                .unwrap();
            for p in 0..n_p {
                if present[p][r] && rc[r][p] == Some(tail_k) {
                    present[p][r] = false;
                }
            }
            changed = true;
        }
        if !changed {
            break;
        }
    }

    // Extraction. A man with an empty list is left unmatched; a man still engaged
    // to two or more women cannot be resolved, so no super-stable matching exists.
    let mut proposer = vec![None; n_p];
    let mut receiver = vec![None; n_r];
    for (p, prow) in engaged.iter().enumerate() {
        let mut mine = (0..n_r).filter(|&r| prow[r]);
        match (mine.next(), mine.next()) {
            (Some(r), None) => {
                proposer[p] = Some(r);
                receiver[r] = Some(p);
            }
            (None, _) => {}                    // unmatched: the list emptied
            (Some(_), Some(_)) => return None, // multiply-engaged man
        }
    }
    let m = Matching { proposer, receiver };
    is_super_stable(proposer_prefs, receiver_prefs, &m).then_some(m)
}

/// A strongly-stable matching if one exists, by brute-force search (see
/// [`super_stable`] for the complexity caveat).
pub fn strongly_stable(
    proposer_prefs: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<Vec<usize>>],
) -> Option<Matching> {
    find_matching(proposer_prefs.len(), receiver_prefs.len(), |m| {
        is_strongly_stable(proposer_prefs, receiver_prefs, m)
    })
}

/// Enumerate injective proposer→receiver assignments and return the first whose
/// matching satisfies `pred`.
fn find_matching<F: Fn(&Matching) -> bool>(n_p: usize, n_r: usize, pred: F) -> Option<Matching> {
    let mut assign = vec![0usize; n_p]; // value n_r == unmatched
    loop {
        let mut receiver = vec![None; n_r];
        let mut ok = true;
        for (p, &a) in assign.iter().enumerate() {
            if a < n_r {
                if receiver[a].is_some() {
                    ok = false;
                    break;
                }
                receiver[a] = Some(p);
            }
        }
        if ok {
            let proposer = assign
                .iter()
                .map(|&a| if a < n_r { Some(a) } else { None })
                .collect();
            let m = Matching { proposer, receiver };
            if pred(&m) {
                return Some(m);
            }
        }
        // mixed-radix increment, base (n_r + 1)
        let mut i = 0;
        loop {
            if i == n_p {
                return None;
            }
            assign[i] += 1;
            if assign[i] <= n_r {
                break;
            }
            assign[i] = 0;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Wrap strict preference lists as singleton tiers.
    fn as_tiers(prefs: &[Vec<usize>]) -> Vec<Vec<Vec<usize>>> {
        prefs
            .iter()
            .map(|list| list.iter().map(|&x| vec![x]).collect())
            .collect()
    }

    #[test]
    fn without_ties_all_three_collapse_to_plain_stability() {
        let mut rng = Rng::new(0x71E5);
        for _ in 0..400 {
            let n = 1 + rng.below(4);
            let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let m = crate::matching::gale_shapley(&prop, &recv);
            let (pt, rt) = (as_tiers(&prop), as_tiers(&recv));
            let plain = crate::matching::is_stable(&prop, &recv, &m);
            assert_eq!(is_weakly_stable(&pt, &rt, &m), plain);
            assert_eq!(is_strongly_stable(&pt, &rt, &m), plain);
            assert_eq!(is_super_stable(&pt, &rt, &m), plain);
            // Gale-Shapley with strict lists is super-stable.
            assert!(is_super_stable(&pt, &rt, &m));
        }
    }

    #[test]
    fn weakly_stable_output_is_weakly_stable() {
        let mut rng = Rng::new(0x71E6);
        for _ in 0..500 {
            let n = 1 + rng.below(4);
            let prop = random_tiered(n, n, &mut rng);
            let recv = random_tiered(n, n, &mut rng);
            let m = weakly_stable(&prop, &recv, &mut rng);
            assert!(
                is_weakly_stable(&prop, &recv, &m),
                "not weakly stable: p={prop:?} r={recv:?} m={m:?}"
            );
        }
    }

    #[test]
    fn stability_hierarchy_holds_for_found_matchings() {
        // Whenever a super-/strongly-stable matching is found, it must satisfy
        // the weaker notions too (super => strong => weak).
        let mut rng = Rng::new(0x4123);
        let mut super_found = 0;
        let mut strong_found = 0;
        for _ in 0..600 {
            let n = 1 + rng.below(3);
            let prop = random_tiered(n, n, &mut rng);
            let recv = random_tiered(n, n, &mut rng);
            if let Some(m) = super_stable(&prop, &recv) {
                super_found += 1;
                assert!(is_strongly_stable(&prop, &recv, &m));
                assert!(is_weakly_stable(&prop, &recv, &m));
            }
            if let Some(m) = strongly_stable(&prop, &recv) {
                strong_found += 1;
                assert!(is_weakly_stable(&prop, &recv, &m));
            }
        }
        assert!(
            super_found > 0 && strong_found > 0,
            "search never succeeded"
        );
    }

    #[test]
    fn super_stability_can_fail_to_exist() {
        // A 2x2 instance where both proposers are indifferent between the two
        // receivers, who strictly disagree: no super-stable matching exists, yet
        // a weakly-stable one always does.
        let prop = vec![vec![vec![0, 1]], vec![vec![0, 1]]];
        let recv = vec![vec![vec![0], vec![1]], vec![vec![0], vec![1]]];
        assert!(super_stable(&prop, &recv).is_none());
        let mut rng = Rng::new(5);
        let w = weakly_stable(&prop, &recv, &mut rng);
        assert!(is_weakly_stable(&prop, &recv, &w));
    }

    #[test]
    fn irving_reduces_to_gale_shapley_on_strict_lists() {
        // With strict lists the super-stable matching is the ordinary
        // proposer-optimal stable matching, i.e. exactly Gale-Shapley.
        let mut rng = Rng::new(0xA1B2);
        for _ in 0..500 {
            let n = 1 + rng.below(5);
            let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let gs = crate::matching::gale_shapley(&prop, &recv);
            let (pt, rt) = (as_tiers(&prop), as_tiers(&recv));
            let irving = super_stable_irving(&pt, &rt);
            assert_eq!(
                irving.map(|m| m.proposer),
                Some(gs.proposer),
                "irving != GS on strict lists: p={prop:?} r={recv:?}"
            );
        }
    }

    #[test]
    fn irving_super_stable_agrees_with_brute_force() {
        // Irving's polynomial algorithm finds a super-stable matching exactly when
        // the brute-force oracle does, and its output is always super-stable.
        let mut rng = Rng::new(0x5A17);
        let (mut some_cnt, mut none_cnt) = (0, 0);
        for _ in 0..3000 {
            let n = 1 + rng.below(4);
            let prop = random_tiered(n, n, &mut rng);
            let recv = random_tiered(n, n, &mut rng);
            let brute = super_stable(&prop, &recv);
            let fast = super_stable_irving(&prop, &recv);
            assert_eq!(
                brute.is_some(),
                fast.is_some(),
                "existence mismatch: p={prop:?} r={recv:?} brute={brute:?} fast={fast:?}"
            );
            match &fast {
                Some(m) => {
                    some_cnt += 1;
                    assert!(is_super_stable(&prop, &recv, m));
                }
                None => none_cnt += 1,
            }
        }
        assert!(
            some_cnt > 0 && none_cnt > 0,
            "test did not exercise both outcomes: some={some_cnt} none={none_cnt}"
        );
    }

    #[test]
    fn irving_super_stable_agrees_on_incomplete_lists() {
        // The same agreement holds with incomplete lists (some pairs mutually
        // unacceptable), where a super-stable matching may leave agents unmatched.
        let mut rng = Rng::new(0xC0FFEE);
        let (mut some_cnt, mut none_cnt) = (0, 0);
        for _ in 0..3000 {
            let n = 1 + rng.below(4);
            let prop = random_tiered_incomplete(n, n, &mut rng);
            let recv = random_tiered_incomplete(n, n, &mut rng);
            let brute = super_stable_brute_ir(&prop, &recv);
            let fast = super_stable_irving(&prop, &recv);
            assert_eq!(
                brute.is_some(),
                fast.is_some(),
                "existence mismatch (incomplete): p={prop:?} r={recv:?} brute={brute:?} fast={fast:?}"
            );
            match &fast {
                Some(m) => {
                    some_cnt += 1;
                    assert!(is_super_stable(&prop, &recv, m));
                }
                None => none_cnt += 1,
            }
        }
        assert!(
            some_cnt > 0 && none_cnt > 0,
            "incomplete test did not exercise both outcomes: some={some_cnt} none={none_cnt}"
        );
    }

    /// Individual rationality: every matched pair is mutually acceptable. The
    /// plain [`is_super_stable`] checks only blocking, not that the matching
    /// itself is valid, which matters once preference lists are incomplete.
    fn is_ir(
        proposer_prefs: &[Vec<Vec<usize>>],
        receiver_prefs: &[Vec<Vec<usize>>],
        m: &Matching,
    ) -> bool {
        let pc = class_table(proposer_prefs, receiver_prefs.len());
        let rc = class_table(receiver_prefs, proposer_prefs.len());
        m.proposer.iter().enumerate().all(|(p, &q)| match q {
            None => true,
            Some(r) => pc[p][r].is_some() && rc[r][p].is_some(),
        })
    }

    /// Brute-force oracle that, unlike [`super_stable`], also requires the
    /// matching to be individually rational — the correct existence test for
    /// super stability with incomplete lists.
    fn super_stable_brute_ir(
        proposer_prefs: &[Vec<Vec<usize>>],
        receiver_prefs: &[Vec<Vec<usize>>],
    ) -> Option<Matching> {
        find_matching(proposer_prefs.len(), receiver_prefs.len(), |m| {
            is_super_stable(proposer_prefs, receiver_prefs, m)
                && is_ir(proposer_prefs, receiver_prefs, m)
        })
    }

    /// A random tiered profile with possibly-incomplete lists: keep a random
    /// prefix of a permutation as acceptable (the rest unacceptable), then cut the
    /// kept prefix into indifference classes. May yield an empty (all-unacceptable)
    /// list.
    fn random_tiered_incomplete(
        n_agents: usize,
        n_other: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<Vec<usize>>> {
        (0..n_agents)
            .map(|_| {
                let perm = rng.permutation(n_other);
                let keep = rng.below(n_other + 1);
                let kept = &perm[..keep];
                let mut tiers = Vec::new();
                let mut i = 0;
                while i < kept.len() {
                    let take = 1 + rng.below(kept.len() - i);
                    tiers.push(kept[i..i + take].to_vec());
                    i += take;
                }
                tiers
            })
            .collect()
    }

    /// A random tiered preference profile: a permutation of `n_other` objects cut
    /// into 1..=n_other contiguous indifference classes.
    fn random_tiered(n_agents: usize, n_other: usize, rng: &mut Rng) -> Vec<Vec<Vec<usize>>> {
        (0..n_agents)
            .map(|_| {
                let perm = rng.permutation(n_other);
                let mut tiers = Vec::new();
                let mut i = 0;
                while i < perm.len() {
                    // Each class takes 1..=remaining elements.
                    let remaining = perm.len() - i;
                    let take = 1 + rng.below(remaining);
                    tiers.push(perm[i..i + take].to_vec());
                    i += take;
                }
                tiers
            })
            .collect()
    }
}
