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
/// Exponential — `O((R+1)^P)` — and intended for small instances and analysis
/// (the polynomial Irving algorithm is future work). Returns the first super-
/// stable matching found, or `None` if none exists (super stability can fail).
pub fn super_stable(
    proposer_prefs: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<Vec<usize>>],
) -> Option<Matching> {
    find_matching(proposer_prefs.len(), receiver_prefs.len(), |m| {
        is_super_stable(proposer_prefs, receiver_prefs, m)
    })
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
