//! Gale-Shapley deferred-acceptance stable matching (from scratch).
//!
//! Computes the **proposer-optimal** stable matching for a two-sided market in
//! which each side ranks (some of) the other. Preference lists may be partial:
//! anyone not on a list is *unacceptable*, and unacceptable agents are never
//! matched. The two sides may have different sizes; agents left without an
//! acceptable partner stay unmatched.

/// A two-sided matching.
///
/// `proposer[p] == Some(r)` iff proposer `p` is matched to receiver `r`, and
/// symmetrically for `receiver`. The two views are always consistent.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Matching {
    /// Partner of each proposer, if matched.
    pub proposer: Vec<Option<usize>>,
    /// Partner of each receiver, if matched.
    pub receiver: Vec<Option<usize>>,
}

impl Matching {
    /// Number of matched pairs.
    pub fn pairs(&self) -> usize {
        self.proposer.iter().filter(|m| m.is_some()).count()
    }
}

/// Build `rank[a][b] = Some(position)` for each `b` on `a`'s list, else `None`.
///
/// `n_other` is the size of the opposite side, so every list index is in range.
pub(crate) fn rank_table(prefs: &[Vec<usize>], n_other: usize) -> Vec<Vec<Option<usize>>> {
    let mut table = vec![vec![None; n_other]; prefs.len()];
    for (a, list) in prefs.iter().enumerate() {
        for (pos, &b) in list.iter().enumerate() {
            if b < n_other {
                table[a][b] = Some(pos);
            }
        }
    }
    table
}

/// Proposer-optimal stable matching via Gale-Shapley deferred acceptance.
///
/// `proposer_prefs[p]` is `p`'s ranking of receivers (most preferred first);
/// `receiver_prefs[r]` is `r`'s ranking of proposers. A pair is acceptable only
/// if each appears on the other's list. Runs in `O(P * R)` time.
pub fn gale_shapley(proposer_prefs: &[Vec<usize>], receiver_prefs: &[Vec<usize>]) -> Matching {
    let n_p = proposer_prefs.len();
    let n_r = receiver_prefs.len();
    let recv_rank = rank_table(receiver_prefs, n_p);

    let mut proposer = vec![None; n_p];
    let mut receiver = vec![None; n_r];
    let mut next = vec![0usize; n_p]; // index of next receiver `p` will propose to

    // Every proposer starts free and tries its list in order.
    let mut free: Vec<usize> = (0..n_p).collect();

    while let Some(p) = free.pop() {
        while next[p] < proposer_prefs[p].len() {
            let r = proposer_prefs[p][next[p]];
            next[p] += 1;
            if r >= n_r {
                continue; // out-of-range entry, ignore defensively
            }
            let Some(p_rank) = recv_rank[r][p] else {
                continue; // `r` finds `p` unacceptable; keep proposing
            };
            match receiver[r] {
                None => {
                    receiver[r] = Some(p);
                    proposer[p] = Some(r);
                    break;
                }
                Some(cur) => {
                    let cur_rank = recv_rank[r][cur].expect("matched pair is acceptable");
                    if p_rank < cur_rank {
                        // `r` prefers `p`; the incumbent becomes free again.
                        proposer[cur] = None;
                        receiver[r] = Some(p);
                        proposer[p] = Some(r);
                        free.push(cur);
                        break;
                    }
                    // `r` keeps the incumbent; `p` continues down its list.
                }
            }
        }
    }

    Matching { proposer, receiver }
}

/// Whether `m` is stable under the given strict preferences.
///
/// A matching is stable when no *blocking pair* exists: a proposer `p` and
/// receiver `r`, mutually acceptable, who would each rather be matched together
/// than with their current partner (being unmatched counts as worst).
pub fn is_stable(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
    m: &Matching,
) -> bool {
    let n_p = proposer_prefs.len();
    let n_r = receiver_prefs.len();
    let prop_rank = rank_table(proposer_prefs, n_r);
    let recv_rank = rank_table(receiver_prefs, n_p);

    for p in 0..n_p {
        for &r in &proposer_prefs[p] {
            if r >= n_r {
                continue;
            }
            // Does `p` strictly prefer `r` to its current partner?
            let p_wants = match m.proposer[p] {
                Some(cur) => prop_rank[p][r] < prop_rank[p][cur],
                None => true, // unmatched: any acceptable `r` is an improvement
            };
            if !p_wants {
                continue;
            }
            // Is `p` acceptable to `r`, and does `r` strictly prefer `p`?
            let Some(p_rank) = recv_rank[r][p] else {
                continue;
            };
            let r_wants = match m.receiver[r] {
                Some(cur) => p_rank < recv_rank[r][cur].expect("matched pair is acceptable"),
                None => true,
            };
            if r_wants {
                return false; // (p, r) is a blocking pair
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn classic_three_by_three() {
        // A well-known instance with a unique proposer-optimal matching.
        // Proposers 0,1,2 ; receivers 0,1,2.
        let prop = vec![vec![0, 1, 2], vec![1, 0, 2], vec![0, 1, 2]];
        let recv = vec![vec![1, 0, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let m = gale_shapley(&prop, &recv);
        assert!(is_stable(&prop, &recv, &m));
        assert_eq!(m.pairs(), 3);
    }

    #[test]
    fn proposer_optimal_when_preferences_conflict() {
        // Both proposers prefer receiver 0; receiver 0 prefers proposer 0.
        // Proposer-optimal: 0<->0, 1<->1.
        let prop = vec![vec![0, 1], vec![0, 1]];
        let recv = vec![vec![0, 1], vec![0, 1]];
        let m = gale_shapley(&prop, &recv);
        assert_eq!(m.proposer, vec![Some(0), Some(1)]);
        assert!(is_stable(&prop, &recv, &m));
    }

    #[test]
    fn unequal_sizes_leave_some_unmatched() {
        // 3 proposers, 2 receivers: exactly one proposer goes unmatched.
        let prop = vec![vec![0, 1], vec![0, 1], vec![0, 1]];
        let recv = vec![vec![0, 1, 2], vec![0, 1, 2]];
        let m = gale_shapley(&prop, &recv);
        assert_eq!(m.pairs(), 2);
        assert!(is_stable(&prop, &recv, &m));
    }

    #[test]
    fn partial_lists_respect_unacceptability() {
        // Proposer 0 only accepts receiver 1; receiver 1 only accepts proposer 1.
        // So 0 cannot match 1, and ends unmatched; 1 matches 1.
        let prop = vec![vec![1], vec![1, 0]];
        let recv = vec![vec![1], vec![1]];
        let m = gale_shapley(&prop, &recv);
        assert_eq!(m.proposer[0], None);
        assert_eq!(m.proposer[1], Some(1));
        assert!(is_stable(&prop, &recv, &m));
    }

    /// Brute-force oracle: enumerate all stable matchings, confirm GS produces one.
    fn all_stable_matchings(prop: &[Vec<usize>], recv: &[Vec<usize>]) -> Vec<Matching> {
        let n_p = prop.len();
        let n_r = recv.len();
        let mut out = Vec::new();
        // Each proposer maps to a receiver index in 0..n_r or "unmatched" (n_r).
        let mut assign = vec![0usize; n_p];
        'outer: loop {
            // Reject assignments where two proposers share a receiver.
            let mut receiver = vec![None; n_r];
            let mut ok = true;
            for p in 0..n_p {
                if assign[p] < n_r {
                    if receiver[assign[p]].is_some() {
                        ok = false;
                        break;
                    }
                    receiver[assign[p]] = Some(p);
                }
            }
            if ok {
                let proposer: Vec<Option<usize>> = assign
                    .iter()
                    .map(|&a| if a < n_r { Some(a) } else { None })
                    .collect();
                let m = Matching { proposer, receiver };
                if is_stable(prop, recv, &m) {
                    out.push(m);
                }
            }
            // Increment the mixed-radix counter (base n_r+1 per proposer).
            let mut i = 0;
            loop {
                if i == n_p {
                    break 'outer;
                }
                assign[i] += 1;
                if assign[i] <= n_r {
                    break;
                }
                assign[i] = 0;
                i += 1;
            }
        }
        out
    }

    #[test]
    fn gs_matches_brute_force_oracle_on_random_instances() {
        let mut rng = Rng::new(20260606);
        for _ in 0..300 {
            let n_p = 1 + rng.below(4); // 1..=4
            let n_r = 1 + rng.below(4);
            let prop: Vec<Vec<usize>> = (0..n_p).map(|_| rng.permutation(n_r)).collect();
            let recv: Vec<Vec<usize>> = (0..n_r).map(|_| rng.permutation(n_p)).collect();
            let m = gale_shapley(&prop, &recv);
            // GS output must be stable...
            assert!(
                is_stable(&prop, &recv, &m),
                "GS produced an unstable matching for prop={prop:?} recv={recv:?}"
            );
            // ...and must appear in the brute-force set of stable matchings.
            let oracle = all_stable_matchings(&prop, &recv);
            assert!(oracle.contains(&m), "GS matching not in oracle set");
        }
    }

    #[test]
    fn gs_is_proposer_optimal_among_stable_matchings() {
        // For complete-list instances, each proposer's GS partner is at least as
        // good as in any other stable matching (proposer-optimality).
        let mut rng = Rng::new(7);
        for _ in 0..200 {
            let n = 1 + rng.below(4);
            let prop: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let recv: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let gs = gale_shapley(&prop, &recv);
            let prop_rank = rank_table(&prop, n);
            for other in all_stable_matchings(&prop, &recv) {
                for (p, ranks) in prop_rank.iter().enumerate() {
                    if let (Some(g), Some(o)) = (gs.proposer[p], other.proposer[p]) {
                        // Lower rank = more preferred; GS is never worse.
                        assert!(ranks[g] <= ranks[o]);
                    }
                }
            }
        }
    }
}
