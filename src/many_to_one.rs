//! Many-to-one stable matching: the Hospital-Residents problem.
//!
//! A generalization of [one-to-one Gale-Shapley](crate::matching) in which each
//! receiver `r` (a "hospital") can hold up to `capacity[r]` proposers (its
//! "residents"). Proposer-proposing deferred acceptance yields the
//! proposer-optimal stable matching; `capacity[r] == 1` recovers the 1:1 case.

use crate::matching::rank_table;

/// A many-to-one matching.
///
/// Each proposer is matched to at most one receiver; each receiver `r` holds at
/// most `capacity[r]` proposers. The two views are consistent: `proposer[p] ==
/// Some(r)` iff `p` appears in `receiver[r]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManyToOne {
    /// Receiver each proposer is assigned to, if any.
    pub proposer: Vec<Option<usize>>,
    /// Proposers held by each receiver (at most `capacity[r]` of them).
    pub receiver: Vec<Vec<usize>>,
}

impl ManyToOne {
    /// Total number of matched proposers.
    pub fn matched(&self) -> usize {
        self.proposer.iter().filter(|m| m.is_some()).count()
    }
}

/// Proposer-optimal stable matching for the Hospital-Residents problem.
///
/// `proposer_prefs[p]` ranks receivers; `receiver_prefs[r]` ranks proposers;
/// `capacities[r]` is `r`'s quota. A pair is acceptable only if each appears on
/// the other's list. Runs in `O(P * R)`.
///
/// ```
/// use match_learn::hospital_residents;
///
/// // Three residents all want the one hospital, which has two slots and
/// // prefers 0 > 1 > 2. Resident 2 is left unmatched (-1 in `proposer`).
/// let residents = vec![vec![0], vec![0], vec![0]];
/// let hospitals = vec![vec![0, 1, 2]];
/// let m = hospital_residents(&residents, &hospitals, &[2]);
/// assert_eq!(m.matched(), 2);
/// assert_eq!(m.proposer[2], None);
/// ```
pub fn hospital_residents(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
    capacities: &[usize],
) -> ManyToOne {
    let n_p = proposer_prefs.len();
    let n_r = receiver_prefs.len();
    assert_eq!(capacities.len(), n_r, "one capacity per receiver");
    let recv_rank = rank_table(receiver_prefs, n_p);

    let mut proposer = vec![None; n_p];
    let mut held: Vec<Vec<usize>> = vec![Vec::new(); n_r];
    let mut next = vec![0usize; n_p];
    let mut free: Vec<usize> = (0..n_p).collect();

    while let Some(p) = free.pop() {
        while next[p] < proposer_prefs[p].len() {
            let r = proposer_prefs[p][next[p]];
            next[p] += 1;
            if r >= n_r {
                continue;
            }
            if recv_rank[r][p].is_none() {
                continue; // `r` finds `p` unacceptable
            }
            // Tentatively accept `p`.
            held[r].push(p);
            proposer[p] = Some(r);
            if held[r].len() <= capacities[r] {
                break; // within quota; `p` is held
            }
            // Over quota: reject the worst-ranked held proposer.
            let (worst_pos, _) = held[r]
                .iter()
                .enumerate()
                .max_by_key(|(_, q)| recv_rank[r][**q].expect("held pair is acceptable"))
                .expect("held is non-empty");
            let worst = held[r].remove(worst_pos);
            proposer[worst] = None;
            if worst == p {
                continue; // `p` was the one rejected; keep going down its list
            } else {
                free.push(worst); // a previously-held proposer is bumped
                break;
            }
        }
    }

    ManyToOne {
        proposer,
        receiver: held,
    }
}

/// Whether `m` is stable for the Hospital-Residents instance.
///
/// A blocking pair `(p, r)` is mutually acceptable, `p` prefers `r` to its
/// assignment (or is unmatched), and `r` either has a free slot or prefers `p`
/// to its worst currently-held proposer.
pub fn is_stable(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
    capacities: &[usize],
    m: &ManyToOne,
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
            let p_wants = match m.proposer[p] {
                Some(cur) => prop_rank[p][r] < prop_rank[p][cur],
                None => true,
            };
            if !p_wants {
                continue;
            }
            let Some(p_rank) = recv_rank[r][p] else {
                continue;
            };
            let held = &m.receiver[r];
            let r_wants = if held.len() < capacities[r] {
                true
            } else {
                let worst = held
                    .iter()
                    .map(|&q| recv_rank[r][q].expect("held pair is acceptable"))
                    .max()
                    .expect("held is non-empty when at capacity");
                p_rank < worst
            };
            if r_wants {
                return false;
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
    fn capacity_one_matches_one_to_one_gale_shapley() {
        let prop = vec![vec![0, 1], vec![0, 1]];
        let recv = vec![vec![0, 1], vec![0, 1]];
        let m = hospital_residents(&prop, &recv, &[1, 1]);
        let gs = crate::matching::gale_shapley(&prop, &recv);
        assert_eq!(m.proposer, gs.proposer);
        assert!(is_stable(&prop, &recv, &[1, 1], &m));
    }

    #[test]
    fn one_hospital_absorbs_up_to_capacity() {
        // 3 residents, 1 hospital with 2 slots: it keeps its 2 favourites.
        let prop = vec![vec![0], vec![0], vec![0]];
        let recv = vec![vec![0, 1, 2]]; // hospital prefers 0 > 1 > 2
        let m = hospital_residents(&prop, &recv, &[2]);
        assert_eq!(m.matched(), 2);
        let mut held = m.receiver[0].clone();
        held.sort_unstable();
        assert_eq!(held, vec![0, 1]);
        assert_eq!(m.proposer[2], None);
        assert!(is_stable(&prop, &recv, &[2], &m));
    }

    /// Brute force: enumerate every capacity-respecting assignment and collect
    /// the stable ones.
    fn all_stable(prop: &[Vec<usize>], recv: &[Vec<usize>], cap: &[usize]) -> Vec<ManyToOne> {
        let n_p = prop.len();
        let n_r = recv.len();
        let mut out = Vec::new();
        let mut assign = vec![0usize; n_p]; // value n_r == unmatched
        loop {
            let mut receiver = vec![Vec::new(); n_r];
            let mut ok = true;
            for (p, &a) in assign.iter().enumerate() {
                if a < n_r {
                    receiver[a].push(p);
                    if receiver[a].len() > cap[a] {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                let proposer = assign
                    .iter()
                    .map(|&a| if a < n_r { Some(a) } else { None })
                    .collect();
                let m = ManyToOne { proposer, receiver };
                if is_stable(prop, recv, cap, &m) {
                    out.push(m);
                }
            }
            // mixed-radix increment, base (n_r + 1)
            let mut i = 0;
            loop {
                if i == n_p {
                    return out;
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

    #[test]
    fn hr_matches_brute_force_oracle_on_random_instances() {
        let mut rng = Rng::new(0x4D544F);
        for _ in 0..300 {
            let n_p = 1 + rng.below(4);
            let n_r = 1 + rng.below(3);
            let prop: Vec<Vec<usize>> = (0..n_p).map(|_| rng.permutation(n_r)).collect();
            let recv: Vec<Vec<usize>> = (0..n_r).map(|_| rng.permutation(n_p)).collect();
            let cap: Vec<usize> = (0..n_r).map(|_| 1 + rng.below(2)).collect();
            let m = hospital_residents(&prop, &recv, &cap);
            assert!(
                is_stable(&prop, &recv, &cap, &m),
                "HR produced an unstable matching: prop={prop:?} recv={recv:?} cap={cap:?}"
            );
            let oracle = all_stable(&prop, &recv, &cap);
            assert!(
                oracle.iter().any(|o| o.proposer == m.proposer),
                "HR matching not in oracle set"
            );
        }
    }
}
