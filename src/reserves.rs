//! Many-to-one matching with diversity reserves (Phase 8 / equity).
//!
//! Real allocation problems carry distributional goals: school choice reserves
//! seats for a target group, residency programs reserve posts for under-served
//! regions. This module runs proposer-proposing deferred acceptance where each
//! receiver applies a **minority-reserve choice function** (Hafalir-Yenmez-
//! Yildirim): of its `capacity[r]` slots, `reserve[r]` give priority to
//! proposers of the reserved type (`type 0`), and the rest are open to all by
//! preference. `reserve[r] == 0` recovers ordinary Hospital-Residents.
//!
//! Reserves never reduce the reserved type's access and typically raise it,
//! while keeping the matching stable with respect to the choice function.

use crate::many_to_one::ManyToOne;
use crate::matching::rank_table;

/// Choose which applicants a receiver keeps, honoring its reserve.
///
/// `applicants` are proposer indices; `rank[a]` is the receiver's rank of `a`
/// (lower = preferred); `types[a]` is `0` (reserved type) or otherwise. Reserved
/// slots are filled first by the best reserved-type applicants, then all
/// remaining slots by the best applicants of any type. Returns the kept set.
fn reserve_choice(
    applicants: &[usize],
    rank: &[Option<usize>],
    types: &[usize],
    capacity: usize,
    reserve: usize,
) -> Vec<usize> {
    // Acceptable applicants (the receiver lists them), best first.
    let mut sorted: Vec<usize> = applicants
        .iter()
        .copied()
        .filter(|&a| rank[a].is_some())
        .collect();
    sorted.sort_by_key(|&a| rank[a].unwrap());

    let mut kept = Vec::new();
    // Phase 1: reserved slots, best reserved-type applicants up to `reserve`.
    let mut reserved_used = 0;
    for &a in &sorted {
        if kept.len() == capacity || reserved_used == reserve {
            break;
        }
        if types[a] == 0 {
            kept.push(a);
            reserved_used += 1;
        }
    }
    // Phase 2: open slots, best remaining applicants of any type.
    for &a in &sorted {
        if kept.len() == capacity {
            break;
        }
        if !kept.contains(&a) {
            kept.push(a);
        }
    }
    kept
}

/// Proposer-proposing deferred acceptance with per-receiver diversity reserves.
///
/// `types[p]` is proposer `p`'s type (`0` is the reserved type); `reserve[r]` is
/// how many of receiver `r`'s `capacities[r]` slots prioritize the reserved type.
pub fn deferred_acceptance_with_reserves(
    proposer_prefs: &[Vec<usize>],
    receiver_prefs: &[Vec<usize>],
    capacities: &[usize],
    types: &[usize],
    reserve: &[usize],
) -> ManyToOne {
    let n_p = proposer_prefs.len();
    let n_r = receiver_prefs.len();
    assert_eq!(capacities.len(), n_r, "one capacity per receiver");
    assert_eq!(reserve.len(), n_r, "one reserve per receiver");
    assert_eq!(types.len(), n_p, "one type per proposer");
    let recv_rank = rank_table(receiver_prefs, n_p);

    let mut proposer = vec![None; n_p];
    let mut held: Vec<Vec<usize>> = vec![Vec::new(); n_r];
    let mut next = vec![0usize; n_p];
    let mut free: Vec<usize> = (0..n_p).collect();

    while let Some(p) = free.pop() {
        while next[p] < proposer_prefs[p].len() {
            let r = proposer_prefs[p][next[p]];
            next[p] += 1;
            if r >= n_r || recv_rank[r][p].is_none() {
                continue; // out of range or unacceptable
            }
            // Apply the choice function to the incumbents plus `p`.
            let mut applicants = held[r].clone();
            applicants.push(p);
            let kept = reserve_choice(&applicants, &recv_rank[r], types, capacities[r], reserve[r]);

            // Anyone previously held but not kept is rejected and freed.
            for &q in &held[r] {
                if !kept.contains(&q) {
                    proposer[q] = None;
                    free.push(q);
                }
            }
            let p_kept = kept.contains(&p);
            held[r] = kept;
            for &q in &held[r] {
                proposer[q] = Some(r);
            }
            if p_kept {
                break; // `p` is (tentatively) placed
            }
            // else: `p` was rejected immediately; keep going down its list.
        }
    }

    ManyToOne {
        proposer,
        receiver: held,
    }
}

/// Number of matched proposers of the reserved type (`type 0`).
pub fn reserved_type_matched(m: &ManyToOne, types: &[usize]) -> usize {
    m.proposer
        .iter()
        .enumerate()
        .filter(|&(p, slot)| slot.is_some() && types[p] == 0)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::many_to_one::hospital_residents;
    use crate::rng::Rng;

    #[test]
    fn zero_reserve_matches_hospital_residents() {
        let mut rng = Rng::new(1);
        for _ in 0..200 {
            let n_p = 2 + rng.below(4);
            let n_r = 1 + rng.below(3);
            let prop: Vec<Vec<usize>> = (0..n_p).map(|_| rng.permutation(n_r)).collect();
            let recv: Vec<Vec<usize>> = (0..n_r).map(|_| rng.permutation(n_p)).collect();
            let cap: Vec<usize> = (0..n_r).map(|_| 1 + rng.below(2)).collect();
            let types: Vec<usize> = (0..n_p).map(|_| rng.below(2)).collect();
            let zero = vec![0usize; n_r];

            let hr = hospital_residents(&prop, &recv, &cap);
            let res = deferred_acceptance_with_reserves(&prop, &recv, &cap, &types, &zero);
            assert_eq!(res.proposer, hr.proposer, "zero reserve != HR");
        }
    }

    #[test]
    fn a_reserve_admits_an_otherwise_rejected_minority() {
        // One school, 1 seat. The school prefers the majority applicant (1), so
        // without a reserve the minority (0) is shut out; a reserve flips it.
        let prop = vec![vec![0], vec![0]]; // both want school 0
        let recv = vec![vec![1, 0]]; // school prefers proposer 1 (majority)
        let cap = vec![1];
        let types = vec![0usize, 1]; // proposer 0 is the reserved type
        let m0 = deferred_acceptance_with_reserves(&prop, &recv, &cap, &types, &[0]);
        assert_eq!(m0.proposer[0], None); // minority shut out
        assert_eq!(m0.proposer[1], Some(0));
        let m1 = deferred_acceptance_with_reserves(&prop, &recv, &cap, &types, &[1]);
        assert_eq!(m1.proposer[0], Some(0)); // reserve seats the minority
        assert_eq!(m1.proposer[1], None);
    }

    #[test]
    fn reserves_never_reduce_reserved_type_access() {
        // Across random instances, reserving every slot for the minority gives the
        // minority at least as many matches as no reserve.
        let mut rng = Rng::new(7);
        for _ in 0..300 {
            let n_p = 3 + rng.below(5);
            let n_r = 1 + rng.below(3);
            let prop: Vec<Vec<usize>> = (0..n_p).map(|_| rng.permutation(n_r)).collect();
            let recv: Vec<Vec<usize>> = (0..n_r).map(|_| rng.permutation(n_p)).collect();
            let cap: Vec<usize> = (0..n_r).map(|_| 1 + rng.below(2)).collect();
            let types: Vec<usize> = (0..n_p).map(|_| rng.below(2)).collect();

            let none = vec![0usize; n_r];
            let full: Vec<usize> = cap.clone();
            let m_none = deferred_acceptance_with_reserves(&prop, &recv, &cap, &types, &none);
            let m_full = deferred_acceptance_with_reserves(&prop, &recv, &cap, &types, &full);

            assert!(
                reserved_type_matched(&m_full, &types) >= reserved_type_matched(&m_none, &types),
                "reserves reduced minority access"
            );
        }
    }
}
