//! Kidney exchange — Top Trading Cycles for incompatible patient-donor pairs.
//!
//! A patient who needs a kidney often arrives with a willing living donor who is
//! *incompatible* with them (wrong blood type, positive crossmatch). Two such
//! pairs can rescue each other: if pair A's donor matches pair B's patient and
//! vice versa, they swap donors. Longer **cycles** generalize the swap, and
//! clearing them well is the market-design problem that, since the first paired
//! exchanges in the 2000s, has given tens of thousands of patients a transplant.
//!
//! This module implements the Roth-Sönmez-Ünver (2004) mechanism for the case of
//! exchange **cycles**: model the pool as a housing market where each patient is
//! endowed with its own (incompatible) donor and has strict preferences over the
//! donors it *is* compatible with, then run [Top Trading Cycles](crate::ttc). The
//! own donor is the patient's last-ranked option — the *outside option* of "no
//! exchange, stay on the waiting list" — so by TTC's individual rationality a
//! patient is never assigned a donor worse than keeping its own. The result is
//!
//! - **individually rational** — every patient receives a compatible donor or
//!   keeps its own (it never does worse than not participating),
//! - **Pareto efficient** for the patients, and
//! - **strategy-proof** — no patient can gain by misreporting or hiding part of
//!   its compatibility list (Roth 1982; Roth-Sönmez-Ünver 2004).
//!
//! Only exchange cycles are formed here. Chains seeded by a non-directed
//! (altruistic) donor — which let an exchange end on the waiting list rather than
//! close back on itself — need the wait-list *w*-chain selection rules and are a
//! noted extension, not implemented in this module.

use crate::ttc::top_trading_cycle;

/// An ABO blood type. Compatibility runs O → {O, A, B, AB}, A → {A, AB},
/// B → {B, AB}, AB → {AB}: O is the universal donor, AB the universal recipient.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blood {
    O,
    A,
    B,
    AB,
}

/// Whether a kidney from a `donor`-typed donor can go to a `patient`-typed
/// recipient under ABO rules (ignoring crossmatch / tissue typing).
///
/// ```
/// use match_learn::kidney::{abo_compatible, Blood};
///
/// assert!(abo_compatible(Blood::O, Blood::AB)); // O donates to anyone
/// assert!(abo_compatible(Blood::A, Blood::A));
/// assert!(!abo_compatible(Blood::A, Blood::B));  // A cannot donate to B
/// assert!(!abo_compatible(Blood::AB, Blood::O)); // AB donates only to AB
/// ```
pub fn abo_compatible(donor: Blood, patient: Blood) -> bool {
    use Blood::*;
    match donor {
        O => true,
        A => matches!(patient, A | AB),
        B => matches!(patient, B | AB),
        AB => patient == AB,
    }
}

/// An incompatible patient-donor pair entering the exchange pool. By assumption
/// the patient is incompatible with its own `donor` (the reason it seeks an
/// exchange); the algorithm tolerates a self-compatible pair but treats keeping
/// the own donor as the no-exchange outcome either way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pair {
    pub patient: Blood,
    pub donor: Blood,
}

/// Build each patient's list of *acceptable* (compatible) donors from the pool,
/// in pair-index order, excluding the patient's own donor. This is the input to
/// [`ttc_kidney_exchange`]; richer medical scores (HLA match quality) would
/// refine the within-list order, which the mechanism otherwise leaves to index.
pub fn compatibility_prefs(pairs: &[Pair]) -> Vec<Vec<usize>> {
    let n = pairs.len();
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i && abo_compatible(pairs[j].donor, pairs[i].patient))
                .collect()
        })
        .collect()
}

/// Clear the exchange by Top Trading Cycles.
///
/// `acceptable[i]` lists patient `i`'s compatible donors, best first, *excluding*
/// its own donor `i`. Returns `receives`, where `receives[i]` is the donor patient
/// `i` ends up with; `receives[i] == i` means **no exchange** — patient `i` keeps
/// its own donor and stays on the waiting list. The result is always a
/// permutation (no donor is used twice), individually rational, Pareto efficient,
/// and strategy-proof.
///
/// ```
/// use match_learn::kidney::ttc_kidney_exchange;
///
/// // 0 can only use 1's donor, 1 only 2's, 2 only 0's: no 2-way swap exists,
/// // but the 3-cycle clears everyone.
/// let acceptable = vec![vec![1], vec![2], vec![0]];
/// assert_eq!(ttc_kidney_exchange(&acceptable), vec![1, 2, 0]);
/// ```
pub fn ttc_kidney_exchange(acceptable: &[Vec<usize>]) -> Vec<usize> {
    let n = acceptable.len();
    // Append the own donor as the last-ranked option. It is the patient's
    // endowment, so TTC keeps it available until the patient is assigned and,
    // by individual rationality, never assigns anything ranked below it.
    let prefs: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut p = acceptable[i].clone();
            if !p.contains(&i) {
                p.push(i);
            }
            p
        })
        .collect();
    top_trading_cycle(&prefs)
}

/// Clear a pool of blood-typed pairs: build the ABO compatibility lists and run
/// [`ttc_kidney_exchange`].
///
/// ```
/// use match_learn::kidney::{kidney_exchange, Blood, Pair};
///
/// // An A-patient/B-donor pair and a B-patient/A-donor pair: each is
/// // incompatible with its own donor but matches the other's. They swap.
/// let pairs = vec![
///     Pair { patient: Blood::A, donor: Blood::B },
///     Pair { patient: Blood::B, donor: Blood::A },
/// ];
/// assert_eq!(kidney_exchange(&pairs), vec![1, 0]);
/// ```
pub fn kidney_exchange(pairs: &[Pair]) -> Vec<usize> {
    ttc_kidney_exchange(&compatibility_prefs(pairs))
}

/// Number of patients who receive a transplant (are part of some exchange cycle).
pub fn num_transplants(receives: &[usize]) -> usize {
    receives
        .iter()
        .enumerate()
        .filter(|&(i, &d)| d != i)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn is_permutation(v: &[usize]) -> bool {
        let mut s = v.to_vec();
        s.sort_unstable();
        s == (0..v.len()).collect::<Vec<_>>()
    }

    /// In-place next lexicographic permutation; false when the last is reached.
    fn next_permutation(a: &mut [usize]) -> bool {
        if a.len() < 2 {
            return false;
        }
        let mut i = a.len() - 1;
        while i > 0 && a[i - 1] >= a[i] {
            i -= 1;
        }
        if i == 0 {
            return false;
        }
        let mut j = a.len() - 1;
        while a[j] <= a[i - 1] {
            j -= 1;
        }
        a.swap(i - 1, j);
        a[i..].reverse();
        true
    }

    fn random_pair(rng: &mut Rng) -> Pair {
        let blood = |r: &mut Rng| match r.below(4) {
            0 => Blood::O,
            1 => Blood::A,
            2 => Blood::B,
            _ => Blood::AB,
        };
        Pair {
            patient: blood(rng),
            donor: blood(rng),
        }
    }

    /// Rank of donor `d` for patient with acceptable list `acc` (own donor /
    /// no-exchange ranks last).
    fn rank_of(acc: &[usize], d: usize) -> usize {
        acc.iter().position(|&x| x == d).unwrap_or(acc.len())
    }

    #[test]
    fn two_way_blood_type_swap() {
        let pairs = vec![
            Pair {
                patient: Blood::A,
                donor: Blood::B,
            },
            Pair {
                patient: Blood::B,
                donor: Blood::A,
            },
        ];
        // Each is incompatible with its own donor (why they need the exchange).
        assert!(!abo_compatible(pairs[0].donor, pairs[0].patient));
        assert!(!abo_compatible(pairs[1].donor, pairs[1].patient));
        assert_eq!(kidney_exchange(&pairs), vec![1, 0]);
    }

    #[test]
    fn three_way_cycle_when_no_two_way_exists() {
        let acceptable = vec![vec![1], vec![2], vec![0]];
        assert_eq!(ttc_kidney_exchange(&acceptable), vec![1, 2, 0]);
    }

    #[test]
    fn sensitized_patient_blocks_the_cycle() {
        // Patient 1 is compatible with nobody (highly sensitized). The would-be
        // 0 -> 1 -> 2 -> 0 cycle cannot form, so everyone keeps its own donor.
        let acceptable = vec![vec![1], vec![], vec![0]];
        assert_eq!(ttc_kidney_exchange(&acceptable), vec![0, 1, 2]);
    }

    #[test]
    fn result_is_a_valid_individually_rational_exchange() {
        // Every patient either keeps its own donor or receives a compatible one,
        // and no donor is used twice.
        let mut rng = Rng::new(0xC1D5);
        for _ in 0..3000 {
            let n = 1 + rng.below(7);
            let pairs: Vec<Pair> = (0..n).map(|_| random_pair(&mut rng)).collect();
            let receives = kidney_exchange(&pairs);
            assert!(is_permutation(&receives), "not a permutation: {receives:?}");
            for (i, &d) in receives.iter().enumerate() {
                assert!(
                    d == i || abo_compatible(pairs[d].donor, pairs[i].patient),
                    "patient {i} received incompatible donor {d}: {pairs:?}"
                );
            }
        }
    }

    #[test]
    fn exchange_is_pareto_efficient_for_patients() {
        // Brute force: no other valid exchange makes every patient weakly better
        // and someone strictly better.
        let mut rng = Rng::new(0xC2E6);
        for _ in 0..400 {
            let n = 1 + rng.below(5);
            let pairs: Vec<Pair> = (0..n).map(|_| random_pair(&mut rng)).collect();
            let acceptable = compatibility_prefs(&pairs);
            let result = ttc_kidney_exchange(&acceptable);

            let valid = |a: &[usize]| {
                a.iter()
                    .enumerate()
                    .all(|(i, &d)| d == i || acceptable[i].contains(&d))
            };
            let dominates = |other: &[usize]| {
                let mut strict = false;
                for (i, acc) in acceptable.iter().enumerate() {
                    let ro = rank_of(acc, other[i]);
                    let rb = rank_of(acc, result[i]);
                    if ro > rb {
                        return false;
                    }
                    if ro < rb {
                        strict = true;
                    }
                }
                strict
            };

            let mut perm: Vec<usize> = (0..n).collect();
            loop {
                if valid(&perm) && dominates(&perm) {
                    panic!("not Pareto efficient: {perm:?} beats {result:?} for {acceptable:?}");
                }
                if !next_permutation(&mut perm) {
                    break;
                }
            }
        }
    }

    /// Every ordered sub-list (subset, in any order) of `items`, including empty —
    /// the manipulations available to a patient: reorder and/or hide donors.
    fn sub_orderings(items: &[usize]) -> Vec<Vec<usize>> {
        fn rec(
            items: &[usize],
            used: &mut [bool],
            cur: &mut Vec<usize>,
            out: &mut Vec<Vec<usize>>,
        ) {
            for (i, &it) in items.iter().enumerate() {
                if used[i] {
                    continue;
                }
                used[i] = true;
                cur.push(it);
                out.push(cur.clone());
                rec(items, used, cur, out);
                cur.pop();
                used[i] = false;
            }
        }
        let mut out = vec![Vec::new()];
        let mut used = vec![false; items.len()];
        let mut cur = Vec::new();
        rec(items, &mut used, &mut cur, &mut out);
        out
    }

    #[test]
    fn ttc_is_strategy_proof_for_a_patient() {
        // No reordering or truncation of patient 0's compatibility list yields a
        // donor it truly prefers to the truthful outcome.
        let mut rng = Rng::new(0xC3F7);
        for _ in 0..300 {
            let n = 2 + rng.below(4);
            let pairs: Vec<Pair> = (0..n).map(|_| random_pair(&mut rng)).collect();
            let truth = compatibility_prefs(&pairs);
            let honest = ttc_kidney_exchange(&truth)[0];
            let honest_rank = rank_of(&truth[0], honest);

            for report in sub_orderings(&truth[0]) {
                let mut manip = truth.clone();
                manip[0] = report;
                let got = ttc_kidney_exchange(&manip)[0];
                // Score the manipulated outcome by patient 0's TRUE preferences.
                assert!(
                    rank_of(&truth[0], got) >= honest_rank,
                    "patient 0 gained by misreporting: truth={truth:?}"
                );
            }
        }
    }

    #[test]
    fn counts_transplants() {
        assert_eq!(num_transplants(&[1, 0, 2]), 2); // 0<->1 swap, 2 keeps own
        assert_eq!(num_transplants(&[0, 1, 2]), 0); // nobody exchanged
        assert_eq!(num_transplants(&[1, 2, 0]), 3); // a full 3-cycle
    }
}
