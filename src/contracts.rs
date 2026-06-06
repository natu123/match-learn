//! Matching with contracts — the cumulative offer process (Hatfield-Milgrom).
//!
//! Plain [Hospital-Residents](crate::many_to_one) matches a doctor *to* a
//! hospital. Matching with contracts matches a doctor to a hospital **on terms**:
//! a contract bundles a doctor, a hospital, and the terms of their agreement
//! (a wage level, a position, a length of service). Doctors rank whole contracts,
//! and a hospital chooses a *set* of contracts via a choice function — so the
//! employer's decision can depend on the terms, not just on who the doctor is.
//!
//! This is the framework behind cadet-branch matching (the U.S. Military Academy
//! assigns cadets to branches with a service-length term, redesigned on exactly
//! this theory) and entry-level labor markets with wages. It also unifies earlier
//! mechanisms: with a single contract per doctor-hospital pair it reduces to
//! Hospital-Residents, and thence to Gale-Shapley.
//!
//! The mechanism is the **cumulative offer process**: each unassigned doctor
//! offers its most-preferred not-yet-rejected contract; each hospital holds its
//! choice from *every* contract ever offered to it (the offers accumulate); a
//! doctor whose held contract is dropped offers again. When the hospital choice
//! functions are **substitutable** and satisfy the irrelevance of rejected
//! contracts — as the responsive, capacity-limited choice built here is — the
//! result is a stable allocation: no doctor and hospital can both improve by
//! signing a contract outside it ([`is_stable_with_contracts`]).

/// A contract: an agreement between a `doctor` and a `hospital` on some `terms`
/// (an opaque label — a wage band, a position, a length of service). Preference
/// lists rank contracts by index, so the terms enter through that ranking.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Contract {
    pub doctor: usize,
    pub hospital: usize,
    pub terms: usize,
}

/// An allocation of contracts.
///
/// `doctor[d]` is the contract index doctor `d` signs, if any; `hospital[h]` is
/// the set of contract indices hospital `h` holds. The two views are consistent.
#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContractMatch {
    pub doctor: Vec<Option<usize>>,
    pub hospital: Vec<Vec<usize>>,
}

/// A hospital's choice from the contracts offered to it: walk its ranking and
/// greedily take acceptable contracts, at most one per doctor and at most
/// `capacity` in total. This responsive, capacity-limited rule is substitutable
/// and satisfies the irrelevance of rejected contracts — the conditions under
/// which the cumulative offer process is stable.
fn choose(
    rank: &[usize],
    offered: &[bool],
    capacity: usize,
    contracts: &[Contract],
    n_d: usize,
) -> Vec<usize> {
    let mut chosen = Vec::new();
    let mut taken = vec![false; n_d];
    for &c in rank {
        if chosen.len() >= capacity {
            break;
        }
        if offered[c] && !taken[contracts[c].doctor] {
            taken[contracts[c].doctor] = true;
            chosen.push(c);
        }
    }
    chosen
}

/// Run the cumulative offer process and return the stable allocation.
///
/// `contracts` is the list of all possible contracts; `doctor_prefs[d]` ranks the
/// contracts of doctor `d` (most preferred first); `hospital_ranks[h]` ranks the
/// contracts hospital `h` finds acceptable; `capacities[h]` is `h`'s quota. A
/// contract is signed only if it appears on both its doctor's and its hospital's
/// list.
///
/// ```
/// use match_learn::contracts::{Contract, cumulative_offer_process};
///
/// // One post, one applicant, two possible terms; the employer only accepts the
/// // higher commitment (terms 1), so that is the contract signed.
/// let contracts = vec![
///     Contract { doctor: 0, hospital: 0, terms: 0 },
///     Contract { doctor: 0, hospital: 0, terms: 1 },
/// ];
/// let doctor_prefs = vec![vec![0, 1]]; // applicant prefers the lighter terms 0
/// let hospital_ranks = vec![vec![1]];  // employer only accepts terms 1
/// let m = cumulative_offer_process(&contracts, &doctor_prefs, &hospital_ranks, &[1]);
/// assert_eq!(m.doctor[0], Some(1)); // it signs the higher-commitment contract
/// ```
pub fn cumulative_offer_process(
    contracts: &[Contract],
    doctor_prefs: &[Vec<usize>],
    hospital_ranks: &[Vec<usize>],
    capacities: &[usize],
) -> ContractMatch {
    let n_d = doctor_prefs.len();
    let n_h = hospital_ranks.len();
    let n_c = contracts.len();
    assert_eq!(capacities.len(), n_h, "one capacity per hospital");

    let mut next = vec![0usize; n_d]; // next contract index doctor d will offer
    let mut offered = vec![false; n_c]; // contract has been offered to its hospital

    loop {
        // Each hospital holds its choice from all contracts offered to it so far.
        let chosen: Vec<Vec<usize>> = (0..n_h)
            .map(|h| choose(&hospital_ranks[h], &offered, capacities[h], contracts, n_d))
            .collect();
        let mut held = vec![false; n_d];
        for set in &chosen {
            for &c in set {
                held[contracts[c].doctor] = true;
            }
        }

        // The first unassigned doctor with an un-offered contract makes an offer.
        let mut offerer = None;
        for d in 0..n_d {
            if !held[d] && next[d] < doctor_prefs[d].len() {
                offerer = Some(d);
                break;
            }
        }
        match offerer {
            Some(d) => {
                let c = doctor_prefs[d][next[d]];
                next[d] += 1;
                offered[c] = true; // offers accumulate: never withdrawn
            }
            None => {
                let mut doctor = vec![None; n_d];
                for set in &chosen {
                    for &c in set {
                        doctor[contracts[c].doctor] = Some(c);
                    }
                }
                return ContractMatch {
                    doctor,
                    hospital: chosen,
                };
            }
        }
    }
}

/// Whether `m` is a stable allocation for the contracts instance.
///
/// Stability has two parts. *Individual rationality*: every hospital holds
/// exactly its own choice from what it holds (it wastes no seat and keeps no
/// contract it would drop), and every doctor's contract is on its list. *No
/// blocking contract*: there is no unsigned contract `z` whose doctor strictly
/// prefers it to its current assignment and whose hospital would choose it
/// alongside what it already holds.
pub fn is_stable_with_contracts(
    contracts: &[Contract],
    doctor_prefs: &[Vec<usize>],
    hospital_ranks: &[Vec<usize>],
    capacities: &[usize],
    m: &ContractMatch,
) -> bool {
    let n_d = doctor_prefs.len();
    let n_h = hospital_ranks.len();
    let n_c = contracts.len();
    let drank = |d: usize, c: usize| doctor_prefs[d].iter().position(|&x| x == c);

    // Individual rationality, hospital side: each held set is its own choice.
    for h in 0..n_h {
        let mut off = vec![false; n_c];
        for &c in &m.hospital[h] {
            off[c] = true;
        }
        let mut ch = choose(&hospital_ranks[h], &off, capacities[h], contracts, n_d);
        ch.sort_unstable();
        let mut held = m.hospital[h].clone();
        held.sort_unstable();
        if ch != held {
            return false;
        }
    }
    // Individual rationality, doctor side: each held contract is acceptable.
    for (d, slot) in m.doctor.iter().enumerate() {
        if let Some(c) = slot
            && (contracts[*c].doctor != d || drank(d, *c).is_none())
        {
            return false;
        }
    }
    // No blocking contract.
    for (z, ct) in contracts.iter().enumerate() {
        let Some(zr) = drank(ct.doctor, z) else {
            continue; // unacceptable to its doctor: cannot block
        };
        let d_wants = match m.doctor[ct.doctor] {
            Some(cur) => zr < drank(ct.doctor, cur).expect("held contract is acceptable"),
            None => true,
        };
        if !d_wants {
            continue;
        }
        let mut off = vec![false; n_c];
        for &c in &m.hospital[ct.hospital] {
            off[c] = true;
        }
        off[z] = true;
        let ch = choose(
            &hospital_ranks[ct.hospital],
            &off,
            capacities[ct.hospital],
            contracts,
            n_d,
        );
        if ch.contains(&z) {
            return false; // doctor and hospital would both sign z
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::many_to_one::hospital_residents;
    use crate::rng::Rng;

    /// A contracts instance: contracts, doctor prefs, hospital ranks, capacities.
    type Instance = (Vec<Contract>, Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>);

    #[test]
    fn reduces_to_hospital_residents() {
        // One contract per doctor-hospital pair (trivial terms): the cumulative
        // offer process must reproduce Hospital-Residents deferred acceptance.
        let mut rng = Rng::new(0xC047);
        for _ in 0..400 {
            let n_d = 1 + rng.below(4);
            let n_h = 1 + rng.below(3);
            let prop: Vec<Vec<usize>> = (0..n_d).map(|_| rng.permutation(n_h)).collect();
            let recv: Vec<Vec<usize>> = (0..n_h).map(|_| rng.permutation(n_d)).collect();
            let cap: Vec<usize> = (0..n_h).map(|_| 1 + rng.below(2)).collect();

            let contracts: Vec<Contract> = (0..n_d)
                .flat_map(|d| {
                    (0..n_h).map(move |h| Contract {
                        doctor: d,
                        hospital: h,
                        terms: 0,
                    })
                })
                .collect();
            let cid = |d: usize, h: usize| d * n_h + h;
            let doctor_prefs: Vec<Vec<usize>> = (0..n_d)
                .map(|d| prop[d].iter().map(|&h| cid(d, h)).collect())
                .collect();
            let hospital_ranks: Vec<Vec<usize>> = (0..n_h)
                .map(|h| recv[h].iter().map(|&d| cid(d, h)).collect())
                .collect();

            let m = cumulative_offer_process(&contracts, &doctor_prefs, &hospital_ranks, &cap);
            let hr = hospital_residents(&prop, &recv, &cap);
            let got: Vec<Option<usize>> = m
                .doctor
                .iter()
                .map(|s| s.map(|c| contracts[c].hospital))
                .collect();
            assert_eq!(
                got, hr.proposer,
                "COP != HR for prop={prop:?} recv={recv:?} cap={cap:?}"
            );
        }
    }

    /// Build a random small contracts instance (a few terms per pair).
    fn random_instance(rng: &mut Rng) -> Instance {
        let n_d = 1 + rng.below(3);
        let n_h = 1 + rng.below(2);
        let mut contracts = Vec::new();
        for d in 0..n_d {
            for h in 0..n_h {
                let terms = 1 + rng.below(2); // one or two contracts per pair
                for t in 0..terms {
                    contracts.push(Contract {
                        doctor: d,
                        hospital: h,
                        terms: t,
                    });
                }
            }
        }
        let n_c = contracts.len();
        // Each doctor ranks a random ordering of its own contracts.
        let doctor_prefs: Vec<Vec<usize>> = (0..n_d)
            .map(|d| {
                let mut own: Vec<usize> = (0..n_c).filter(|&c| contracts[c].doctor == d).collect();
                shuffle(rng, &mut own);
                own
            })
            .collect();
        // Each hospital ranks a random acceptable subset of its contracts.
        let hospital_ranks: Vec<Vec<usize>> = (0..n_h)
            .map(|h| {
                let mut own: Vec<usize> = (0..n_c)
                    .filter(|&c| contracts[c].hospital == h)
                    .filter(|_| rng.below(4) != 0) // drop ~1/4 as unacceptable
                    .collect();
                shuffle(rng, &mut own);
                own
            })
            .collect();
        let cap: Vec<usize> = (0..n_h).map(|_| 1 + rng.below(2)).collect();
        (contracts, doctor_prefs, hospital_ranks, cap)
    }

    fn shuffle(rng: &mut Rng, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            v.swap(i, rng.below(i + 1));
        }
    }

    #[test]
    fn cop_is_stable() {
        let mut rng = Rng::new(0xC048);
        for _ in 0..3000 {
            let (contracts, dp, hr, cap) = random_instance(&mut rng);
            let m = cumulative_offer_process(&contracts, &dp, &hr, &cap);
            assert!(
                is_stable_with_contracts(&contracts, &dp, &hr, &cap, &m),
                "COP unstable: contracts={contracts:?} dp={dp:?} hr={hr:?} cap={cap:?}"
            );
        }
    }

    #[test]
    fn cop_matches_brute_force_oracle() {
        let mut rng = Rng::new(0xC049);
        for _ in 0..1500 {
            let (contracts, dp, hr, cap) = random_instance(&mut rng);
            let n_d = dp.len();
            let n_h = hr.len();
            let n_c = contracts.len();
            let m = cumulative_offer_process(&contracts, &dp, &hr, &cap);

            // Enumerate every contract subset that gives each doctor at most one
            // contract; keep the stable ones.
            let mut oracle: Vec<Vec<usize>> = Vec::new();
            for mask in 0u32..(1 << n_c) {
                let set: Vec<usize> = (0..n_c).filter(|&c| mask & (1 << c) != 0).collect();
                let mut per_doctor = vec![0usize; n_d];
                for &c in &set {
                    per_doctor[contracts[c].doctor] += 1;
                }
                if per_doctor.iter().any(|&k| k > 1) {
                    continue;
                }
                let mut doctor = vec![None; n_d];
                let mut hospital = vec![Vec::new(); n_h];
                for &c in &set {
                    doctor[contracts[c].doctor] = Some(c);
                    hospital[contracts[c].hospital].push(c);
                }
                let cand = ContractMatch { doctor, hospital };
                if is_stable_with_contracts(&contracts, &dp, &hr, &cap, &cand) {
                    let mut s = set.clone();
                    s.sort_unstable();
                    oracle.push(s);
                }
            }

            let mut got: Vec<usize> = m.doctor.iter().filter_map(|&s| s).collect();
            got.sort_unstable();
            assert!(
                oracle.contains(&got),
                "COP result not in stable oracle: got={got:?} oracle={oracle:?}"
            );
        }
    }

    #[test]
    fn terms_let_a_doctor_win_by_committing() {
        // Two cadets compete for one branch seat. The branch prefers a longer
        // service commitment (terms 1) to a shorter one (terms 0), cadet 0 over
        // cadet 1 at equal terms. Cadets would rather serve short. To win the
        // single seat, cadet 0 must offer long service.
        let contracts = vec![
            Contract {
                doctor: 0,
                hospital: 0,
                terms: 0,
            }, // c0: cadet0 short
            Contract {
                doctor: 0,
                hospital: 0,
                terms: 1,
            }, // c1: cadet0 long
            Contract {
                doctor: 1,
                hospital: 0,
                terms: 0,
            }, // c2: cadet1 short
            Contract {
                doctor: 1,
                hospital: 0,
                terms: 1,
            }, // c3: cadet1 long
        ];
        let doctor_prefs = vec![vec![0, 1], vec![2, 3]]; // each prefers short
        let hospital_ranks = vec![vec![1, 3, 0, 2]]; // long first, cadet 0 ahead
        let m = cumulative_offer_process(&contracts, &doctor_prefs, &hospital_ranks, &[1]);
        assert_eq!(m.doctor[0], Some(1), "cadet 0 signs long service to win");
        assert_eq!(m.doctor[1], None, "cadet 1 is squeezed out");
        assert!(is_stable_with_contracts(
            &contracts,
            &doctor_prefs,
            &hospital_ranks,
            &[1],
            &m
        ));
    }

    #[test]
    fn detects_a_blocking_contract() {
        // The empty allocation is unstable: an acceptable contract blocks it.
        let contracts = vec![Contract {
            doctor: 0,
            hospital: 0,
            terms: 0,
        }];
        let dp = vec![vec![0]];
        let hr = vec![vec![0]];
        let empty = ContractMatch {
            doctor: vec![None],
            hospital: vec![Vec::new()],
        };
        assert!(!is_stable_with_contracts(
            &contracts,
            &dp,
            &hr,
            &[1],
            &empty
        ));
        let m = cumulative_offer_process(&contracts, &dp, &hr, &[1]);
        assert_eq!(m.doctor[0], Some(0));
        assert!(is_stable_with_contracts(&contracts, &dp, &hr, &[1], &m));
    }
}
