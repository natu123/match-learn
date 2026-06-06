//! The Boston (immediate-acceptance) mechanism for school choice.
//!
//! The textbook alternative to [Gale-Shapley deferred acceptance](crate::many_to_one):
//! in round `k`, every still-unassigned student applies to its `k`-th choice, and
//! each school *immediately and permanently* admits the highest-priority
//! applicants up to its remaining capacity. There is no deferral — a seat given
//! in round 1 is never reclaimed.
//!
//! That one change has large consequences, and this module exists to make them
//! visible next to the deferred-acceptance core:
//!
//! - Boston is **not stable**: a student can have priority at a school that filled
//!   up with lower-priority students in an earlier round, leaving a blocking pair.
//! - Boston is **not strategy-proof**: a student who would be rejected from its
//!   true top choice can do better by ranking a school it can actually win first
//!   (it "wastes" no round on a lost cause). This is why real districts (Boston
//!   itself, 2005) replaced it with deferred acceptance.
//! - Truthful Boston *is* Pareto efficient for students with respect to the
//!   reported preferences — the property that made it tempting.

use crate::many_to_one::ManyToOne;
use crate::matching::rank_table;

/// Run the Boston (immediate-acceptance) mechanism.
///
/// `student_prefs[s]` ranks schools; `school_priorities[c]` ranks students;
/// `capacities[c]` is school `c`'s number of seats. A student/school pair is
/// acceptable only if each appears on the other's list. Returns a [`ManyToOne`]
/// with students as proposers and schools as receivers.
///
/// ```
/// use match_learn::boston_mechanism;
///
/// // Both students want school 0 first; it has one seat and prefers student 1.
/// // Student 0 is rejected in round 1 and lands its second choice, school 1.
/// let students = vec![vec![0, 1], vec![0, 1]];
/// let schools = vec![vec![1, 0], vec![0, 1]];
/// let m = boston_mechanism(&students, &schools, &[1, 1]);
/// assert_eq!(m.proposer, vec![Some(1), Some(0)]);
/// ```
pub fn boston_mechanism(
    student_prefs: &[Vec<usize>],
    school_priorities: &[Vec<usize>],
    capacities: &[usize],
) -> ManyToOne {
    let n_s = student_prefs.len();
    let n_c = school_priorities.len();
    assert_eq!(capacities.len(), n_c, "one capacity per school");
    let prio = rank_table(school_priorities, n_s);

    let mut assigned: Vec<Option<usize>> = vec![None; n_s];
    let mut admitted: Vec<Vec<usize>> = vec![Vec::new(); n_c];
    let mut remaining = capacities.to_vec();
    let mut next = vec![0usize; n_s]; // next choice index per student

    loop {
        // Each unassigned student applies to its next listed school this round.
        let mut applicants: Vec<Vec<usize>> = vec![Vec::new(); n_c];
        let mut any = false;
        for s in 0..n_s {
            if assigned[s].is_some() {
                continue;
            }
            while next[s] < student_prefs[s].len() {
                let c = student_prefs[s][next[s]];
                next[s] += 1;
                if c < n_c && prio[c][s].is_some() {
                    applicants[c].push(s);
                    any = true;
                    break; // one application per student per round
                }
                // unacceptable / out of range: skip to the next listed school
            }
        }
        if !any {
            break; // nobody applied: everyone is assigned or exhausted
        }

        // Each school admits the highest-priority applicants up to remaining seats,
        // permanently. The rest are rejected and try again next round.
        for c in 0..n_c {
            if remaining[c] == 0 || applicants[c].is_empty() {
                continue;
            }
            applicants[c].sort_by_key(|&s| prio[c][s].expect("applicant is acceptable"));
            for &s in applicants[c].iter().take(remaining[c]) {
                assigned[s] = Some(c);
                admitted[c].push(s);
            }
            remaining[c] = remaining[c].saturating_sub(applicants[c].len());
        }
    }

    ManyToOne {
        proposer: assigned,
        receiver: admitted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation::is_pareto_efficient;
    use crate::many_to_one::{hospital_residents, is_stable};
    use crate::rng::Rng;

    /// The canonical school-choice instance separating Boston from deferred
    /// acceptance: student 0 has priority at school 1 but loses it to student 2,
    /// who reached it first.
    fn classic() -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>) {
        let students = vec![
            vec![0, 1, 2], // s0
            vec![0, 2, 1], // s1
            vec![1, 0, 2], // s2
        ];
        let schools = vec![
            vec![1, 0, 2], // c0 prefers s1
            vec![0, 2, 1], // c1 prefers s0 over s2
            vec![0, 1, 2], // c2
        ];
        (students, schools, vec![1, 1, 1])
    }

    #[test]
    fn boston_can_be_unstable_where_deferred_acceptance_is_not() {
        let (students, schools, cap) = classic();
        let b = boston_mechanism(&students, &schools, &cap);
        // Boston: s0 -> c2 (lost c0 to s1, then c1 was already taken by s2).
        assert_eq!(b.proposer, vec![Some(2), Some(0), Some(1)]);
        // (s0, c1) blocks: s0 prefers c1 to c2 and c1 prioritizes s0 over s2.
        assert!(!is_stable(&students, &schools, &cap, &b));
        // Deferred acceptance on the same instance is stable.
        let da = hospital_residents(&students, &schools, &cap);
        assert!(is_stable(&students, &schools, &cap, &da));
    }

    #[test]
    fn boston_is_manipulable() {
        // Student 0 does strictly better by ranking c1 (winnable) ahead of c0
        // (a lost cause): it secures c1 (true rank 1) instead of c2 (true rank 2).
        let (students, schools, cap) = classic();
        let truthful = boston_mechanism(&students, &schools, &cap);
        assert_eq!(truthful.proposer[0], Some(2));

        let mut lied = students.clone();
        lied[0] = vec![1, 0, 2]; // report c1 first
        let manip = boston_mechanism(&lied, &schools, &cap);
        assert_eq!(manip.proposer[0], Some(1));
        // c1 (true rank 1) is strictly preferred to c2 (true rank 2).
        let rank = |o: usize| students[0].iter().position(|&x| x == o).unwrap();
        assert!(rank(1) < rank(2), "the misreport must strictly improve s0");
    }

    #[test]
    fn truthful_boston_is_pareto_efficient_for_students() {
        // Capacity-1 schools: the truthful Boston outcome admits no student-side
        // Pareto improvement (its one redeeming property).
        let mut rng = Rng::new(0xB05);
        for _ in 0..2000 {
            let n = 1 + rng.below(5);
            let students: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let schools: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
            let cap = vec![1usize; n];
            let b = boston_mechanism(&students, &schools, &cap);
            assert!(
                is_pareto_efficient(&students, n, &b.proposer),
                "Boston not Pareto efficient: students={students:?} schools={schools:?} b={:?}",
                b.proposer
            );
        }
    }

    #[test]
    fn respects_capacities_and_acceptability() {
        let mut rng = Rng::new(0xB06);
        for _ in 0..1000 {
            let n_s = 1 + rng.below(5);
            let n_c = 1 + rng.below(4);
            let students: Vec<Vec<usize>> = (0..n_s).map(|_| rng.permutation(n_c)).collect();
            let schools: Vec<Vec<usize>> = (0..n_c).map(|_| rng.permutation(n_s)).collect();
            let cap: Vec<usize> = (0..n_c).map(|_| 1 + rng.below(2)).collect();
            let b = boston_mechanism(&students, &schools, &cap);
            for (c, seats) in b.receiver.iter().enumerate() {
                assert!(seats.len() <= cap[c], "school {c} over capacity");
            }
            // The two views agree.
            for (s, slot) in b.proposer.iter().enumerate() {
                if let Some(c) = slot {
                    assert!(b.receiver[*c].contains(&s));
                }
            }
        }
    }
}
