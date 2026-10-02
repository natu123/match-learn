#[path = "support/tiny_matching_oracle.rs"]
mod oracle;

use match_learn::matching::{Matching, all_stable_matchings, gale_shapley, is_stable};
use match_learn::rng::Rng;
use match_learn::ties::{
    is_strongly_stable, is_super_stable, is_weakly_stable, strongly_stable, super_stable,
    super_stable_irving, weakly_stable,
};
use oracle::{Profile, Stability};

// Every ordered subset / ordered partition on at most two partners.
fn profiles(agents: usize, partners: usize, ties: bool) -> Vec<Profile> {
    let mut rows = vec![vec![]];
    if partners >= 1 {
        rows.push(vec![vec![0]]);
    }
    if partners == 2 {
        rows.extend([
            vec![vec![1]],
            vec![vec![0], vec![1]],
            vec![vec![1], vec![0]],
        ]);
        if ties {
            rows.push(vec![vec![0, 1]]);
        }
    }
    let mut out = vec![vec![]];
    for _ in 0..agents {
        out = out
            .iter()
            .flat_map(|prefix| {
                rows.iter().map(move |row| {
                    let mut p = prefix.clone();
                    p.push(row.clone());
                    p
                })
            })
            .collect();
    }
    out
}

fn assert_same_set(a: &[Matching], b: &[Matching]) {
    assert_eq!(a.len(), b.len(), "actual={a:?} expected={b:?}");
    assert!(
        a.iter().all(|m| b.contains(m)) && b.iter().all(|m| a.contains(m)),
        "actual={a:?} expected={b:?}"
    );
}

#[test]
fn exhaustive_strict_profiles_through_two_by_two() {
    let mut count = 0;
    for np in 0..=2 {
        for nr in 0..=2 {
            for p in profiles(np, nr, false) {
                for r in profiles(nr, np, false) {
                    count += 1;
                    let ps: Vec<_> = p
                        .iter()
                        .map(|tiers| tiers.iter().flatten().copied().collect())
                        .collect();
                    let rs: Vec<_> = r
                        .iter()
                        .map(|tiers| tiers.iter().flatten().copied().collect())
                        .collect();
                    let expected = oracle::stable_matchings(&p, &r, Stability::Weak);
                    assert_same_set(&all_stable_matchings(&ps, &rs), &expected);
                    assert!(expected.contains(&gale_shapley(&ps, &rs)));
                    let full_p = vec![vec![(0..nr).collect()]; np];
                    let full_r = vec![vec![(0..np).collect()]; nr];
                    for m in oracle::feasible_matchings(&full_p, &full_r) {
                        assert_eq!(is_stable(&ps, &rs, &m), expected.contains(&m));
                    }
                }
            }
        }
    }
    assert_eq!(count, 674);
}

#[test]
fn exhaustive_tied_profiles_through_two_by_two() {
    let mut count = 0;
    let mut rng = Rng::new(17);
    for np in 0..=2 {
        for nr in 0..=2 {
            for p in profiles(np, nr, true) {
                for r in profiles(nr, np, true) {
                    count += 1;
                    let weak = oracle::stable_matchings(&p, &r, Stability::Weak);
                    let strong = oracle::stable_matchings(&p, &r, Stability::Strong);
                    let sup = oracle::stable_matchings(&p, &r, Stability::Super);
                    assert!(weak.contains(&weakly_stable(&p, &r, &mut rng)));
                    for (actual, expected) in [
                        (strongly_stable(&p, &r), &strong),
                        (super_stable(&p, &r), &sup),
                        (super_stable_irving(&p, &r), &sup),
                    ] {
                        assert_eq!(actual.is_some(), !expected.is_empty(), "p={p:?} r={r:?}");
                        if let Some(m) = actual {
                            assert!(expected.contains(&m), "p={p:?} r={r:?} m={m:?}");
                        }
                    }
                    // Include unacceptable assignments in validator comparisons by
                    // generating all injective assignments on the complete market.
                    let full_p = vec![vec![(0..nr).collect()]; np];
                    let full_r = vec![vec![(0..np).collect()]; nr];
                    for m in oracle::feasible_matchings(&full_p, &full_r) {
                        assert_eq!(is_weakly_stable(&p, &r, &m), weak.contains(&m));
                        assert_eq!(is_strongly_stable(&p, &r, &m), strong.contains(&m));
                        assert_eq!(is_super_stable(&p, &r, &m), sup.contains(&m));
                    }
                }
            }
        }
    }
    assert_eq!(count, 1353);
}

#[test]
fn oracle_hand_verified_cardinality_retention_tradeoff() {
    // Exactly two weak-stable matchings: {0-0} and {0-1, 1-0}.
    let p = vec![vec![vec![0, 1]], vec![vec![0]]];
    let r = p.clone();
    let small = Matching {
        proposer: vec![Some(0), None],
        receiver: vec![Some(0), None],
    };
    let large = Matching {
        proposer: vec![Some(1), Some(0)],
        receiver: vec![Some(1), Some(0)],
    };
    assert_same_set(
        &oracle::stable_matchings(&p, &r, Stability::Weak),
        &[small.clone(), large.clone()],
    );
    assert_same_set(
        &oracle::stable_matchings(&p, &r, Stability::Strong),
        std::slice::from_ref(&large),
    );
    assert!(oracle::stable_matchings(&p, &r, Stability::Super).is_empty());
    assert_eq!(oracle::cardinality_retention(&p, &r, &small), Some((2, 0)));
    assert_eq!(oracle::cardinality_retention(&p, &r, &large), Some((2, 2)));
}

#[test]
fn oracle_hand_verified_complete_ties_and_strict_cycle() {
    let diagonal = Matching {
        proposer: vec![Some(0), Some(1)],
        receiver: vec![Some(0), Some(1)],
    };
    let crossed = Matching {
        proposer: vec![Some(1), Some(0)],
        receiver: vec![Some(1), Some(0)],
    };
    let tied = vec![vec![vec![0, 1]]; 2];
    // Seven feasible assignments: empty, four single pairs, two perfect.
    assert_eq!(oracle::feasible_matchings(&tied, &tied).len(), 7);
    for kind in [Stability::Weak, Stability::Strong] {
        assert_same_set(
            &oracle::stable_matchings(&tied, &tied, kind),
            &[diagonal.clone(), crossed.clone()],
        );
    }
    assert!(oracle::stable_matchings(&tied, &tied, Stability::Super).is_empty());
    // Both perfect assignments are stable: proposers prefer the diagonal,
    // receivers the crossed assignment. Partial assignments all block.
    let p = vec![vec![vec![0], vec![1]], vec![vec![1], vec![0]]];
    let r = vec![vec![vec![1], vec![0]], vec![vec![0], vec![1]]];
    for kind in [Stability::Weak, Stability::Strong, Stability::Super] {
        assert_same_set(
            &oracle::stable_matchings(&p, &r, kind),
            &[diagonal.clone(), crossed.clone()],
        );
    }
}
