use match_learn::matching::{Matching, all_stable_matchings, is_stable};
use match_learn::ties::{
    is_strongly_stable, is_super_stable, is_weakly_stable, strongly_stable, super_stable,
};

#[test]
fn empty_lists_reject_assigned_pair() {
    let prefs = vec![vec![]];
    let tiers = vec![vec![]];
    let assigned = Matching {
        proposer: vec![Some(0)],
        receiver: vec![Some(0)],
    };
    assert!(!is_stable(&prefs, &prefs, &assigned));
    assert!(!is_weakly_stable(&tiers, &tiers, &assigned));
    assert!(!is_strongly_stable(&tiers, &tiers, &assigned));
    assert!(!is_super_stable(&tiers, &tiers, &assigned));
    let unmatched = Matching {
        proposer: vec![None],
        receiver: vec![None],
    };
    assert_eq!(
        all_stable_matchings(&prefs, &prefs),
        vec![unmatched.clone()]
    );
    assert_eq!(strongly_stable(&tiers, &tiers), Some(unmatched.clone()));
    assert_eq!(super_stable(&tiers, &tiers), Some(unmatched));
}

#[test]
fn malformed_assignments_return_false_without_panicking() {
    let prefs = vec![vec![0, 1], vec![0, 1]];
    let tiers = vec![vec![vec![0, 1]], vec![vec![0, 1]]];
    let malformed = [
        (vec![], vec![None, None]),
        (vec![None, None, None], vec![None, None]),
        (vec![None, None], vec![]),
        (vec![None, None], vec![None, None, None]),
        (vec![Some(2), None], vec![None, None]),
        (vec![None, None], vec![Some(2), None]),
        (vec![Some(usize::MAX), None], vec![None, None]),
        (vec![None, None], vec![Some(usize::MAX), None]),
        (vec![Some(0), None], vec![None, None]),
        (vec![None, None], vec![Some(0), None]),
        (vec![Some(0), Some(0)], vec![Some(0), None]),
        (vec![Some(0), None], vec![Some(0), Some(0)]),
        (vec![Some(0), Some(1)], vec![Some(1), Some(0)]),
    ];
    for (proposer, receiver) in malformed {
        let m = Matching { proposer, receiver };
        assert!(!is_stable(&prefs, &prefs, &m), "{m:?}");
        assert!(!is_weakly_stable(&tiers, &tiers, &m), "{m:?}");
        assert!(!is_strongly_stable(&tiers, &tiers, &m), "{m:?}");
        assert!(!is_super_stable(&tiers, &tiers, &m), "{m:?}");
    }
}

#[test]
fn acceptability_is_required_on_both_sides() {
    let m = Matching {
        proposer: vec![Some(0)],
        receiver: vec![Some(0)],
    };
    for (p, r) in [(vec![vec![]], vec![vec![0]]), (vec![vec![0]], vec![vec![]])] {
        let pt: Vec<_> = p
            .iter()
            .map(|row| row.iter().map(|&x| vec![x]).collect())
            .collect();
        let rt: Vec<_> = r
            .iter()
            .map(|row| row.iter().map(|&x| vec![x]).collect())
            .collect();
        assert!(!is_stable(&p, &r, &m));
        assert!(!is_weakly_stable(&pt, &rt, &m));
        assert!(!is_strongly_stable(&pt, &rt, &m));
        assert!(!is_super_stable(&pt, &rt, &m));
    }
}
