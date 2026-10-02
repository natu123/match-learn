//! Original exact dynamic-update controls; not a TBLS implementation or benchmark.
#[path = "support/tiny_matching_oracle.rs"]
mod oracle;

use match_learn::matching::Matching;
use match_learn::ties::is_weakly_stable;
use oracle::{Profile, Stability};

fn diagonal() -> Matching {
    Matching {
        proposer: vec![Some(0), Some(1)],
        receiver: vec![Some(0), Some(1)],
    }
}

// Preserve numeric IDs (including inactive slots). Rebuild both views after
// dropping newly unacceptable edges; never truncate or renumber one side.
fn project(p: &Profile, r: &Profile, old: &Matching) -> Matching {
    let mut out = Matching {
        proposer: vec![None; p.len()],
        receiver: vec![None; r.len()],
    };
    for (i, partner) in old.proposer.iter().enumerate() {
        if let Some(j) = *partner
            && p.get(i)
                .is_some_and(|row| row.iter().any(|tier| tier.contains(&j)))
            && r.get(j)
                .is_some_and(|row| row.iter().any(|tier| tier.contains(&i)))
        {
            assert!(out.receiver[j].is_none());
            out.proposer[i] = Some(j);
            out.receiver[j] = Some(i);
        }
    }
    out
}

fn audit(p: Profile, r: Profile, expected: (usize, usize), loss: (usize, usize, usize)) {
    let old = diagonal();
    let tied = vec![vec![vec![0, 1]]; 2];
    assert!(is_weakly_stable(&tied, &tied, &old));
    let projected = project(&p, &r, &old);
    assert!(oracle::feasible_matchings(&p, &r).contains(&projected));
    let stable = oracle::stable_matchings(&p, &r, Stability::Weak);
    assert!(stable.iter().all(|m| is_weakly_stable(&p, &r, m)));
    for stronger in [Stability::Strong, Stability::Super] {
        assert!(
            oracle::stable_matchings(&p, &r, stronger)
                .iter()
                .all(|m| stable.contains(m))
        );
    }
    let retention = |m: &Matching| {
        m.proposer
            .iter()
            .enumerate()
            .filter(|&(i, x)| x.is_some() && old.proposer.get(i) == Some(x))
            .count()
    };
    let optimum = stable
        .iter()
        .map(|m| (m.pairs(), retention(m)))
        .max()
        .unwrap();
    assert_eq!(optimum, expected);
    assert_eq!(oracle::cardinality_retention(&p, &r, &old), Some(expected));
    let any_size = stable.iter().map(retention).max().unwrap();
    let eligible = projected.pairs();
    let observed = (
        old.pairs() - eligible,
        eligible - any_size,
        any_size - optimum.1,
    );
    assert_eq!(observed, loss);
    assert_eq!(loss.0 + loss.1 + loss.2, old.pairs() - optimum.1);
}

#[test]
fn unilateral_matched_edge_deletion_separates_infeasibility_and_cardinality_loss() {
    audit(
        vec![vec![vec![1]], vec![vec![0, 1]]],
        vec![vec![vec![0, 1]]; 2],
        (2, 0),
        (1, 0, 1),
    );
}

#[test]
fn proposer_departure_keeps_ids_and_reciprocal_surviving_pair() {
    audit(
        vec![vec![vec![0, 1]], vec![]],
        vec![vec![vec![0]]; 2],
        (1, 1),
        (1, 0, 0),
    );
}

#[test]
fn one_sided_tie_split_can_leave_previous_matching_stable() {
    audit(
        vec![vec![vec![1], vec![0]], vec![vec![0, 1]]],
        vec![vec![vec![0, 1]]; 2],
        (2, 2),
        (0, 0, 0),
    );
}

#[test]
fn two_changed_rows_can_force_loss_without_deleting_old_edges() {
    audit(
        vec![vec![vec![1], vec![0]], vec![vec![0, 1]]],
        vec![vec![vec![0, 1]], vec![vec![0], vec![1]]],
        (2, 0),
        (0, 2, 0),
    );
}
