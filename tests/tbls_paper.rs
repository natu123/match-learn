//! Bounded paper-reimplementation correctness checks; no SOTA/time claim.
#[path = "support/tiny_matching_oracle.rs"]
mod oracle;
#[path = "../research/tbls.rs"]
mod tbls;
use match_learn::matching::is_stable;
use match_learn::rng::Rng;
use oracle::{Profile, Stability};
use tbls::{Config, RepairBudget};

fn cfg() -> Config {
    Config {
        max_iterations: 16,
        repair_budget: Some(RepairBudget::CandidateChecks(256)),
        ..Config::default()
    }
}

fn check(p: &Profile, r: &Profile, seed: u64) -> tbls::Output {
    let out = tbls::solve(p, r, seed, &cfg()).unwrap();
    let weak = oracle::stable_matchings(p, r, Stability::Weak);
    assert!(weak.contains(&out.matching));
    let optimum = oracle::cardinality_retention(p, r, &out.initial).unwrap();
    assert!(out.matching.pairs() <= optimum.0);
    assert!(is_stable(
        &out.strategy.proposer,
        &out.strategy.receiver,
        &out.matching
    ));
    for stronger in [Stability::Strong, Stability::Super] {
        assert!(
            oracle::stable_matchings(p, r, stronger)
                .iter()
                .all(|m| weak.contains(m))
        );
    }
    assert_eq!(out.stats.validation_fallbacks, 0);
    assert!(out.score.is_finite() && out.big_m.is_finite());
    assert!(out.stats.iterations <= cfg().max_iterations);
    for (tiers, strict) in [(p, &out.strategy.proposer), (r, &out.strategy.receiver)] {
        for (row, order) in tiers.iter().zip(strict) {
            assert_eq!(row.iter().map(Vec::len).sum::<usize>(), order.len());
            let mut index = 0;
            for tier in row {
                let mut expected = tier.clone();
                expected.sort_unstable();
                let mut actual = order[index..index + tier.len()].to_vec();
                actual.sort_unstable();
                assert_eq!(actual, expected);
                index += tier.len();
            }
        }
    }
    let repeat = tbls::solve(p, r, seed, &cfg()).unwrap();
    assert_eq!(out.matching, repeat.matching);
    assert_eq!(out.strategy, repeat.strategy);
    assert_eq!(out.stats, repeat.stats);
    assert_eq!(out.score, repeat.score);
    // Calibration is observational even in work mode; not reproducibility truth.
    let _calibration = out.calibrated_repair_time;
    out
}

#[test]
fn exhaustive_reciprocal_tied_profiles_through_two_by_two() {
    let rows = [
        vec![],
        vec![vec![0]],
        vec![vec![1]],
        vec![vec![0], vec![1]],
        vec![vec![1], vec![0]],
        vec![vec![0, 1]],
    ];
    let mut checked = 0;
    for a in &rows {
        for b in &rows {
            for c in &rows {
                for d in &rows {
                    let p = vec![a.clone(), b.clone()];
                    let r = vec![c.clone(), d.clone()];
                    let acceptable = |profile: &Profile, i: usize, j: usize| {
                        profile[i].iter().any(|t| t.contains(&j))
                    };
                    if (0..2).any(|i| (0..2).any(|j| acceptable(&p, i, j) != acceptable(&r, j, i)))
                    {
                        continue;
                    }
                    for seed in [17, 29] {
                        check(&p, &r, seed);
                    }
                    checked += 1;
                }
            }
        }
    }
    // By reciprocal graph: 1 empty + 4 single-edge + 2 disjoint-edge +
    // 4 shared-endpoint graphs * 3 orders + 4 three-edge graphs * 9 orders +
    // 1 full graph * 81 orders = 136. Asymmetric lists are rejected, not sampled.
    assert_eq!(checked, 136);
}

#[test]
fn held_out_small_synthetic_profiles_have_checked_stable_incumbents() {
    // Separate from tiny exhaustive controls; reciprocal edges, independently
    // shuffled sides, short ties. 24 fixed input seeds, n=3 or n=4.
    for seed in 1001..=1024 {
        let n = 3 + (seed as usize % 2);
        let mut rng = Rng::new(seed);
        let mut p = vec![vec![]; n];
        let mut r = vec![vec![]; n];
        for (i, row) in p.iter_mut().enumerate() {
            for (j, receiver) in r.iter_mut().enumerate() {
                if rng.uniform() < 0.6 {
                    row.push(vec![j]);
                    receiver.push(vec![i]);
                }
            }
        }
        for profile in [&mut p, &mut r] {
            for row in profile {
                rng.shuffle(row);
                if row.len() >= 2 && rng.uniform() < 0.7 {
                    let second = row.remove(1);
                    row[0].extend(second);
                }
            }
        }
        for search_seed in [20001, 20002] {
            check(&p, &r, search_seed);
        }
    }
}

#[test]
fn paper_table_one_control_reaches_perfect_matching_in_bounded_runs() {
    let p = vec![
        vec![vec![0, 2], vec![1]],
        vec![vec![0], vec![1], vec![3]],
        vec![vec![0]],
        vec![vec![1]],
    ];
    let r = vec![
        vec![vec![0], vec![2], vec![1]],
        vec![vec![1, 3], vec![0]],
        vec![vec![0]],
        vec![vec![1]],
    ];
    assert_eq!(
        oracle::stable_matchings(&p, &r, Stability::Weak)
            .iter()
            .map(|m| m.pairs())
            .max(),
        Some(4)
    );
    for seed in [17, 29, 20001, 20002] {
        assert_eq!(check(&p, &r, seed).matching.pairs(), 4);
    }
}

#[test]
fn calibrated_time_mode_still_emits_strictly_and_weakly_stable_result() {
    let p = vec![vec![vec![0, 1]], vec![vec![0]]];
    let r = p.clone();
    let cfg = Config {
        max_iterations: 8,
        ..Config::default()
    };
    let out = tbls::solve(&p, &r, 17, &cfg).unwrap();
    assert!(oracle::stable_matchings(&p, &r, Stability::Weak).contains(&out.matching));
    assert!(is_stable(
        &out.strategy.proposer,
        &out.strategy.receiver,
        &out.matching
    ));
    assert_eq!(out.stats.validation_fallbacks, 0);
}
