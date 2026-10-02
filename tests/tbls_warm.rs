//! Original warm adapter correctness checks, not author TBLS or recourse search.
#[path = "support/tiny_matching_oracle.rs"]
mod oracle;
#[path = "../research/tbls.rs"]
mod tbls;
use match_learn::matching::{Matching, gale_shapley, is_stable};
use match_learn::rng::Rng;
use oracle::{Profile, Stability};
use tbls::{
    Config, RepairBudget, Strategy,
    warm::{Previous, WarmOutput},
};

fn strict(p: &Profile) -> Vec<Vec<usize>> {
    p.iter()
        .map(|row| row.iter().flatten().copied().collect())
        .collect()
}

fn cfg() -> Config {
    Config {
        max_iterations: 8,
        repair_budget: Some(RepairBudget::CandidateChecks(256)),
        ..Config::default()
    }
}

fn previous<'a>(p: &'a Profile, r: &'a Profile, m: &'a Matching, s: &'a Strategy) -> Previous<'a> {
    Previous {
        proposer_profile: p,
        receiver_profile: r,
        matching: m,
        strategy: s,
    }
}

fn check(old_p: &Profile, old_r: &Profile, p: &Profile, r: &Profile, seed: u64) -> WarmOutput {
    let s = Strategy {
        proposer: strict(old_p),
        receiver: strict(old_r),
    };
    let m = gale_shapley(&s.proposer, &s.receiver);
    let out = tbls::warm::solve(p, r, previous(old_p, old_r, &m, &s), seed, &cfg()).unwrap();
    let stable = oracle::stable_matchings(p, r, Stability::Weak);
    assert!(stable.contains(&out.result.matching));
    assert!(is_stable(
        &out.result.strategy.proposer,
        &out.result.strategy.receiver,
        &out.result.matching
    ));
    assert!(oracle::feasible_matchings(p, r).contains(&out.projected));
    assert!(stable.contains(&out.result.initial));
    assert_eq!(
        out.discarded_previous_pairs,
        m.pairs() - out.projected.pairs()
    );
    assert_eq!(out.result.stats.validation_fallbacks, 0);
    assert_eq!(out.initial_repair.validation_fallbacks, 0);
    // Dummy padding/reciprocal pruning must preserve the ORIGINAL exact objective.
    let mut padded_p = out.proposer_profile.clone();
    let mut padded_r = out.receiver_profile.clone();
    padded_p.resize(out.padded_size, vec![]);
    padded_r.resize(out.padded_size, vec![]);
    assert_eq!(
        oracle::cardinality_retention(p, r, &m),
        oracle::cardinality_retention(&padded_p, &padded_r, &m)
    );
    for kind in [Stability::Strong, Stability::Super] {
        assert!(
            oracle::stable_matchings(p, r, kind)
                .iter()
                .all(|x| stable.contains(x))
        );
    }
    let repeat = tbls::warm::solve(p, r, previous(old_p, old_r, &m, &s), seed, &cfg()).unwrap();
    assert_eq!(out.result.matching, repeat.result.matching);
    assert_eq!(out.result.strategy, repeat.result.strategy);
    assert_eq!(out.result.stats, repeat.result.stats);
    assert_eq!(out.initial_repair, repeat.initial_repair);
    assert_eq!(out.result.score, repeat.result.score);
    // Observational time calibration is excluded from deterministic truth.
    let _calibration = out.result.calibrated_repair_time;
    out
}

#[test]
fn stable_warm_arrival_continues_search_for_cardinality() {
    let old_p = vec![vec![vec![0, 1]]];
    let old_r = vec![vec![vec![0]]; 2];
    let p = vec![vec![vec![0, 1]], vec![vec![0]]];
    let r = vec![vec![vec![0, 1]], vec![vec![0]]];
    for seed in [17, 29, 20001, 20002] {
        let out = check(&old_p, &old_r, &p, &r, seed);
        assert_eq!(out.projected.pairs(), 1);
        assert_eq!(out.result.initial.pairs(), 1);
        assert_eq!(out.result.matching.pairs(), 2);
        assert!(out.result.stats.iterations > 0);
    }
}

#[test]
fn unilateral_deletion_and_departure_rebuild_both_views_without_renumbering() {
    let old = vec![vec![vec![0, 1]]; 2];
    let deleted = check(&old, &old, &vec![vec![vec![1]], vec![vec![0, 1]]], &old, 17);
    assert_eq!(deleted.discarded_previous_pairs, 1);
    assert_eq!(deleted.result.matching.proposer, vec![Some(1), Some(0)]);
    let departed = check(
        &old,
        &old,
        &vec![vec![vec![0, 1]], vec![]],
        &vec![vec![vec![0]]; 2],
        17,
    );
    assert_eq!(departed.discarded_previous_pairs, 1);
    assert_eq!(departed.result.matching.proposer, vec![Some(0), None]);
    assert_eq!(departed.result.matching.receiver, vec![Some(0), None]);
}

#[test]
fn unequal_padding_preserves_real_pairs_but_uses_padded_score_and_iteration_stop() {
    let old = vec![vec![vec![0]]];
    let p = vec![vec![vec![0]], vec![vec![1]]];
    let r = vec![vec![vec![0]], vec![vec![1]], vec![]];
    let out = check(&old, &old, &p, &r, 17);
    assert_eq!(out.result.matching.proposer, vec![Some(0), Some(1)]);
    assert_eq!(out.result.matching.receiver, vec![Some(0), Some(1), None]);
    assert_eq!(out.padded_size, 3);
    assert_eq!(out.result.stats.iterations, 8);
    assert!((out.result.big_m - 2.4).abs() < 1e-12);
    assert_eq!(out.calibration_gs_cardinality, 2);
}

#[test]
fn malformed_old_state_and_inputs_are_rejected_before_pruning() {
    let old = vec![vec![vec![0]]];
    let s = Strategy {
        proposer: vec![vec![0]],
        receiver: vec![vec![0]],
    };
    let m = Matching {
        proposer: vec![Some(0)],
        receiver: vec![Some(0)],
    };
    for p in [
        vec![vec![vec![0, 0]]],
        vec![vec![vec![1]]],
        vec![vec![vec![]]],
        vec![],
    ] {
        assert!(tbls::warm::solve(&p, &old, previous(&old, &old, &m, &s), 17, &cfg()).is_err());
    }
    let bad = Matching {
        proposer: vec![Some(0)],
        receiver: vec![None],
    };
    assert!(tbls::warm::solve(&old, &old, previous(&old, &old, &bad, &s), 17, &cfg()).is_err());
    let bad_s = Strategy {
        proposer: vec![vec![]],
        receiver: vec![vec![0]],
    };
    assert!(tbls::warm::solve(&old, &old, previous(&old, &old, &m, &bad_s), 17, &cfg()).is_err());
}

#[test]
fn carried_tie_order_is_not_old_partner_first() {
    let p = vec![vec![vec![0, 1]], vec![]];
    let r = vec![vec![], vec![vec![0]]];
    let s = Strategy {
        proposer: vec![vec![0, 1], vec![]],
        receiver: vec![vec![], vec![0]],
    };
    let m = Matching {
        proposer: vec![Some(1), None],
        receiver: vec![None, Some(0)],
    };
    let new_r = vec![vec![vec![0]], vec![vec![0]]];
    let config = Config {
        max_iterations: 0,
        ..cfg()
    };
    let out = tbls::warm::solve(&p, &new_r, previous(&p, &r, &m, &s), 17, &config).unwrap();
    assert_eq!(out.result.strategy.proposer[0], vec![0, 1]);
    assert_eq!(out.result.matching.proposer, vec![Some(0), None]);
    assert_eq!(out.result.stats.iterations, 0);
    assert!(out.initial_repair.removed_blockers > 0);
}

#[test]
fn held_out_updates_and_second_step_states_remain_checked() {
    for seed in 3001..=3024 {
        let n = 2 + seed as usize % 2;
        let mut rng = Rng::new(seed);
        let mut old_p = vec![vec![]; n];
        let mut old_r = vec![vec![]; n];
        for (i, row) in old_p.iter_mut().enumerate() {
            for (j, receiver) in old_r.iter_mut().enumerate() {
                if rng.uniform() < 0.6 {
                    row.push(vec![j]);
                    receiver.push(vec![i]);
                }
            }
        }
        for profile in [&mut old_p, &mut old_r] {
            for row in profile {
                rng.shuffle(row);
                if row.len() >= 2 {
                    let next = row.remove(1);
                    row[0].extend(next);
                }
            }
        }
        let mut p = old_p.clone();
        let mut r = old_r.clone();
        match seed % 4 {
            0 => {
                if let Some(tier) = p[0].first_mut() {
                    tier.pop();
                }
                p[0].retain(|tier| !tier.is_empty());
            }
            1 => {
                p[0].clear();
                for row in &mut r {
                    for tier in row.iter_mut() {
                        tier.retain(|&i| i != 0);
                    }
                    row.retain(|tier| !tier.is_empty());
                }
            }
            2 => {
                if p[0].first().is_some_and(|tier| tier.len() == 2) {
                    let tier = p[0].remove(0);
                    p[0].insert(0, vec![tier[1]]);
                    p[0].insert(0, vec![tier[0]]);
                }
            }
            _ => {
                p.push(vec![vec![0]]);
                r[0].push(vec![n]);
            }
        }
        let out = check(&old_p, &old_r, &p, &r, 20001);
        let next = tbls::warm::solve(
            &p,
            &r,
            previous(
                &out.proposer_profile,
                &out.receiver_profile,
                &out.result.matching,
                &out.result.strategy,
            ),
            20002,
            &cfg(),
        )
        .unwrap();
        assert!(oracle::stable_matchings(&p, &r, Stability::Weak).contains(&next.result.matching));
        assert_eq!(next.discarded_previous_pairs, 0);
        assert_eq!(next.initial_repair.validation_fallbacks, 0);
    }
}

#[test]
fn receiver_side_updates_and_empty_market_arrival_are_supported() {
    let old_p = vec![vec![vec![0]]; 2];
    let old_r = vec![vec![vec![0, 1]]];
    let p = vec![vec![vec![0, 1]], vec![vec![0]]];
    let r = p.clone();
    assert_eq!(check(&old_p, &old_r, &p, &r, 17).result.matching.pairs(), 2);
    let tied = vec![vec![vec![0, 1]]; 2];
    let departed = check(
        &tied,
        &tied,
        &vec![vec![vec![0]]; 2],
        &vec![vec![vec![0, 1]], vec![]],
        17,
    );
    assert_eq!(departed.discarded_previous_pairs, 1);
    // Cardinality/stability are required; basic warm TBLS does not optimize
    // retained pairs and can choose the other surviving proposer.
    assert_eq!(departed.result.matching.pairs(), 1);
    assert_eq!(departed.result.matching.receiver[1], None);
    let empty = vec![];
    let p = vec![vec![vec![0]]];
    let r = vec![vec![vec![0]], vec![]];
    assert_eq!(check(&empty, &empty, &p, &r, 17).result.matching.pairs(), 1);
}

#[test]
fn zero_work_fallback_and_calibrated_time_mode_emit_checked_warm_state() {
    let old = vec![vec![vec![0, 1]]; 2];
    let s = Strategy {
        proposer: strict(&old),
        receiver: strict(&old),
    };
    let m = gale_shapley(&s.proposer, &s.receiver);
    let p = vec![vec![vec![1]], vec![vec![0, 1]]];
    for budget in [Some(RepairBudget::CandidateChecks(0)), None] {
        let config = Config {
            max_iterations: 0,
            repair_budget: budget,
            ..cfg()
        };
        let out = tbls::warm::solve(&p, &old, previous(&old, &old, &m, &s), 17, &config).unwrap();
        assert!(oracle::stable_matchings(&p, &old, Stability::Weak).contains(&out.result.matching));
        assert_eq!(out.result.matching.pairs(), 2);
        assert_eq!(out.initial_repair.validation_fallbacks, 0);
        if budget.is_some() {
            assert_eq!(out.initial_repair.budget_fallbacks, 1);
        }
    }
}
