//! Bounded exact smoke pilot, not a solver or performance benchmark.
//! Run: cargo run --example retention_smoke -- ../retention-smoke
#[path = "../tests/support/tiny_matching_oracle.rs"]
mod oracle;

use match_learn::matching::{Matching, gale_shapley};
use match_learn::rng::Rng;
use match_learn::ties::is_weakly_stable;
use oracle::{Profile, Stability};
use serde_json::{Value, json};
use std::{env, fs, path::PathBuf, time::Instant};

struct Case {
    name: String,
    seed: Option<u64>,
    event: String,
    before_p: Profile,
    before_r: Profile,
    previous: Matching,
    after_p: Profile,
    after_r: Profile,
}

fn matching(proposer: Vec<Option<usize>>, nr: usize) -> Matching {
    let mut receiver = vec![None; nr];
    for (p, &partner) in proposer.iter().enumerate() {
        if let Some(r) = partner {
            assert!(r < nr && receiver[r].is_none());
            receiver[r] = Some(p);
        }
    }
    Matching { proposer, receiver }
}

fn fixed_tie_gs(p: &Profile, r: &Profile) -> Matching {
    fn strict(profile: &Profile) -> Vec<Vec<usize>> {
        profile
            .iter()
            .map(|tiers| {
                tiers
                    .iter()
                    .flat_map(|tier| {
                        let mut ordered = tier.clone();
                        ordered.sort_unstable();
                        ordered
                    })
                    .collect()
            })
            .collect()
    }
    gale_shapley(&strict(p), &strict(r))
}

fn retained(m: &Matching, previous: &Matching) -> usize {
    m.proposer
        .iter()
        .enumerate()
        .filter(|&(p, partner)| partner.is_some() && previous.proposer.get(p) == Some(partner))
        .count()
}

fn matching_json(m: &Matching) -> Value {
    json!({"proposer": m.proposer, "receiver": m.receiver})
}

fn cycle(n: usize) -> Case {
    assert!((2..=3).contains(&n));
    let p: Profile = (0..n).map(|i| vec![vec![i], vec![(i + 1) % n]]).collect();
    let r: Profile = (0..n)
        .map(|i| vec![vec![(i + n - 1) % n], vec![i]])
        .collect();
    let previous = matching((0..n).map(Some).collect(), n);
    let mut after_p = p.clone();
    after_p[1].swap(0, 1);
    let shifted = matching((0..n).map(|i| Some((i + 1) % n)).collect(), n);
    // Explicitly verify the control's claimed forced shift, including strict
    // stability equivalence. Do not assume a unique stable matching.
    for kind in [Stability::Weak, Stability::Strong, Stability::Super] {
        assert_eq!(
            oracle::stable_matchings(&after_p, &r, kind),
            vec![shifted.clone()]
        );
    }
    Case {
        name: format!("cycle_degree2_n{n}"),
        seed: None,
        event: "swap proposer 1's two strict tiers; all other rows unchanged".into(),
        before_p: p,
        before_r: r.clone(),
        previous,
        after_p,
        after_r: r,
    }
}

fn arrival() -> Case {
    Case {
        name: "arrival_cardinality_control".into(),
        seed: None,
        event:
            "add proposer 1 accepting only receiver 0; receiver 0 ties the entrant with proposer 0"
                .into(),
        before_p: vec![vec![vec![0, 1]]],
        before_r: vec![vec![vec![0]], vec![vec![0]]],
        previous: matching(vec![Some(0)], 2),
        after_p: vec![vec![vec![0, 1]], vec![vec![0]]],
        after_r: vec![vec![vec![0, 1]], vec![vec![0]]],
    }
}

fn tie_churn() -> Case {
    Case {
        name: "avoidable_tie_churn_control".into(),
        seed: None,
        event: "swap receiver 0's two strict tiers; proposer ties unchanged".into(),
        before_p: vec![vec![vec![0, 1]]; 2],
        before_r: vec![vec![vec![1], vec![0]], vec![vec![0], vec![1]]],
        previous: matching(vec![Some(1), Some(0)], 2),
        after_p: vec![vec![vec![0, 1]]; 2],
        after_r: vec![vec![vec![0], vec![1]], vec![vec![0], vec![1]]],
    }
}

fn seeded_edit(seed: u64) -> Case {
    let rows = [
        vec![],
        vec![vec![0]],
        vec![vec![1]],
        vec![vec![0], vec![1]],
        vec![vec![1], vec![0]],
        vec![vec![0, 1]],
    ];
    let mut rng = Rng::new(seed);
    let p = vec![rows[3 + rng.below(2)].clone(), rows[rng.below(6)].clone()];
    let r = vec![rows[rng.below(6)].clone(), rows[rng.below(6)].clone()];
    let old_stable = oracle::stable_matchings(&p, &r, Stability::Weak);
    let previous = old_stable[rng.below(old_stable.len())].clone();
    let mut after_p = p.clone();
    after_p[0].swap(0, 1);
    Case {
        name: format!("seeded_adjacent_swap_{seed}"),
        seed: Some(seed),
        event: "swap proposer 0's two strict tiers; all other rows unchanged".into(),
        before_p: p,
        before_r: r.clone(),
        previous,
        after_p,
        after_r: r,
    }
}

fn evaluate(case: &Case) -> Value {
    assert!(case.before_p.len() <= 3 && case.before_r.len() <= 3);
    assert!(case.after_p.len() <= 3 && case.after_r.len() <= 3);
    let old_stable = oracle::stable_matchings(&case.before_p, &case.before_r, Stability::Weak);
    assert!(
        old_stable.contains(&case.previous),
        "invalid previous matching: {}",
        case.name
    );
    assert!(is_weakly_stable(
        &case.before_p,
        &case.before_r,
        &case.previous
    ));
    let eligible_old_pairs = case
        .previous
        .proposer
        .iter()
        .enumerate()
        .filter(|&(p, partner)| {
            partner.is_some_and(|r| {
                case.after_p
                    .get(p)
                    .is_some_and(|tiers| tiers.iter().any(|t| t.contains(&r)))
                    && case
                        .after_r
                        .get(r)
                        .is_some_and(|tiers| tiers.iter().any(|t| t.contains(&p)))
            })
        })
        .count();
    let feasible = oracle::feasible_matchings(&case.after_p, &case.after_r);
    let started = Instant::now();
    let stable = oracle::stable_matchings(&case.after_p, &case.after_r, Stability::Weak);
    let oracle_ns = u64::try_from(started.elapsed().as_nanos()).unwrap();
    assert!(!stable.is_empty());
    let optimum = stable
        .iter()
        .map(|m| (m.pairs(), retained(m, &case.previous)))
        .max()
        .unwrap();
    assert_eq!(
        oracle::cardinality_retention(&case.after_p, &case.after_r, &case.previous),
        Some(optimum)
    );
    let optimal = stable
        .iter()
        .find(|m| (m.pairs(), retained(m, &case.previous)) == optimum)
        .unwrap();
    let best_retention_any_size = stable
        .iter()
        .map(|m| retained(m, &case.previous))
        .max()
        .unwrap();
    let recomputed = fixed_tie_gs(&case.after_p, &case.after_r);
    let mut projected = case.previous.clone();
    // IDs are stable; arrival appends an unmatched slot, never remaps old IDs.
    projected.proposer.resize(case.after_p.len(), None);
    projected.receiver.resize(case.after_r.len(), None);
    let keep_valid = stable.contains(&projected);
    let keep_or_recompute = if keep_valid { &projected } else { &recomputed };
    if case.name.starts_with("cycle_degree2") {
        assert_eq!(optimum, (case.after_p.len(), 0));
        assert!(!keep_valid);
    } else if case.name == "arrival_cardinality_control" {
        assert!(keep_valid);
        assert_eq!(optimum, (2, 0));
        assert_eq!(
            (recomputed.pairs(), retained(&recomputed, &case.previous)),
            (1, 1)
        );
    } else if case.name == "avoidable_tie_churn_control" {
        assert!(keep_valid);
        assert_eq!(optimum, (2, 2));
        assert_eq!(
            (recomputed.pairs(), retained(&recomputed, &case.previous)),
            (2, 0)
        );
    }
    let baselines: Vec<_> = [
        ("exact_lexicographic_oracle", optimal),
        ("fixed_tie_gs_recompute", &recomputed),
        ("keep_if_stable_else_fixed_tie_gs", keep_or_recompute),
    ].into_iter().map(|(name, m)| {
        let feasible_result = feasible.contains(m);
        let stable_result = stable.contains(m);
        assert!(feasible_result && stable_result, "{} {name}", case.name);
        assert!(is_weakly_stable(&case.after_p, &case.after_r, m));
        let retention = retained(m, &case.previous);
        json!({"baseline": name, "matching": matching_json(m),
            "feasible": feasible_result, "weakly_stable": stable_result,
            "cardinality": m.pairs(), "retained_feasible_old_pairs": retention,
            "total_old_pairs_lost": case.previous.pairs() - retention,
            "cardinality_gap": optimum.0 - m.pairs(),
            "retention_gap_at_max_cardinality": if m.pairs() == optimum.0 { Some(optimum.1 - retention) } else { None }})
    }).collect();
    json!({"case": case.name, "seed": case.seed, "event": case.event,
        "before": {"proposer_tiers": case.before_p, "receiver_tiers": case.before_r},
        "after": {"proposer_tiers": case.after_p, "receiver_tiers": case.after_r},
        "previous": matching_json(&case.previous),
        "previous_feasible_and_weakly_stable_before": true,
        "projected_previous": matching_json(&projected),
        "previous_still_feasible_after": feasible.contains(&projected),
        "previous_still_weakly_stable_after": keep_valid,
        "previous_pairs": case.previous.pairs(), "eligible_old_pairs": eligible_old_pairs,
        "old_pairs_lost_to_infeasibility": case.previous.pairs() - eligible_old_pairs,
        "new_feasible_matching_count": feasible.len(), "new_weak_stable_matching_count": stable.len(),
        "maximum_cardinality": optimum.0, "retention_at_maximum_cardinality": optimum.1,
        "maximum_retention_at_any_cardinality": best_retention_any_size,
        "unavoidable_eligible_pair_loss_for_weak_stability": eligible_old_pairs - best_retention_any_size,
        "additional_retention_loss_from_cardinality_priority": best_retention_any_size - optimum.1,
        "oracle_diagnostic_elapsed_ns": oracle_ns, "baselines": baselines})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = PathBuf::from(env::args().nth(1).ok_or("provide output directory")?);
    let seeds: Vec<u64> = (1..=16).collect();
    let mut cases = vec![cycle(2), cycle(3), arrival(), tie_churn()];
    cases.extend(seeds.iter().copied().map(seeded_edit));
    let results: Vec<_> = cases.iter().map(evaluate).collect();
    let data = json!({"schema_version": 1, "upstream_base": "9ab03c2cbedef9370b1d754cbe140364caf93303",
        "scope": "20 bounded smoke cases, not SOTA, not an estimate of global headroom",
        "objective": "maximize weak-stable cardinality, then retained feasible old pairs",
        "fixed_tie_rule": "ascending numeric ID within each tier; proposer-side Gale-Shapley",
        "seeds": seeds, "rng": "repository splitmix64 Rng; one independent stream per seed",
        "sampling": "2x2; proposer 0 one of two complete strict rows; other rows one of all six partial tiered orders; previous sampled from exact old weak-stable set; swap proposer 0 tiers",
        "bounds": {"max_proposers": 3, "max_receivers": 3, "case_count": results.len()},
        "runtime_note": "single debug-build oracle enumeration elapsed time; excludes compilation; diagnostic only, not a performance comparison",
        "results": results});
    let mut report = String::from(
        "Weak-stable retention smoke pilot\n\nObjective: maximum cardinality first, retained old pairs second. Smoke baselines only; no SOTA or global-headroom claim.\nIDs are zero-based and stable across events. Seeds: 1..=16. Controls deterministic.\nAll previous matchings verified weakly stable; every returned baseline checked for feasibility and weak stability against the independent oracle and production validator.\n\ncase | exact(size, retained) | fixed-GS(size, retained) | keep-if-stable(size, retained) | old eligible | stability-forced loss | cardinality-priority loss\n",
    );
    for result in data["results"].as_array().unwrap() {
        let rows = result["baselines"].as_array().unwrap();
        report.push_str(&format!(
            "{} | ({},{}) | ({},{}) | ({},{}) | {} | {} | {}\n",
            result["case"].as_str().unwrap(),
            rows[0]["cardinality"],
            rows[0]["retained_feasible_old_pairs"],
            rows[1]["cardinality"],
            rows[1]["retained_feasible_old_pairs"],
            rows[2]["cardinality"],
            rows[2]["retained_feasible_old_pairs"],
            result["eligible_old_pairs"],
            result["unavoidable_eligible_pair_loss_for_weak_stability"],
            result["additional_retention_loss_from_cardinality_priority"]
        ));
    }
    report.push_str("\nControls verified by assertions:\nCycle n=2/n=3: one proposer tier swap forces the unique shifted assignment. Every old pair stays acceptable, but all old pairs must be lost for stability.\nArrival: unchanged assignment stays weakly stable at size 1; exact size 2 loses its one old pair. Both smoke baselines miss cardinality.\nTie churn: fixed-tie GS loses two pairs unnecessarily at the same cardinality; keeping the old weak-stable matching achieves exact (2,2).\n\nJSON contains all original/new tiers, both matching views, seeds, feasibility/stability, baseline gaps, and separate infeasibility/stability/cardinality-priority losses. Retention gaps are null for cardinality-deficient baselines, since objectives are lexicographic. Diagnostic timings are not performance evidence.\nThe oracle is exponential and assumes valid profiles without duplicate entries. Enumeration here is capped at 3x3. All current events preserve old-pair acceptability, so nonzero infeasibility loss is not exercised. The 16 seeded cases are a selected design, not a representative event distribution. Results do not establish general Irving correctness or global headroom.\nNext prerequisites: define the event distribution and ID rules; compare exact objective values on identical inputs; add established baselines only after reviewing primary definitions and implementation provenance; set solver budgets and tie rules; report unavoidable losses separately. No TBLS or new solver implemented.\n\nReproduce: cargo run --example retention_smoke -- ../retention-smoke\nMachine output: results.json. Remove oracle_diagnostic_elapsed_ns when comparing repeated results.\n");
    fs::create_dir_all(&output)?;
    fs::write(
        output.join("results.json"),
        serde_json::to_string_pretty(&data)? + "\n",
    )?;
    fs::write(output.join("REPORT.txt"), &report)?;
    print!("{report}");
    Ok(())
}
