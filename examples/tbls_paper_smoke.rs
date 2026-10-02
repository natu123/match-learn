//! Static paper-based TBLS control, not author-code/performance reproduction.
//! cargo run --example tbls_paper_smoke -- 17 work
//! Omit 'work' to use measured initial-GS repair time (timing is not deterministic).
#[path = "../research/tbls.rs"]
mod tbls;
use match_learn::matching::is_stable;
use match_learn::ties::is_weakly_stable;
use serde_json::json;
use tbls::{Config, RepairBudget};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let seed = args
        .get(1)
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(17);
    let mode = args.get(2).map(String::as_str).unwrap_or("time");
    let budget = match mode {
        "time" => None,
        "work" => Some(RepairBudget::CandidateChecks(256)),
        _ => return Err("mode must be time or work".into()),
    };
    let cfg = Config {
        max_iterations: 16,
        repair_budget: budget,
        ..Config::default()
    };
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
    let out = tbls::solve(&p, &r, seed, &cfg)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "implementation": "original Rust, paper-based static TBLS v2; not author code",
            "scope": "Table 1 correctness control; not SOTA or equal-CPU comparison",
            "seed": seed, "mode": mode, "maximum_iterations": cfg.max_iterations,
            "proposer_tiers": p, "receiver_tiers": r,
            "initial_proposer": out.initial.proposer, "initial_cardinality": out.initial.pairs(),
            "proposer": out.matching.proposer, "receiver": out.matching.receiver,
            "cardinality": out.matching.pairs(), "paper_score": out.score, "big_m": out.big_m,
            "best_strategy_proposer": out.strategy.proposer,
            "best_strategy_receiver": out.strategy.receiver,
            "strict_stable_under_best_strategy": is_stable(&out.strategy.proposer, &out.strategy.receiver, &out.matching),
            "original_weak_stable": is_weakly_stable(&p, &r, &out.matching),
            "iterations": out.stats.iterations, "adjustments": out.stats.adjustments,
            "disruptions": out.stats.disruptions, "candidate_checks": out.stats.candidate_checks,
            "removed_blockers": out.stats.removed_blockers,
            "budget_fallbacks": out.stats.budget_fallbacks,
            "validation_fallbacks": out.stats.validation_fallbacks,
            "cardinality_regressions": out.stats.cardinality_regressions,
            "diagnostic_calibrated_repair_ns": out.calibrated_repair_time.as_nanos()
        }))?
    );
    Ok(())
}
