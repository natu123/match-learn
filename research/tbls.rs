//! Experimental paper-based TBLS (arXiv:2409.10575v2, Algorithms 1-4).
//! Original Rust implementation, not author code or a performance reproduction.
//! Unit capacities, equal side sizes, valid mutually acceptable lists only.
use match_learn::matching::{Matching, gale_shapley, is_stable};
use match_learn::rng::Rng;
use match_learn::ties::is_weakly_stable;
use std::time::{Duration, Instant};

// This separately tested adapter is not invoked by the static-only harness.
#[allow(dead_code)]
#[path = "tbls_warm.rs"]
pub mod warm;

pub type Profile = Vec<Vec<Vec<usize>>>;
type Ranks = Vec<Vec<Option<usize>>>;

#[derive(Clone, Copy, Debug)]
pub enum RepairBudget {
    /// Actual repair elapsed time; default is initial GS elapsed time.
    Time(Duration),
    /// Deterministic correctness-test mode, not paper timing or equal CPU.
    CandidateChecks(usize),
}

#[derive(Clone, Debug)]
pub struct Config {
    pub max_iterations: usize,
    pub disruption_probability: f64,
    pub estimate_ratio: f64,
    pub disrupted_rows_per_side: usize,
    pub repair_budget: Option<RepairBudget>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_iterations: 3000,
            disruption_probability: 0.05,
            estimate_ratio: 0.9,
            disrupted_rows_per_side: 1,
            repair_budget: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strategy {
    pub proposer: Vec<Vec<usize>>,
    pub receiver: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub iterations: usize,
    pub adjustments: usize,
    pub disruptions: usize,
    pub candidate_checks: usize,
    pub removed_blockers: usize,
    pub budget_fallbacks: usize,
    /// Fail-closed guard; any nonzero count is a reproduction defect to inspect.
    pub validation_fallbacks: usize,
    pub cardinality_regressions: usize,
}

#[derive(Clone, Debug)]
pub struct Output {
    pub matching: Matching,
    pub strategy: Strategy,
    pub initial: Matching,
    pub big_m: f64,
    pub score: f64,
    pub stats: Stats,
    pub calibrated_repair_time: Duration,
}

fn ranks(profile: &Profile, n: usize) -> Vec<Vec<Option<usize>>> {
    let mut out = vec![vec![None; n]; profile.len()];
    for (i, row) in profile.iter().enumerate() {
        for (rank, tier) in row.iter().enumerate() {
            for &j in tier {
                out[i][j] = Some(rank);
            }
        }
    }
    out
}

fn validate(p: &Profile, r: &Profile, cfg: &Config) -> Result<(), &'static str> {
    let n = p.len();
    if n != r.len() || n > 64 {
        return Err("requires equal sides of at most 64 agents");
    }
    if cfg.max_iterations > 3000
        || cfg.disrupted_rows_per_side > n.max(1)
        || !cfg.disruption_probability.is_finite()
        || !(0.0..=1.0).contains(&cfg.disruption_probability)
        || !cfg.estimate_ratio.is_finite()
        || !(0.0..1.0).contains(&cfg.estimate_ratio)
    {
        return Err("invalid or out-of-bound research configuration");
    }
    for profile in [p, r] {
        for row in profile {
            let mut seen = vec![false; n];
            for tier in row {
                if tier.is_empty() {
                    return Err("empty tiers are not valid");
                }
                for &j in tier {
                    if j >= n || seen[j] {
                        return Err("duplicate or out-of-range preference");
                    }
                    seen[j] = true;
                }
            }
        }
    }
    let pr = ranks(p, n);
    let rr = ranks(r, n);
    for (i, row) in pr.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            if entry.is_some() != rr[j][i].is_some() {
                return Err("paper graph requires reciprocal acceptable edges");
            }
        }
    }
    Ok(())
}

fn shuffled_row(row: &[Vec<usize>], rng: &mut Rng) -> Vec<usize> {
    row.iter()
        .flat_map(|tier| {
            let mut t = tier.clone();
            rng.shuffle(&mut t);
            t
        })
        .collect()
}

fn random_strategy(p: &Profile, r: &Profile, rng: &mut Rng) -> Strategy {
    Strategy {
        proposer: p.iter().map(|row| shuffled_row(row, rng)).collect(),
        receiver: r.iter().map(|row| shuffled_row(row, rng)).collect(),
    }
}

fn push_unique(queue: &mut Vec<usize>, value: usize) {
    if !queue.contains(&value) {
        queue.push(value);
    }
}

// Agent IDs 0..n are proposers, n..2n receivers. Algorithm 3 samples
// one eligible adjustment per free agent before Algorithm 2 chooses one.
fn adjustments(
    pr: &[Vec<Option<usize>>],
    rr: &[Vec<Option<usize>>],
    m: &Matching,
    rng: &mut Rng,
) -> Vec<(usize, usize)> {
    let n = pr.len();
    let mut out = Vec::new();
    for (i, row) in pr.iter().enumerate() {
        if m.proposer[i].is_some() {
            continue;
        }
        let options: Vec<_> = row
            .iter()
            .enumerate()
            .filter_map(|(j, rank)| {
                m.receiver[j]
                    .filter(|&cur| rank.is_some() && rr[j][i] == rr[j][cur])
                    .map(|_| (i, n + j))
            })
            .collect();
        if !options.is_empty() {
            out.push(options[rng.below(options.len())]);
        }
    }
    for (j, row) in rr.iter().enumerate() {
        if m.receiver[j].is_some() {
            continue;
        }
        let options: Vec<_> = row
            .iter()
            .enumerate()
            .filter_map(|(i, rank)| {
                m.proposer[i]
                    .filter(|&cur| rank.is_some() && pr[i][j] == pr[i][cur])
                    .map(|_| (n + j, i))
            })
            .collect();
        if !options.is_empty() {
            out.push(options[rng.below(options.len())]);
        }
    }
    out
}

fn refine(
    profiles: (&Profile, &Profile),
    original_ranks: (&Ranks, &Ranks),
    m: &Matching,
    s: &mut Strategy,
    cfg: &Config,
    rng: &mut Rng,
    stats: &mut Stats,
) -> Vec<usize> {
    let (p, r) = profiles;
    let (pr, rr) = original_ranks;
    let n = p.len();
    let options = adjustments(pr, rr, m, rng);
    let mut changed = Vec::new();
    if options.is_empty() || rng.uniform() < cfg.disruption_probability {
        stats.disruptions += 1;
        for (profile, strict, offset) in [(p, &mut s.proposer, 0), (r, &mut s.receiver, n)] {
            let mut ids: Vec<_> = (0..n).collect();
            rng.shuffle(&mut ids);
            for &i in ids.iter().take(cfg.disrupted_rows_per_side) {
                strict[i] = shuffled_row(&profile[i], rng);
                push_unique(&mut changed, offset + i);
            }
        }
    } else {
        stats.adjustments += 1;
        let (free, target) = options[rng.below(options.len())];
        let (row, original, candidate) = if target < n {
            (&mut s.proposer[target], &pr[target], free - n)
        } else {
            (&mut s.receiver[target - n], &rr[target - n], free)
        };
        let rank = original[candidate].unwrap();
        let first = row.iter().position(|&x| original[x] == Some(rank)).unwrap();
        let old = row.iter().position(|&x| x == candidate).unwrap();
        row.remove(old);
        row.insert(first, candidate);
        changed.push(target);
    }
    changed
}

fn strict_ranks(rows: &[Vec<usize>], n: usize) -> Vec<Vec<Option<usize>>> {
    let mut ranks = vec![vec![None; n]; n];
    for (i, row) in rows.iter().enumerate() {
        for (k, &j) in row.iter().enumerate() {
            ranks[i][j] = Some(k);
        }
    }
    ranks
}

fn remove_blocker(m: &mut Matching, p: usize, r: usize, queue: &mut Vec<usize>) {
    let n = m.proposer.len();
    if let Some(old_r) = m.proposer[p] {
        m.receiver[old_r] = None;
        push_unique(queue, n + old_r);
    }
    if let Some(old_p) = m.receiver[r] {
        m.proposer[old_p] = None;
        push_unique(queue, old_p);
    }
    m.proposer[p] = Some(r);
    m.receiver[r] = Some(p);
}

// Algorithm 4, with a global strict check to detect a queue implementation bug.
fn repair(
    s: &Strategy,
    m: &mut Matching,
    mut queue: Vec<usize>,
    budget: RepairBudget,
    rng: &mut Rng,
    stats: &mut Stats,
) {
    let started = Instant::now();
    let n = m.proposer.len();
    let pr = strict_ranks(&s.proposer, n);
    let rr = strict_ranks(&s.receiver, n);
    let mut checks = 0;
    let mut exceeded = false;
    'queue: while !queue.is_empty() {
        if let RepairBudget::Time(limit) = budget
            && started.elapsed() > limit
        {
            exceeded = true;
            break;
        }
        let selected = rng.below(queue.len());
        let agent = queue.swap_remove(selected);
        let row = if agent < n {
            &s.proposer[agent]
        } else {
            &s.receiver[agent - n]
        };
        for &other in row {
            let expired = match budget {
                RepairBudget::Time(limit) => started.elapsed() > limit,
                RepairBudget::CandidateChecks(limit) => checks >= limit,
            };
            if expired {
                exceeded = true;
                break 'queue;
            }
            checks += 1;
            stats.candidate_checks += 1;
            let (p, r) = if agent < n {
                (agent, other)
            } else {
                (other, agent - n)
            };
            if m.proposer[p] == Some(r) {
                continue;
            }
            let p_improves = m.proposer[p].is_none_or(|cur| pr[p][r] < pr[p][cur]);
            let r_improves = m.receiver[r].is_none_or(|cur| rr[r][p] < rr[r][cur]);
            let agent_improves = if agent < n { p_improves } else { r_improves };
            if !agent_improves {
                break;
            }
            if p_improves && r_improves {
                remove_blocker(m, p, r, &mut queue);
                stats.removed_blockers += 1;
            }
        }
    }
    if exceeded {
        stats.budget_fallbacks += 1;
        *m = gale_shapley(&s.proposer, &s.receiver);
    } else if !is_stable(&s.proposer, &s.receiver, m) {
        stats.validation_fallbacks += 1;
        *m = gale_shapley(&s.proposer, &s.receiver);
    }
}

fn score(p: &Profile, r: &Profile, m: &Matching, big_m: f64) -> f64 {
    let free_lengths: usize = [(p, &m.proposer), (r, &m.receiver)]
        .into_iter()
        .map(|(profile, partners)| {
            profile
                .iter()
                .zip(partners)
                .filter(|(_, partner)| partner.is_none())
                .map(|(row, _)| row.iter().map(Vec::len).sum::<usize>())
                .sum::<usize>()
        })
        .sum();
    m.pairs() as f64 * big_m + free_lengths as f64
}

pub fn solve(p: &Profile, r: &Profile, seed: u64, cfg: &Config) -> Result<Output, &'static str> {
    validate(p, r, cfg)?;
    let mut rng = Rng::new(seed);
    let s = random_strategy(p, r, &mut rng);
    let started = Instant::now();
    let current = gale_shapley(&s.proposer, &s.receiver);
    let calibrated_repair_time = started.elapsed();
    search(p, r, cfg, current, s, rng, calibrated_repair_time)
}

fn search(
    p: &Profile,
    r: &Profile,
    cfg: &Config,
    mut current: Matching,
    mut s: Strategy,
    mut rng: Rng,
    calibrated_repair_time: Duration,
) -> Result<Output, &'static str> {
    let n = p.len();
    let pr = ranks(p, n);
    let rr = ranks(r, n);
    let budget = cfg
        .repair_budget
        .unwrap_or(RepairBudget::Time(calibrated_repair_time));
    let max_len = |profile: &Profile| {
        profile
            .iter()
            .map(|row| row.iter().map(Vec::len).sum::<usize>())
            .max()
            .unwrap_or(0)
    };
    let big_m =
        (max_len(p) + max_len(r)) as f64 * (n as f64 - cfg.estimate_ratio * current.pairs() as f64);
    let initial = current.clone();
    let mut best = current.clone();
    let mut best_s = s.clone();
    let mut best_score = score(p, r, &best, big_m);
    let mut stats = Stats::default();
    // Follow the prose perfect-stop rule, and treat max_iterations as an exact
    // cap (the pseudocode's inclusive counter is ambiguous by one iteration).
    while stats.iterations < cfg.max_iterations && current.pairs() < n {
        let changed = refine(
            (p, r),
            (&pr, &rr),
            &current,
            &mut s,
            cfg,
            &mut rng,
            &mut stats,
        );
        repair(&s, &mut current, changed, budget, &mut rng, &mut stats);
        if !is_weakly_stable(p, r, &current) {
            return Err("weak stability guard failed");
        }
        stats.iterations += 1;
        let value = score(p, r, &current, big_m);
        if value >= best_score {
            if current.pairs() < best.pairs() {
                stats.cardinality_regressions += 1;
            }
            best = current.clone();
            best_s = s.clone();
            best_score = value;
        }
    }
    Ok(Output {
        matching: best,
        strategy: best_s,
        initial,
        big_m,
        score: best_score,
        stats,
        calibrated_repair_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work_config() -> Config {
        Config {
            max_iterations: 16,
            repair_budget: Some(RepairBudget::CandidateChecks(256)),
            ..Config::default()
        }
    }

    #[test]
    fn receiver_free_adjustment_promotes_only_within_proposer_tie() {
        let p = vec![vec![vec![0, 1]], vec![]];
        let r = vec![vec![vec![0]], vec![vec![0]]];
        let m = Matching {
            proposer: vec![Some(0), None],
            receiver: vec![Some(0), None],
        };
        let pr = ranks(&p, 2);
        let rr = ranks(&r, 2);
        let mut rng = Rng::new(5);
        assert_eq!(adjustments(&pr, &rr, &m, &mut rng), vec![(3, 0)]);
        let mut s = Strategy {
            proposer: vec![vec![0, 1], vec![]],
            receiver: vec![vec![0], vec![0]],
        };
        let mut stats = Stats::default();
        let cfg = Config {
            disruption_probability: 0.0,
            ..work_config()
        };
        assert_eq!(
            refine((&p, &r), (&pr, &rr), &m, &mut s, &cfg, &mut rng, &mut stats),
            vec![0]
        );
        assert_eq!(s.proposer[0], vec![1, 0]);
        assert_eq!(stats.adjustments, 1);
    }

    #[test]
    fn repair_enqueues_both_displaced_endpoints() {
        let s = Strategy {
            proposer: vec![vec![1, 0], vec![0, 1]],
            receiver: vec![vec![1, 0], vec![0, 1]],
        };
        let mut m = Matching {
            proposer: vec![Some(0), Some(1)],
            receiver: vec![Some(0), Some(1)],
        };
        let mut stats = Stats::default();
        repair(
            &s,
            &mut m,
            vec![0],
            RepairBudget::CandidateChecks(32),
            &mut Rng::new(7),
            &mut stats,
        );
        assert_eq!(m.proposer, vec![Some(1), Some(0)]);
        assert!(is_stable(&s.proposer, &s.receiver, &m));
        assert_eq!(stats.removed_blockers, 2);
        assert_eq!(stats.validation_fallbacks, 0);
        assert_eq!(stats.budget_fallbacks, 0);
    }

    #[test]
    fn zero_work_budget_uses_gs_fallback() {
        let s = Strategy {
            proposer: vec![vec![1, 0], vec![0, 1]],
            receiver: vec![vec![1, 0], vec![0, 1]],
        };
        let mut m = Matching {
            proposer: vec![Some(0), Some(1)],
            receiver: vec![Some(0), Some(1)],
        };
        let mut stats = Stats::default();
        repair(
            &s,
            &mut m,
            vec![0],
            RepairBudget::CandidateChecks(0),
            &mut Rng::new(7),
            &mut stats,
        );
        assert_eq!(m, gale_shapley(&s.proposer, &s.receiver));
        assert_eq!(stats.budget_fallbacks, 1);
        assert_eq!(stats.removed_blockers, 0);
    }

    #[test]
    fn invalid_incremental_queue_is_detected_instead_of_emitting_unstable_result() {
        let s = Strategy {
            proposer: vec![vec![1, 0], vec![0, 1]],
            receiver: vec![vec![1, 0], vec![0, 1]],
        };
        let mut m = Matching {
            proposer: vec![Some(0), Some(1)],
            receiver: vec![Some(0), Some(1)],
        };
        let mut stats = Stats::default();
        repair(
            &s,
            &mut m,
            vec![],
            RepairBudget::CandidateChecks(32),
            &mut Rng::new(7),
            &mut stats,
        );
        assert!(is_stable(&s.proposer, &s.receiver, &m));
        assert_eq!(stats.validation_fallbacks, 1);
    }

    #[test]
    fn no_adjustments_disrupts_and_perfect_initial_state_stops() {
        let p = vec![vec![vec![0]], vec![]];
        let r = p.clone();
        let out = solve(&p, &r, 4, &work_config()).unwrap();
        assert_eq!(out.stats.disruptions, 16);
        assert_eq!(out.stats.adjustments, 0);
        let tied = vec![vec![vec![0, 1]]; 2];
        let out = solve(&tied, &tied, 4, &work_config()).unwrap();
        assert_eq!(out.stats.iterations, 0);
        assert_eq!(out.matching, out.initial);
    }

    #[test]
    fn validation_rejects_invalid_profiles_and_handles_empty_market() {
        let cfg = work_config();
        assert!(
            solve(&vec![], &vec![], 4, &cfg)
                .unwrap()
                .matching
                .proposer
                .is_empty()
        );
        for (p, r) in [
            (vec![vec![vec![0, 0]]], vec![vec![vec![0]]]),
            (vec![vec![vec![1]]], vec![vec![vec![0]]]),
            (vec![vec![vec![]]], vec![vec![]]),
            (vec![vec![vec![0]]], vec![vec![]]),
            (vec![vec![]], vec![]),
        ] {
            assert!(solve(&p, &r, 4, &cfg).is_err());
        }
    }

    #[test]
    fn paper_score_counts_free_list_lengths_on_both_sides() {
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
        let m = Matching {
            proposer: vec![Some(0), Some(1), None, None],
            receiver: vec![Some(0), Some(1), None, None],
        };
        // maxlen_U=maxlen_W=3, N=4, M_init=2, c=.9:
        // bigM=6*(4-1.8)=13.2. Four free agents each have list length one.
        assert!((score(&p, &r, &m, 13.2) - 30.4).abs() < 1e-12);
    }
}
