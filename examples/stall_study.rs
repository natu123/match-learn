//! Stall study: measure how often greedy Thompson Sampling stalls, and how much
//! vanishing forced exploration cures it, across market sizes and many seeds.
//!
//! A market is said to *stall* under a learner if its per-round regret in the
//! tail never collapses to ~zero — i.e. the tail mean regret rate stays above a
//! small threshold. That is the frozen-arm pathology: the market has locked onto
//! a wrong stable matching and keeps paying regret forever.
//!
//! For each market size we draw many random markets, run each learner to horizon
//! `2T`, and report the stall rate and the tail-regret distribution (mean, p90,
//! p99, worst). This turns "2 of 40 markets stall" into a measured phenomenon.
//!
//! ```text
//! cargo run --release --example stall_study
//! ```

use match_learn::{Market, Rng, simulate};

const T: usize = 750;
const TWO_T: usize = 2 * T;
const NOISE: f64 = 0.2;
const MARKETS: usize = 400;
/// Tail mean regret rate above this counts the market as stalled (not settled).
const STALL_THRESHOLD: f64 = 0.01;
/// Forced-exploration constant for `ForcedExploreThompson`.
const FORCE_C: f64 = 0.25;

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

#[derive(Clone, Copy)]
enum Learner {
    Thompson,
    ForcedExplore,
    Ucb,
}

fn build(learner: Learner, util: Vec<Vec<f64>>, recv: Vec<Vec<usize>>, seed: u64) -> Market {
    match learner {
        Learner::Thompson => {
            Market::with_thompson(util, recv, 0.5, 1.0, NOISE * NOISE, NOISE, seed)
        }
        Learner::ForcedExplore => {
            Market::with_forced_explore(util, recv, 0.5, 1.0, NOISE * NOISE, FORCE_C, NOISE, seed)
        }
        Learner::Ucb => Market::with_ucb(util, recv, 0.4, NOISE, seed),
    }
}

/// Aggregate stall statistics for one learner over many markets.
struct Stats {
    stalled: usize,
    tail_rates: Vec<f64>,
    worst_ratio: f64,
    mean_total_regret: f64,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx]
}

fn run(learner: Learner, n: usize) -> Stats {
    let tail = TWO_T / 5;
    let mut seedgen = Rng::new(20260606);
    let mut stalled = 0;
    let mut tail_rates = Vec::with_capacity(MARKETS);
    let mut worst_ratio = 0.0_f64;
    let mut total_regret_sum = 0.0;

    for _ in 0..MARKETS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, n);
        let mut market = build(learner, util, recv, seed ^ 0xABCD);
        let rep = simulate(&mut market, TWO_T);

        let tr = rep.tail_mean_regret(tail);
        if tr > STALL_THRESHOLD {
            stalled += 1;
        }
        tail_rates.push(tr);

        let r_t = rep.cumulative_regret[T - 1].max(1e-9);
        let r_2t = rep.cumulative_regret[TWO_T - 1].max(1e-9);
        worst_ratio = worst_ratio.max(r_2t / r_t);
        total_regret_sum += rep.total_regret();
    }

    tail_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Stats {
        stalled,
        tail_rates,
        worst_ratio,
        mean_total_regret: total_regret_sum / MARKETS as f64,
    }
}

fn report(name: &str, s: &Stats) {
    println!(
        "  {name:<14} stalled {:>3}/{MARKETS} ({:>5.2}%)  tail p90={:>8.5} p99={:>8.5} worst={:>8.5}  worst_ratio={:>6.3}  mean_regret={:>8.2}",
        s.stalled,
        100.0 * s.stalled as f64 / MARKETS as f64,
        quantile(&s.tail_rates, 0.90),
        quantile(&s.tail_rates, 0.99),
        s.tail_rates.last().copied().unwrap_or(0.0),
        s.worst_ratio,
        s.mean_total_regret,
    );
}

fn main() {
    println!(
        "Stall study — {MARKETS} random markets per size, horizon 2T={TWO_T}, noise={NOISE}, stall threshold tail-rate>{STALL_THRESHOLD}\n"
    );
    for n in [3usize, 5, 8, 12] {
        println!("== {n}x{n} markets ==");
        report("Thompson", &run(Learner::Thompson, n));
        report("ForcedExplore", &run(Learner::ForcedExplore, n));
        report("UCB1", &run(Learner::Ucb, n));
        println!();
    }

    println!("== horizon sensitivity: genuine lock-in or just slow convergence? ==");
    // If the tail-regret stall fraction collapses as the horizon grows, these are
    // slow-converging markets, not frozen lock-ins. If it persists, they are true
    // stalls. Measured on the same markets, varying only total rounds.
    for n in [5usize, 8] {
        println!("-- {n}x{n} (Thompson) --");
        for horizon in [1500usize, 6000, 24000] {
            let tail = horizon / 5;
            let mut seedgen = Rng::new(20260606);
            let mut stalled = 0;
            let mut tail_rates = Vec::with_capacity(MARKETS);
            for _ in 0..MARKETS {
                let seed = (seedgen.below(1_000_000_000) as u64) + 1;
                let mut mgen = Rng::new(seed);
                let (util, recv) = random_market(&mut mgen, n);
                let mut market = Market::with_thompson(
                    util,
                    recv,
                    0.5,
                    1.0,
                    NOISE * NOISE,
                    NOISE,
                    seed ^ 0xABCD,
                );
                let rep = simulate(&mut market, horizon);
                let tr = rep.tail_mean_regret(tail);
                if tr > STALL_THRESHOLD {
                    stalled += 1;
                }
                tail_rates.push(tr);
            }
            tail_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  horizon {horizon:>6}  stalled {stalled:>3}/{MARKETS} ({:>5.2}%)  tail p99={:>8.5} worst={:>8.5}",
                100.0 * stalled as f64 / MARKETS as f64,
                quantile(&tail_rates, 0.99),
                tail_rates.last().copied().unwrap_or(0.0),
            );
        }
        println!();
    }

    println!("== decisive test: does forcing break the GENUINE (long-horizon) lock-ins? ==");
    // At a long horizon the slow-converging markets have settled, so what remains
    // is the genuine frozen core. We compare learners there.
    for n in [5usize, 8] {
        println!("-- {n}x{n}, horizon {LONG} --");
        report("Thompson", &run_h(Learner::Thompson, n, LONG, None));
        for c in [0.5, 1.0, 2.0] {
            report(
                &format!("FE c={c}"),
                &run_h(Learner::ForcedExplore, n, LONG, Some(c)),
            );
        }
        report("UCB1", &run_h(Learner::Ucb, n, LONG, None));
        println!();
    }
}

const LONG: usize = 24000;

/// Like `run`/`run_c` but with an explicit horizon and optional forced-explore c.
fn run_h(learner: Learner, n: usize, horizon: usize, c: Option<f64>) -> Stats {
    let tail = horizon / 5;
    let mut seedgen = Rng::new(20260606);
    let mut stalled = 0;
    let mut tail_rates = Vec::with_capacity(MARKETS);
    let mut worst_ratio = 0.0_f64;
    let mut total_regret_sum = 0.0;
    let half = horizon / 2;
    for _ in 0..MARKETS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, n);
        let mut market = match (learner, c) {
            (Learner::ForcedExplore, Some(cc)) => Market::with_forced_explore(
                util,
                recv,
                0.5,
                1.0,
                NOISE * NOISE,
                cc,
                NOISE,
                seed ^ 0xABCD,
            ),
            _ => build(learner, util, recv, seed ^ 0xABCD),
        };
        let rep = simulate(&mut market, horizon);
        let tr = rep.tail_mean_regret(tail);
        if tr > STALL_THRESHOLD {
            stalled += 1;
        }
        tail_rates.push(tr);
        let r_h = rep.cumulative_regret[half - 1].max(1e-9);
        let r_2h = rep.cumulative_regret[horizon - 1].max(1e-9);
        worst_ratio = worst_ratio.max(r_2h / r_h);
        total_regret_sum += rep.total_regret();
    }
    tail_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Stats {
        stalled,
        tail_rates,
        worst_ratio,
        mean_total_regret: total_regret_sum / MARKETS as f64,
    }
}
