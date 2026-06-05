//! Benchmark the learners and the loop.
//!
//! ```text
//! cargo run --release --example benchmark
//! ```
//!
//! Two tables: (1) a quality comparison of the learners (Thompson / UCB /
//! Discounted) on the same correlated markets, and (2) throughput (rounds per
//! second) at a few market sizes.
//!
//! Note: this benchmarks `match-learn` against *itself* across configurations.
//! A cross-language comparison (e.g. Python MABWiser / `matching`) needs a Python
//! environment and is intentionally left out rather than approximated.

use match_learn::data::{correlated_market, prefs_from_util};
use match_learn::learner::{DiscountedThompson, PreferenceLearner, Ucb1};
use match_learn::{GaussianThompson, Market, Report, Rng, simulate};
use std::time::Instant;

/// Build a market over the given true utilities with a chosen learner family.
fn market_with(
    kind: &str,
    util_p: &[Vec<f64>],
    receiver_prefs: &[Vec<usize>],
    noise: f64,
    seed: u64,
) -> Market {
    let n_r = receiver_prefs.len();
    let learners: Vec<Box<dyn PreferenceLearner>> = (0..util_p.len())
        .map(|p| {
            let s = seed ^ (p as u64 + 1);
            match kind {
                "thompson" => Box::new(GaussianThompson::new(n_r, 0.5, 1.0, noise * noise, s))
                    as Box<dyn PreferenceLearner>,
                "ucb" => Box::new(Ucb1::new(n_r, 0.4)) as Box<dyn PreferenceLearner>,
                "discounted" => Box::new(DiscountedThompson::new(
                    n_r,
                    0.5,
                    1.0,
                    noise * noise,
                    0.999,
                    s,
                )) as Box<dyn PreferenceLearner>,
                _ => unreachable!(),
            }
        })
        .collect();
    Market::new(
        util_p.to_vec(),
        receiver_prefs.to_vec(),
        learners,
        noise,
        seed,
    )
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn quality_table() {
    let markets = 24;
    let n = 6;
    let rounds = 3000;
    let noise = 0.2;
    let kinds = ["thompson", "ucb", "discounted"];

    println!("Quality over {markets} correlated {n}x{n} markets, {rounds} rounds:\n");
    println!(
        "  {:<11} {:>14} {:>16} {:>14}",
        "learner", "total regret", "tail stable frac", "settled round"
    );

    for kind in kinds {
        let mut regrets = Vec::new();
        let mut stables = Vec::new();
        let mut settled = Vec::new();
        let mut seedgen = Rng::new(99);
        for _ in 0..markets {
            let seed = (seedgen.below(1_000_000_000) as u64) + 1;
            let mut g = Rng::new(seed);
            let (util_p, util_r) = correlated_market(&mut g, n, n, 0.5);
            let receiver_prefs = prefs_from_util(&util_r);
            let mut m = market_with(kind, &util_p, &receiver_prefs, noise, seed ^ 0xABCD);
            let rep: Report = simulate(&mut m, rounds);
            regrets.push(rep.total_regret());
            stables.push(rep.tail_stable_fraction(rounds / 5));
            settled.push(rep.settled_round().map_or(rounds as f64, |s| s as f64));
        }
        println!(
            "  {:<11} {:>14.2} {:>16.3} {:>14.0}",
            kind,
            mean(&regrets),
            mean(&stables),
            mean(&settled)
        );
    }
}

fn throughput_table() {
    let rounds = 2000;
    let noise = 0.2;
    let sizes = [5usize, 10, 25, 50, 100];

    println!("\nThroughput (Thompson, {rounds} rounds):\n");
    println!("  {:>5} {:>14} {:>16}", "size", "millis", "rounds/sec");

    for n in sizes {
        let mut g = Rng::new(7);
        let (util_p, util_r) = correlated_market(&mut g, n, n, 0.5);
        let receiver_prefs = prefs_from_util(&util_r);
        let mut m =
            Market::with_thompson(util_p, receiver_prefs, 0.5, 1.0, noise * noise, noise, 7);

        let start = Instant::now();
        for _ in 0..rounds {
            m.step();
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "  {:>5} {:>14.2} {:>16.0}",
            n,
            elapsed * 1000.0,
            rounds as f64 / elapsed
        );
    }
}

fn main() {
    println!("match-learn benchmark\n=====================\n");
    quality_table();
    throughput_table();
}
