//! Does the near-tie stall — and its annealing cure — generalize to the harder
//! *two-sided* market, where both sides learn?
//!
//! One-sided analysis showed genuine stalls are near-tie driven and that slow
//! annealing reduces them. Two-sided learning is noisier (each side ranks against
//! a moving target), so this is a stronger test. We compare plain Thompson on both
//! sides against annealed Thompson on both sides.
//!
//! ```text
//! cargo run --release --example two_sided_stall
//! ```

use match_learn::{ForcedExploreThompson, PreferenceLearner, Rng, TwoSidedMarket, simulate};

const N: usize = 5;
const NOISE: f64 = 0.2;
const MARKETS: usize = 300;
const HORIZON: usize = 20000;
const STALL_THRESHOLD: f64 = 0.05;

fn side(
    n_arms: usize,
    n_agents: usize,
    tau: Option<f64>,
    base_seed: u64,
) -> Vec<Box<dyn PreferenceLearner>> {
    (0..n_agents)
        .map(|a| {
            let mut l = ForcedExploreThompson::new(
                n_arms,
                0.5,
                1.0,
                NOISE * NOISE,
                0.0,
                base_seed + a as u64,
            );
            if let Some(t) = tau {
                l = l.with_anneal(t);
            }
            Box::new(l) as Box<dyn PreferenceLearner>
        })
        .collect()
}

fn run(tau: Option<f64>) -> (usize, f64) {
    let tail = HORIZON / 5;
    let mut seedgen = Rng::new(20260606);
    let mut stalls = 0;
    let mut regret_sum = 0.0;
    for _ in 0..MARKETS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut g = Rng::new(seed);
        let util_p: Vec<Vec<f64>> = (0..N)
            .map(|_| (0..N).map(|_| g.uniform()).collect())
            .collect();
        let util_r: Vec<Vec<f64>> = (0..N)
            .map(|_| (0..N).map(|_| g.uniform()).collect())
            .collect();
        let plearn = side(N, N, tau, seed ^ 0x2000);
        let rlearn = side(N, N, tau, seed ^ 0x4000);
        let mut market = TwoSidedMarket::new(util_p, util_r, plearn, rlearn, NOISE, seed ^ 0xABCD);
        let rep = simulate(&mut market, HORIZON);
        if rep.tail_mean_regret(tail) > STALL_THRESHOLD {
            stalls += 1;
        }
        regret_sum += rep.total_regret();
    }
    (stalls, regret_sum / MARKETS as f64)
}

fn main() {
    println!(
        "Two-sided stall — {MARKETS} random {N}x{N} markets, both sides learn, horizon {HORIZON}, noise {NOISE}\n"
    );
    let (ts_stalls, ts_regret) = run(None);
    println!(
        "Thompson (both sides):  {ts_stalls:>3}/{MARKETS} stalls, mean total regret {ts_regret:>8.2}"
    );
    for tau in [16000.0, 8000.0] {
        let (an_stalls, an_regret) = run(Some(tau));
        println!(
            "Annealed (tau={tau:>7.0}):   {an_stalls:>3}/{MARKETS} stalls, mean total regret {an_regret:>8.2}"
        );
    }
    println!(
        "\n(Generality check: if annealing lowers the two-sided stall count and/or regret\ntoo, the near-tie phenomenon and its cure are not specific to one-sided markets.)"
    );
}
