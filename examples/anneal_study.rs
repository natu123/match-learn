//! Annealing study: test whether annealing the Thompson temperature cures the
//! *dominant* (near-tie churn) stall that forced exploration alone could not.
//!
//! Four learners on the same long-horizon markets:
//! - `TS`   plain Thompson (c=0, no anneal),
//! - `FE`   forced exploration only (c>0, no anneal) — frozen-arm insurance,
//! - `AN`   annealing only (c=0, finite tau) — churn suppression,
//! - `AF`   both (c>0, finite tau) — robust to both failure modes.
//!
//! We report the genuine-stall rate (tail regret rate above a threshold at a long
//! horizon), the tail-regret distribution, and mean regret. If annealing collapses
//! the stall rate while forcing alone does not, the dominant stall is churn.
//!
//! ```text
//! cargo run --release --example anneal_study
//! ```

use match_learn::{ForcedExploreThompson, Market, PreferenceLearner, Rng, simulate};

const NOISE: f64 = 0.2;
const MARKETS: usize = 400;
const HORIZON: usize = 24000;
const STALL_THRESHOLD: f64 = 0.05;

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

/// Build a market whose proposers are `ForcedExploreThompson(c)` optionally
/// annealed with timescale `tau` (None = no annealing).
fn build(
    util: Vec<Vec<f64>>,
    recv: Vec<Vec<usize>>,
    c: f64,
    tau: Option<f64>,
    seed: u64,
) -> Market {
    let n_r = recv.len();
    let learners: Vec<Box<dyn PreferenceLearner>> = (0..util.len())
        .map(|p| {
            let mut l = ForcedExploreThompson::new(
                n_r,
                0.5,
                1.0,
                NOISE * NOISE,
                c,
                seed ^ (0x2000 + p as u64),
            );
            if let Some(t) = tau {
                l = l.with_anneal(t);
            }
            Box::new(l) as Box<dyn PreferenceLearner>
        })
        .collect();
    Market::new(util, recv, learners, NOISE, seed)
}

struct Stats {
    stalled: usize,
    p99: f64,
    worst: f64,
    mean_regret: f64,
}

fn run(n: usize, c: f64, tau: Option<f64>) -> Stats {
    let tail = HORIZON / 5;
    let mut seedgen = Rng::new(20260606);
    let mut stalled = 0;
    let mut tails = Vec::with_capacity(MARKETS);
    let mut regret_sum = 0.0;
    for _ in 0..MARKETS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, n);
        let mut market = build(util, recv, c, tau, seed ^ 0xABCD);
        let rep = simulate(&mut market, HORIZON);
        let tr = rep.tail_mean_regret(tail);
        if tr > STALL_THRESHOLD {
            stalled += 1;
        }
        tails.push(tr);
        regret_sum += rep.total_regret();
    }
    tails.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| tails[(((tails.len() - 1) as f64) * f).round() as usize];
    Stats {
        stalled,
        p99: q(0.99),
        worst: *tails.last().unwrap(),
        mean_regret: regret_sum / MARKETS as f64,
    }
}

fn report(name: &str, s: &Stats) {
    println!(
        "  {name:<22} stalled {:>3}/{MARKETS} ({:>5.2}%)  tail p99={:>8.5} worst={:>8.5}  mean_regret={:>8.2}",
        s.stalled,
        100.0 * s.stalled as f64 / MARKETS as f64,
        s.p99,
        s.worst,
        s.mean_regret,
    );
}

fn main() {
    println!(
        "Annealing study — {MARKETS} markets/size, horizon {HORIZON}, noise {NOISE}, stall threshold {STALL_THRESHOLD}\n"
    );
    for n in [5usize, 8] {
        println!("== {n}x{n} ==");
        report("TS (c=0)", &run(n, 0.0, None));
        report("FE (c=0.5)", &run(n, 0.5, None));
        for tau in [8000.0, 2000.0, 500.0] {
            report(&format!("AN (tau={tau})"), &run(n, 0.0, Some(tau)));
        }
        for tau in [2000.0, 500.0] {
            report(&format!("AF (c=0.5, tau={tau})"), &run(n, 0.5, Some(tau)));
        }
        println!();
    }
}
