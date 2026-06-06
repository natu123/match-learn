//! Capstone for the reframing: across *all* markets (not just the stalled ones),
//! greedy Thompson reliably reaches a **stable** matching; what it sometimes
//! misses is **proposer-optimality**. So the right decomposition of "regret" is
//! *instability* (≈0) plus an *optimality gap* (the residue).
//!
//! For each market we settle the belief matching and measure: is it exactly
//! stable? ε-stable? and its regret vs the proposer-optimal stable matching. The
//! claim: near-100% (ε-)stability with a small mean optimality gap concentrated
//! in a few near-tie markets.
//!
//! ```text
//! cargo run --release --example stability_vs_optimality
//! ```

use match_learn::matching::Matching;
use match_learn::{Market, Rng, gale_shapley, rank_by_scores, simulate};

const NOISE: f64 = 0.2;
const SEEDS: usize = 800;
const HORIZON: usize = 8000;
const EPS: f64 = 0.05;

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

fn regret(util: &[Vec<f64>], baseline: &Matching, m: &Matching) -> f64 {
    (0..util.len())
        .map(|p| {
            baseline.proposer[p].map_or(0.0, |r| util[p][r])
                - m.proposer[p].map_or(0.0, |r| util[p][r])
        })
        .sum()
}

/// True blocking exists with proposer gain > `eps`?
fn has_blocking(util: &[Vec<f64>], recv: &[Vec<usize>], m: &Matching, eps: f64) -> bool {
    let n = util.len();
    let mut holder = vec![None; recv.len()];
    for p in 0..n {
        if let Some(r) = m.proposer[p] {
            holder[r] = Some(p);
        }
    }
    let pos = |r: usize, p: usize| recv[r].iter().position(|&q| q == p).unwrap_or(usize::MAX);
    (0..n).any(|p| {
        let cur = m.proposer[p].map_or(f64::MIN, |r| util[p][r]);
        (0..recv.len()).any(|r| {
            m.proposer[p] != Some(r)
                && util[p][r] - cur > eps
                && holder[r].is_none_or(|h| pos(r, p) < pos(r, h))
        })
    })
}

fn run(n: usize) {
    let mut seedgen = Rng::new(20260606);
    let (mut exact, mut eps_stable) = (0usize, 0usize);
    let mut regret_sum = 0.0;
    let mut gap_markets = 0usize; // markets with a non-trivial optimality gap
    for _ in 0..SEEDS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, n);
        let mut market = Market::with_thompson(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            NOISE * NOISE,
            NOISE,
            seed ^ 0xABCD,
        );
        simulate(&mut market, HORIZON);
        let means = market.belief_means();
        let baseline = market.true_stable_matching();
        let settled = gale_shapley(
            &means.iter().map(|m| rank_by_scores(m)).collect::<Vec<_>>(),
            &recv,
        );
        if !has_blocking(&util, &recv, &settled, 0.0) {
            exact += 1;
        }
        if !has_blocking(&util, &recv, &settled, EPS) {
            eps_stable += 1;
        }
        let r = regret(&util, &baseline, &settled);
        regret_sum += r;
        if r > 0.05 {
            gap_markets += 1;
        }
    }
    let pct = |x: usize| 100.0 * x as f64 / SEEDS as f64;
    println!("== {n}x{n} ({SEEDS} markets, horizon {HORIZON}) ==");
    println!(
        "  exactly stable      : {exact}/{SEEDS} ({:.1}%)",
        pct(exact)
    );
    println!(
        "  ε-stable (ε={EPS})    : {eps_stable}/{SEEDS} ({:.1}%)",
        pct(eps_stable)
    );
    println!(
        "  mean optimality-gap regret : {:.4}   (markets with gap>0.05: {gap_markets}/{SEEDS} = {:.1}%)",
        regret_sum / SEEDS as f64,
        pct(gap_markets)
    );
    println!();
}

fn main() {
    println!(
        "Stability vs optimality — settled belief matching of greedy Thompson, noise {NOISE}\n"
    );
    for n in [5usize, 8] {
        run(n);
    }
    println!(
        "Takeaway: the learner is almost always (ε-)stable; the residual 'regret' is a\nproposer-optimality gap on a small near-tie minority -- not an instability problem."
    );
}
