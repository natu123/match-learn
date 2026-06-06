//! Does coordination actually cure the cascade stall end-to-end?
//!
//! ```text
//! cargo run --release --example coordinated_validation
//! ```
//!
//! Builds markets with a deliberate near-tie (one proposer nearly indifferent
//! between two receivers) — the cascade trigger — and compares plain Thompson
//! Sampling against `CoordinatedMarket` on tail stability and regret.
//!
//! Spoiler / honest result: the live coordinator does **not** beat plain Thompson
//! on stability here — it maximizes belief welfare, raising proposer welfare but
//! lowering the strict is-stable fraction. The post-hoc cascade cure does not
//! transfer naively to the live loop. The printout explains why.

use match_learn::{CoordinatedMarket, Market, Rng, simulate};

/// A market with `n` proposers where proposer 0 is nearly indifferent between
/// receivers 0 and 1 (means within `gap`), a near-tie that can cascade.
fn near_tie_market(rng: &mut Rng, n: usize, gap: f64) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let mut util: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    // Force a near-tie at the top of proposer 0's preferences.
    util[0][0] = 0.90;
    util[0][1] = 0.90 - gap;
    let receiver_prefs: Vec<Vec<usize>> = (0..n).map(|_| rng.permutation(n)).collect();
    (util, receiver_prefs)
}

fn main() {
    let markets = 200;
    let n = 5;
    let rounds = 3000;
    let noise = 0.2;
    let gap = 0.01; // a tight near-tie
    let tail = rounds / 5;

    let mut seedgen = Rng::new(2026);
    let (mut ts_stable, mut ts_regret) = (0.0, 0.0);
    let (mut co_stable, mut co_regret) = (0.0, 0.0);

    for _ in 0..markets {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut g = Rng::new(seed);
        let (util, recv) = near_tie_market(&mut g, n, gap);

        let mut ts = Market::with_thompson(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            noise * noise,
            noise,
            seed ^ 0xABCD,
        );
        let r = simulate(&mut ts, rounds);
        ts_stable += r.tail_stable_fraction(tail);
        ts_regret += r.tail_mean_regret(tail);

        let mut co = CoordinatedMarket::new(
            util,
            recv,
            0.5,
            1.0,
            noise * noise,
            0.05,
            0.5,
            noise,
            seed ^ 0xABCD,
        );
        let r = simulate(&mut co, rounds);
        co_stable += r.tail_stable_fraction(tail);
        co_regret += r.tail_mean_regret(tail);
    }

    let m = markets as f64;
    println!("Cascade cure validation ({markets} near-tie {n}x{n} markets, gap={gap})\n");
    println!(
        "  {:<22} {:>16} {:>16}",
        "policy", "tail stable frac", "tail regret/round"
    );
    println!(
        "  {:<22} {:>16.3} {:>16.4}",
        "plain Thompson",
        ts_stable / m,
        ts_regret / m
    );
    println!(
        "  {:<22} {:>16.3} {:>16.4}",
        "CoordinatedMarket",
        co_stable / m,
        co_regret / m
    );
    println!();
    println!("Honest reading: the live coordinator maximizes *belief welfare*, so it raises");
    println!("proposer welfare (lower / negative regret) but is *less* stable than plain");
    println!("Thompson on the strict is-stable metric. The post-hoc cascade cure (coordinating");
    println!("*converged* beliefs) does NOT transfer naively to the live loop, where early");
    println!("inaccurate beliefs mislead the welfare search -- a finding for the research");
    println!("track: a live coordinator should gate on belief confidence or target stability,");
    println!("not belief welfare. (Cf. docs/theory-identifiability.md, Prop 3's band assumption.)");
}
