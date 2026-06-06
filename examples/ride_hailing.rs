//! Surge pricing for ride-hailing, learned online.
//!
//! ```text
//! cargo run --release --example ride_hailing
//! ```
//!
//! Each round a fresh batch of riders and drivers appears on the plane. The
//! platform posts a fare; riders below their value and drivers above their cost
//! join, and are matched by proximity. The same UCB bandit that learns any price
//! learns the surge fare that completes the most rides.

use match_learn::applications::random_ride_hailing;
use match_learn::learner::{PreferenceLearner, Ucb1};
use match_learn::pricing::price_grid;
use match_learn::rng::Rng;

fn main() {
    let grid = price_grid(0.05, 0.95, 19);
    let rounds = 20_000;

    // Average rides per fare over many snapshots (the demand-supply curve).
    let mut sample_rng = Rng::new(100);
    let mut totals = vec![0u64; grid.len()];
    for _ in 0..600 {
        let market = random_ride_hailing(&mut sample_rng, 20, 20);
        for (k, &p) in grid.iter().enumerate() {
            totals[k] += market.rides_at(p) as u64;
        }
    }

    // Learn the surge fare online.
    let mut learner = Ucb1::new(grid.len(), 3.0);
    let mut rng = Rng::new(500);
    for _ in 0..rounds {
        let market = random_ride_hailing(&mut rng, 20, 20);
        let arm = learner.ranking()[0];
        let rides = market.rides_at(grid[arm]);
        learner.update(arm, rides as f64);
    }
    let means = learner.means();
    let learned_arm = (0..grid.len())
        .max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap())
        .unwrap();
    let oracle_arm = (0..grid.len()).max_by_key(|&k| totals[k]).unwrap();

    println!("Ride-hailing surge pricing (20 riders x 20 drivers per round)\n");
    println!("  {:>5}  {:>14}", "fare", "avg rides/round");
    let max = *totals.iter().max().unwrap() as f64;
    for (k, &p) in grid.iter().enumerate() {
        let avg = totals[k] as f64 / 600.0;
        let bar = "#".repeat(((totals[k] as f64 / max) * 30.0).round() as usize);
        let mark = if k == oracle_arm { " <- best" } else { "" };
        println!("  {p:>5.2}  {avg:>14.2}  {bar}{mark}");
    }
    println!();
    println!("learned surge fare : {:.2}", grid[learned_arm]);
    println!("oracle  surge fare : {:.2}", grid[oracle_arm]);
    println!("\nThe same bandit that learns matching preferences learns the surge price.");
}
