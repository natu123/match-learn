//! The price of stability: what does insisting on a stable matching cost in
//! welfare, versus the welfare-optimal assignment?
//!
//! ```text
//! cargo run --release --example price_of_stability
//! ```
//!
//! For random markets we compare the proposer-side welfare of the proposer-
//! optimal *stable* matching (Gale-Shapley) against the *welfare-optimal*
//! matching (the Hungarian assignment, which may be unstable). Their ratio is the
//! price of stability — and it connects to the research finding that the learning
//! "stall" is really a proposer-optimality gap, not instability.

use match_learn::{Rng, gale_shapley, max_weight_assignment};

/// Proposer preference rankings implied by a utility matrix (descending).
fn prefs_from_util(util: &[Vec<f64>]) -> Vec<Vec<usize>> {
    util.iter()
        .map(|row| {
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            idx
        })
        .collect()
}

fn main() {
    let markets = 2000;
    let n = 8;

    let mut rng = Rng::new(2026);
    let mut stable_welfare = 0.0;
    let mut optimal_welfare = 0.0;
    let mut worst_ratio = 1.0_f64;

    for _ in 0..markets {
        // Proposer utilities for receivers; receivers have their own preferences.
        let util_p: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.uniform()).collect())
            .collect();
        let util_r: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.uniform()).collect())
            .collect();

        // Stable matching (proposer-optimal) and its proposer-side welfare.
        let stable = gale_shapley(&prefs_from_util(&util_p), &prefs_from_util(&util_r));
        let s_w: f64 = (0..n)
            .map(|p| stable.proposer[p].map_or(0.0, |r| util_p[p][r]))
            .sum();

        // Welfare-optimal proposer-side matching (ignores stability).
        let (_, o_w) = max_weight_assignment(&util_p);

        stable_welfare += s_w;
        optimal_welfare += o_w;
        if s_w > 0.0 {
            worst_ratio = worst_ratio.max(o_w / s_w);
        }
    }

    let ratio = optimal_welfare / stable_welfare;
    println!("Price of stability ({markets} random {n}x{n} markets)\n");
    println!(
        "  mean proposer welfare, stable (Gale-Shapley) : {:.3}",
        stable_welfare / markets as f64
    );
    println!(
        "  mean proposer welfare, welfare-optimal       : {:.3}",
        optimal_welfare / markets as f64
    );
    println!("  price of stability (optimal / stable)        : {ratio:.3}");
    println!("  worst single-market ratio                    : {worst_ratio:.3}");
    println!("\nStability costs some proposer welfare: the welfare-optimal matching is not");
    println!("stable, so a stable mechanism leaves the gap above on the table — the same");
    println!("proposer-optimality gap the learning stall converges into.");
}
