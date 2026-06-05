//! Export a run as CSV for plotting the matching/preference evolution.
//!
//! ```text
//! cargo run --example export_csv > run.csv
//! ```
//!
//! Columns: `round,cumulative_regret,stable` (one row per round). Pipe it to a
//! file and plot with whatever you like (gnuplot, Python, a spreadsheet).

use match_learn::data::{correlated_market, prefs_from_util};
use match_learn::{Market, Rng, simulate};

fn main() {
    let n = 6;
    let rounds = 2000;
    let noise = 0.2;

    let mut rng = Rng::new(2026);
    let (util_p, util_r) = correlated_market(&mut rng, n, n, 0.5);
    let receiver_prefs = prefs_from_util(&util_r);

    let mut market =
        Market::with_thompson(util_p, receiver_prefs, 0.5, 1.0, noise * noise, noise, 7);
    let report = simulate(&mut market, rounds);

    println!("round,cumulative_regret,stable");
    for t in 0..report.rounds {
        println!(
            "{},{:.6},{}",
            t + 1,
            report.cumulative_regret[t],
            if report.stable[t] { 1 } else { 0 }
        );
    }
}
