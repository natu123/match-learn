//! Watch a market learn: run the loop on a synthetic market and print how regret
//! and stability evolve.
//!
//! ```text
//! cargo run --example converge
//! ```

use match_learn::{Market, Rng, simulate};

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

fn main() {
    let n = 5;
    let rounds = 1500;
    let noise = 0.2;

    let mut rng = Rng::new(2026);
    let (true_util, receiver_prefs) = random_market(&mut rng, n);

    // Well-specified Thompson Sampling: assumed noise matches the true noise.
    let mut market = Market::with_thompson(
        true_util.clone(),
        receiver_prefs.clone(),
        0.5,
        1.0,
        noise * noise,
        noise,
        7,
    );

    println!("match-learn — {n}x{n} market, {rounds} rounds, Thompson Sampling\n");

    let report = simulate(&mut market, rounds);

    // Cumulative-regret curve, sampled every 150 rounds, with a tiny bar chart.
    let max = report
        .cumulative_regret
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    println!("cumulative regret over time:");
    for t in (149..rounds).step_by(150) {
        let r = report.cumulative_regret[t];
        let bar = "#".repeat(((r / max) * 40.0).round().max(0.0) as usize);
        println!("  round {:>4}: {:>7.2} |{}", t + 1, r, bar);
    }

    println!();
    println!("rounds to settle      : {:?}", report.settled_round());
    println!(
        "tail stable fraction  : {:.3}",
        report.tail_stable_fraction(rounds / 5)
    );
    println!(
        "tail regret per round : {:.4}",
        report.tail_mean_regret(rounds / 5)
    );
    println!("total regret          : {:.2}", report.total_regret());
    println!();
    println!("The matching converges onto the stable matching of the true market,");
    println!("and cumulative regret flattens — learning, while staying stable.");
}
