//! The when-to-match tradeoff in online dynamic matching.
//!
//! ```text
//! cargo run --release --example online_matching
//! ```
//!
//! Agents arrive over time and abandon if they wait too long. Matching every
//! tick (greedy) keeps waits short but pairs from a thin pool; batching waits to
//! accumulate a richer pool for closer matches, at the cost of more abandonment.
//! This sweeps the batch interval to show the tradeoff.

use match_learn::online::{OnlineMarket, Policy};

fn main() {
    let arrivals = 3.0;
    let abandon = 0.04;
    let ticks = 20_000;
    let seed = 7;

    println!(
        "Online matching: {arrivals} arrivals/side/tick, {:.0}% abandon/tick, {ticks} ticks\n",
        abandon * 100.0
    );
    println!(
        "  {:>12}  {:>9}  {:>11}  {:>13}",
        "policy", "matched", "abandoned", "mean distance"
    );

    let greedy = OnlineMarket::new(arrivals, abandon, seed).run(ticks, Policy::Greedy);
    println!(
        "  {:>12}  {:>9}  {:>11}  {:>13.4}",
        "greedy",
        greedy.matched,
        greedy.abandoned,
        greedy.mean_distance()
    );

    for k in [2usize, 4, 8, 16, 32] {
        let s = OnlineMarket::new(arrivals, abandon, seed).run(ticks, Policy::Batched(k));
        println!(
            "  {:>12}  {:>9}  {:>11}  {:>13.4}",
            format!("batched({k})"),
            s.matched,
            s.abandoned,
            s.mean_distance()
        );
    }

    println!(
        "\nLonger batching lowers mean match distance (better pairs) but abandons more"
    );
    println!("waiting agents — the timing tradeoff a dynamic platform must set.");
}
