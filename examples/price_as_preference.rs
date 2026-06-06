//! Price as a proxy for preference (Phase 8).
//!
//! The project's long-run thesis is that a single price can stand in for a whole
//! profile of preferences: set the price right and the agents who *should* trade
//! self-select in. This example quantifies that. For random two-sided markets it
//! compares three welfare levels:
//!
//! - **efficient** — the maximum gains from trade, pairing the highest-value
//!   buyers with the lowest-cost sellers (an upper bound; the efficient double
//!   auction);
//! - **best price** — the welfare of the preference-stable matching among the
//!   entrants admitted by the single best posted price;
//! - **naive price** — the same at a fixed mid price.
//!
//! The best single price recovers most of the efficient welfare, even though it
//! must also respect the two-sided preference matching — price *is* a strong
//! proxy for preference.
//!
//! ```text
//! cargo run --release --example price_as_preference
//! ```

use match_learn::auction::double_auction;
use match_learn::pricing::price_grid;
use match_learn::{Rng, random_joint_instance};

fn main() {
    let grid = price_grid(0.02, 0.98, 25);
    let markets = 2000;
    let n = 25;

    let mut rng = Rng::new(2026);
    let mut efficient_sum = 0.0;
    let mut best_sum = 0.0;
    let mut naive_sum = 0.0;

    for _ in 0..markets {
        let inst = random_joint_instance(&mut rng, n, n);

        // Efficient upper bound: maximum gains from trade (ignores who-pairs-whom).
        let efficient = double_auction(&inst.demand_values, &inst.supply_costs).welfare;

        // Best single posted price (preference-stable matching among entrants).
        let best = grid
            .iter()
            .map(|&p| inst.welfare_at(p))
            .fold(0.0_f64, f64::max);

        // A naive fixed mid price.
        let naive = inst.welfare_at(0.5);

        efficient_sum += efficient;
        best_sum += best;
        naive_sum += naive;
    }

    let efficient = efficient_sum / markets as f64;
    let best = best_sum / markets as f64;
    let naive = naive_sum / markets as f64;

    println!("Price as a proxy for preference ({markets} random {n}x{n} markets)\n");
    println!("  welfare, efficient upper bound : {efficient:8.3}");
    println!(
        "  welfare, best single price     : {best:8.3}  ({:.1}% of efficient)",
        100.0 * best / efficient
    );
    println!(
        "  welfare, naive mid price        : {naive:8.3}  ({:.1}% of efficient)",
        100.0 * naive / efficient
    );
    println!("\nA single well-chosen price recovers most of the achievable welfare while");
    println!("respecting the two-sided preference matching: price proxies preference.");
}
