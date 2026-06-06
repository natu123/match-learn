//! The regret-queue tradeoff in dynamic pricing (Phase 7-d).
//!
//! Learning the right price means *exploring* prices — and every round spent at
//! an off-clearing price both costs matched volume (regret) and pushes the
//! queues out of balance. More exploration learns faster but disturbs the queue
//! more; too little risks settling on the wrong price. This example sweeps the
//! UCB exploration constant and prints both costs so the tradeoff is visible.
//!
//! ```text
//! cargo run --release --example regret_queue
//! ```

use match_learn::marketplace::{Demand, Marketplace, Supply};
use match_learn::pricing::{LearnedPricer, Objective, price_grid};

fn make_market(seed: u64) -> Marketplace {
    let demand = Demand {
        base: 12.0,
        max_price: 20.0,
    };
    let supply = Supply {
        base: 12.0,
        ref_price: 10.0,
    };
    Marketplace::new(demand, supply, 0.02, seed)
}

fn main() {
    let grid = price_grid(1.0, 18.0, 18);
    let rounds = 20_000;
    let seed = 7;

    // Oracle: always post the (known) clearing price.
    let clearing = make_market(0).clearing_price();
    let mut oracle = make_market(seed);
    let mut oracle_matched = 0usize;
    let mut oracle_imbalance = 0.0;
    for _ in 0..rounds {
        let o = oracle.step(clearing);
        oracle_matched += o.matched;
        oracle_imbalance += (o.demand_queue as f64 - o.supply_queue as f64).abs();
    }

    println!("Regret-queue tradeoff over {rounds} rounds (clearing price p* = {clearing:.2})\n");
    println!(
        "  {:>6}  {:>9}  {:>16}  {:>18}",
        "UCB c", "matched", "regret (vs p*)", "mean |dq - sq|"
    );
    println!(
        "  {:>6}  {:>9}  {:>16}  {:>18.2}",
        "oracle",
        oracle_matched,
        0,
        oracle_imbalance / rounds as f64
    );

    for c in [0.1, 0.3, 0.7, 1.5, 3.0, 6.0] {
        let mut pricer = LearnedPricer::with_ucb(grid.clone(), c, Objective::Throughput);
        let mut m = make_market(seed);
        let mut matched = 0usize;
        let mut imbalance = 0.0;
        for _ in 0..rounds {
            let o = pricer.step(&mut m);
            matched += o.matched;
            imbalance += (o.demand_queue as f64 - o.supply_queue as f64).abs();
        }
        let regret = oracle_matched as i64 - matched as i64;
        println!(
            "  {:>6.1}  {:>9}  {:>16}  {:>18.2}",
            c,
            matched,
            regret,
            imbalance / rounds as f64
        );
    }

    println!("\nLow c exploits fast (little queue disturbance) but risks the wrong price;");
    println!("high c explores more (heavier queue imbalance) for robustness. The sweet");
    println!("spot is interior — the regret-queue tradeoff the platform must tune.");
}
