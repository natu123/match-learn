//! Irreversible matching with interviews — the falsifiable experiment behind
//! "interviews substitute for reversibility" (match-learn paper 1).
//!
//! Each agent faces firms with one `heaven` (0.9), one `decoy` (`0.9 - Δ`, the
//! near-tie that must be resolved), and the rest `hell` (0.1). We sweep the 2x2
//! of {recoverable, irreversible} x {no-interview, interview} and read the
//! *shape* of cumulative regret: only the irreversible market with no interviews
//! grows linearly — every other cell is learnable because reversibility or
//! interviews supplies safe information.
//!
//! Run with: `cargo run --example irreversible_interviews`

use match_learn::irreversible::{Market, Policy, simulate};
use match_learn::rng::Rng;

fn main() {
    let n = 500; // agents
    let m = 4; // firms each
    let threshold = 0.3; // commit below this mean = catastrophe
    let interview = Policy::Interview { per_round: 2 };
    let seed = 20_260_608;

    println!("Irreversible matching: interviews substitute for reversibility");
    println!(
        "{n} agents, {m} firms each (1 heaven=0.9, 1 decoy=0.9-Δ, {} hell=0.1); Δ=0.2\n",
        m - 2
    );

    println!("== 2x2: cumulative regret and its growth from T=2k to T=20k ==");
    println!(
        "{:<28}{:>10}{:>11}{:>9}{:>9}{:>8}",
        "regime", "reg@2k", "reg@20k", "growth", "catastr", "shape"
    );
    let regimes = [
        (
            "irreversible / no-interview",
            false,
            Policy::NoInterview,
            "Ω(T)",
        ),
        ("irreversible / interview", false, interview, "log T"),
        (
            "recoverable  / no-interview",
            true,
            Policy::NoInterview,
            "log T",
        ),
        ("recoverable  / interview", true, interview, "O(1)"),
    ];
    for (label, reversible, policy, shape) in regimes {
        let market = Market::heaven_or_hell(n, m, 0.9, 0.2, 0.1, &mut Rng::new(seed));
        let short = simulate(&market, reversible, policy, 2_000, threshold, 1);
        let long = simulate(&market, reversible, policy, 20_000, threshold, 1);
        println!(
            "{:<28}{:>10.0}{:>11.0}{:>8.2}x{:>8.1}%{:>8}",
            label,
            short.final_regret(),
            long.final_regret(),
            long.final_regret() / short.final_regret(),
            100.0 * long.catastrophe_rate(),
            shape
        );
    }
    println!("  growth ~10x only for irreversible + no-interview  -> Ω(T), unlearnable.");
    println!("  every other cell is sublinear: reversibility OR interviews makes it learnable,");
    println!("  and an interview substitutes for an undo (the off-diagonal both read log T).\n");

    println!("== irreversible + interview: regret vs deciding gap Δ at fixed T  (∝ 1/Δ²) ==");
    println!("{:>6}{:>14}{:>17}", "Δ", "regret@8k", "x vs prev (~4?)");
    let mut prev: Option<f64> = None;
    for &gap in &[0.4, 0.2, 0.1] {
        let market = Market::heaven_or_hell(n, m, 0.9, gap, 0.1, &mut Rng::new(seed));
        let out = simulate(&market, false, interview, 8_000, threshold, 1);
        let r = out.final_regret();
        let mult = prev.map_or_else(|| "-".to_string(), |p| format!("{:.2}x", r / p));
        println!("{gap:>6.2}{r:>14.0}{mult:>17}");
        prev = Some(r);
    }
    println!("  halving Δ ~quadruples regret: the σ²/Δ² identification cost,");
    println!("  paid safely through interviews instead of irreversible mismatches.");
}
