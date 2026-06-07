//! The theory-parameter = computed-object = measured-driver trinity on a general
//! market (match-learn paper 1, the general-market figure).
//!
//! A pivotal near-tie of gap `Δ_A` is embedded in an `n × n` market whose other
//! gaps are all a wide `Δ_big`. Three numbers then coincide:
//!   - the *theory parameter* that sets the regret rate, `Δ_A`;
//!   - the *computed object* `admissible_gap(whole market)`;
//!   - the *measured driver* of the simulated stable regret.
//!
//! And all three ignore the rigid core (`Δ_big`) — outcome-relativity.
//!
//! Run with: `cargo run --example embedding_trinity`

use match_learn::admissible::admissible_gap;
use match_learn::embedding::{embed, simulate_market};
use match_learn::irreversible::Policy;
use match_learn::prefs::rank_by_scores;

fn mean_regret(delta_a: f64, delta_big: f64, reversible: bool, policy: Policy, t: usize) -> f64 {
    let (prop, recv) = embed(delta_a, delta_big, 4, false);
    let recv_ranks: Vec<Vec<usize>> = recv.iter().map(|r| rank_by_scores(r)).collect();
    (1..=12u64)
        .map(|s| simulate_market(&prop, &recv_ranks, reversible, policy, t, s).final_regret())
        .sum::<f64>()
        / 12.0
}

fn main() {
    let (delta_a, delta_big, n) = (0.15, 0.25, 4);
    let iv = Policy::Interview { per_round: 2 };

    let (prop, recv) = embed(delta_a, delta_big, n, false);
    let da = admissible_gap(&prop, &recv);

    println!("Embedding trinity: one pivotal Δ_A near-tie in an {n}×{n} market,");
    println!("every other gap a wide Δ_big={delta_big} rigid core.\n");
    println!(
        "COMPUTED OBJECT : admissible_gap(whole market) = {da:.3}   (= pivot Δ_A = {delta_a})\n"
    );

    println!(
        "MEASURED DRIVER : stable regret, 2×2 over {{recoverable, irreversible}} × {{no-int, interview}}"
    );
    println!(
        "{:<24}{:>10}{:>11}{:>9}{:>8}",
        "regime", "reg@4k", "reg@16k", "growth", "shape"
    );
    for (label, rev, pol, shape) in [
        ("irreversible/no-int", false, Policy::NoInterview, "Ω(T)"),
        ("irreversible/interview", false, iv, "log T"),
        ("recoverable /no-int", true, Policy::NoInterview, "log T"),
        ("recoverable /interview", true, iv, "O(1)"),
    ] {
        let r1 = mean_regret(delta_a, delta_big, rev, pol, 4000);
        let r2 = mean_regret(delta_a, delta_big, rev, pol, 16000);
        println!("{label:<24}{r1:>10.0}{r2:>11.0}{:>8.2}x{shape:>8}", r2 / r1);
    }
    println!("  only irreversible + no-interview is linear -> Ω(T); the rest are sublinear.\n");

    println!("DRIVER = Δ_A : irreversible/interview regret vs the pivot (∝ 1/Δ_A²)");
    let (w, nn) = (
        mean_regret(0.2, 0.25, false, iv, 8000),
        mean_regret(0.1, 0.25, false, iv, 8000),
    );
    println!(
        "  Δ_A=0.20 -> {w:.0}   Δ_A=0.10 -> {nn:.0}   ratio {:.2}x (~4 = inverse square)\n",
        nn / w
    );

    println!("CORE IS FREE : same regret as the rigid core Δ_big varies (outcome-relativity)");
    let (c1, c2) = (
        mean_regret(0.15, 0.2, false, iv, 6000),
        mean_regret(0.15, 0.3, false, iv, 6000),
    );
    println!(
        "  Δ_big=0.20 -> {c1:.0}   Δ_big=0.30 -> {c2:.0}   ratio {:.2}x (~1 = invariant)",
        c2 / c1
    );
}
