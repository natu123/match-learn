//! CSV data for paper-1's central figure: measured stable regret vs horizon T
//! in the four regimes of the Two-Floor 2x2, on the embedded general market.
//!
//! Columns: regime,T,mean_regret  (mean over 24 seeds; market = embed(0.15, 0.25, 4))
//!
//! Run with: `cargo run --release --example fig_two_by_two`
//! Reproduces Figure 1 of the matching paper (doi:10.5281/zenodo.20627356).

use match_learn::embedding::{embed, simulate_market};
use match_learn::irreversible::Policy;
use match_learn::prefs::rank_by_scores;

fn mean_regret(reversible: bool, policy: Policy, t: usize) -> f64 {
    let (prop, recv) = embed(0.15, 0.25, 4, false);
    let recv_ranks: Vec<Vec<usize>> = recv.iter().map(|r| rank_by_scores(r)).collect();
    (1..=24u64)
        .map(|s| simulate_market(&prop, &recv_ranks, reversible, policy, t, s).final_regret())
        .sum::<f64>()
        / 24.0
}

fn main() {
    let iv = Policy::Interview { per_round: 2 };
    let regimes: [(&str, bool, Policy); 4] = [
        ("absorbing/no-interview", false, Policy::NoInterview),
        ("absorbing/interview", false, iv),
        ("recoverable/no-interview", true, Policy::NoInterview),
        ("recoverable/interview", true, iv),
    ];
    println!("regime,T,mean_regret");
    for (name, rev, pol) in regimes {
        for t in [1000usize, 2000, 4000, 6000, 8000, 12000, 16000, 20000] {
            println!("{name},{t},{:.2}", mean_regret(rev, pol, t));
        }
    }
}
