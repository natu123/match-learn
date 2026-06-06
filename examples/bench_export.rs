//! Export instances and Rust timings for the cross-language benchmark.
//!
//! Writes three files under `bench/`:
//! - `instances.txt` — `M` random `n x n` complete-preference instances.
//! - `rust_matching.txt` — Rust Gale-Shapley result for each instance, plus the
//!   total solve time, so Python can verify it produced the same matchings.
//! - `rust_bandit.txt` — time for a fixed UCB1 online bandit workload.
//!
//! ```text
//! cargo run --release --example bench_export
//! ```
//! Then run `bench/python_bench.py` (see `bench/README.md`).

use match_learn::data::{correlated_market, prefs_from_util};
use match_learn::learner::{PreferenceLearner, Ucb1};
use match_learn::{Market, Rng, gale_shapley, simulate};
use std::fmt::Write as _;
use std::fs;
use std::time::Instant;

const M: usize = 2000; // instances
const N: usize = 20; // market size
const BANDIT_ARMS: usize = 10;
const BANDIT_ROUNDS: usize = 200_000;
const INTEGRATED_N: usize = 8; // integrated market size (small enough to converge)
const INTEGRATED_ROUNDS: usize = 4000;

fn main() {
    fs::create_dir_all("bench").expect("create bench/");

    // --- Generate instances (complete preferences, equal sizes). ---
    let mut rng = Rng::new(20260606);
    let mut instances = Vec::with_capacity(M);
    let mut instances_txt = format!("{M} {N}\n");
    for _ in 0..M {
        let prop: Vec<Vec<usize>> = (0..N).map(|_| rng.permutation(N)).collect();
        let recv: Vec<Vec<usize>> = (0..N).map(|_| rng.permutation(N)).collect();
        for list in prop.iter().chain(recv.iter()) {
            let line: Vec<String> = list.iter().map(|x| x.to_string()).collect();
            let _ = writeln!(instances_txt, "{}", line.join(" "));
        }
        instances.push((prop, recv));
    }
    fs::write("bench/instances.txt", &instances_txt).expect("write instances");

    // --- Time Rust Gale-Shapley over all instances and record the matchings. ---
    let start = Instant::now();
    let mut matchings = Vec::with_capacity(M);
    for (prop, recv) in &instances {
        let m = gale_shapley(prop, recv);
        matchings.push(m);
    }
    let gs_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut out = format!("{gs_ms:.3}\n");
    for m in &matchings {
        let line: Vec<String> = m
            .proposer
            .iter()
            .map(|x| x.map_or(-1i64, |r| r as i64).to_string())
            .collect();
        let _ = writeln!(out, "{}", line.join(" "));
    }
    fs::write("bench/rust_matching.txt", &out).expect("write rust_matching");

    // --- Time a fixed UCB1 online bandit workload. ---
    let mut learner = Ucb1::new(BANDIT_ARMS, 0.5);
    let true_means: Vec<f64> = (0..BANDIT_ARMS)
        .map(|a| a as f64 / BANDIT_ARMS as f64)
        .collect();
    let mut env = Rng::new(7);
    let start = Instant::now();
    for _ in 0..BANDIT_ROUNDS {
        let arm = learner.ranking()[0];
        let reward = env.normal(true_means[arm], 0.3);
        learner.update(arm, reward);
    }
    let bandit_ms = start.elapsed().as_secs_f64() * 1000.0;
    fs::write(
        "bench/rust_bandit.txt",
        format!("{BANDIT_ARMS} {BANDIT_ROUNDS} {bandit_ms:.3}\n"),
    )
    .expect("write rust_bandit");

    // --- Time the full integrated learn -> match -> reward -> update loop. ---
    let mut g = Rng::new(2026);
    let (util_p, util_r) = correlated_market(&mut g, INTEGRATED_N, INTEGRATED_N, 0.5);
    let receiver_prefs = prefs_from_util(&util_r);
    let mut market = Market::with_thompson(util_p, receiver_prefs, 0.5, 1.0, 0.04, 0.2, 7);
    let start = Instant::now();
    let report = simulate(&mut market, INTEGRATED_ROUNDS);
    let integrated_ms = start.elapsed().as_secs_f64() * 1000.0;
    let tail_stable = report.tail_stable_fraction(INTEGRATED_ROUNDS / 5);
    fs::write(
        "bench/rust_integrated.txt",
        format!("{INTEGRATED_N} {INTEGRATED_ROUNDS} {integrated_ms:.3} {tail_stable:.4}\n"),
    )
    .expect("write rust_integrated");

    println!("Wrote bench/instances.txt ({M} instances of size {N})");
    println!("Rust Gale-Shapley: {gs_ms:.1} ms for {M} instances");
    println!("Rust UCB1 bandit : {bandit_ms:.1} ms for {BANDIT_ROUNDS} rounds");
    println!(
        "Rust integrated  : {integrated_ms:.1} ms for {INTEGRATED_ROUNDS} rounds (tail stable {tail_stable:.3})"
    );
}
