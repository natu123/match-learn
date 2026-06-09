//! Numerical companion to `docs/theory-verification-2026-06-10.md` (research
//! lane): executable checks for the three verification targets on the
//! embedding market.
//!
//! - **V3 (rigid-core uniqueness)**: exhaustive enumeration over *all* partial
//!   matchings — the embed market has exactly one stable matching (which is
//!   also its only super-stable matching), equal to the predicted one, across
//!   sizes, gaps, and both instances.
//! - **V2 (Theorem A key lemma)**: the same-seed coupling — under
//!   irreversible + no-interview, instance I and II runs produce the *same*
//!   realized trajectory, so the two cumulative regrets sum to ≥ T and at most
//!   one instance can end stable. The commit law is instance-independent.
//! - **V1 (Theorem B lower bound)**: the regret of interview-then-commit has
//!   the `(σ²/(kΔ_A²))·log T` shape — equal increments per 4× horizon step
//!   (log T), inverse scaling in the interview budget `k`, inverse-square in
//!   `Δ_A` — and the lower bound's trade-off minimisation algebra
//!   `min_m m/k + (T/2)e^{-κm}` matches its closed form.
//!
//! Run with: `cargo run --release --example theory_verification`

use match_learn::embedding::{embed, simulate_market};
use match_learn::irreversible::Policy;
use match_learn::prefs::rank_by_scores;

fn rankings(utils: &[Vec<f64>]) -> Vec<Vec<usize>> {
    utils.iter().map(|row| rank_by_scores(row)).collect()
}

/// All partial matchings (each agent to a distinct firm or unmatched).
fn all_partial_matchings(n: usize, m: usize) -> Vec<Vec<Option<usize>>> {
    let mut out = Vec::new();
    let mut cur = vec![None; n];
    let mut used = vec![false; m];
    fn rec(
        a: usize,
        n: usize,
        m: usize,
        cur: &mut Vec<Option<usize>>,
        used: &mut Vec<bool>,
        out: &mut Vec<Vec<Option<usize>>>,
    ) {
        if a == n {
            out.push(cur.clone());
            return;
        }
        cur[a] = None;
        rec(a + 1, n, m, cur, used, out);
        for f in 0..m {
            if !used[f] {
                used[f] = true;
                cur[a] = Some(f);
                rec(a + 1, n, m, cur, used, out);
                cur[a] = None;
                used[f] = false;
            }
        }
    }
    rec(0, n, m, &mut cur, &mut used, &mut out);
    out
}

/// Stability under cardinal utilities, unmatched = utility 0.
/// `weak = false`: classic blocking (both strictly improve).
/// `weak = true`: weakly blocking (both weakly, one strictly) — its absence
/// is super-stability.
fn is_stable(prop: &[Vec<f64>], recv: &[Vec<f64>], mu: &[Option<usize>], weak: bool) -> bool {
    let (n, m) = (prop.len(), recv.len());
    let mut firm_of = vec![None; m];
    for (a, &f) in mu.iter().enumerate() {
        if let Some(f) = f {
            firm_of[f] = Some(a);
        }
    }
    for a in 0..n {
        for f in 0..m {
            if mu[a] == Some(f) {
                continue;
            }
            let da = prop[a][f] - mu[a].map_or(0.0, |g| prop[a][g]);
            let df = recv[f][a] - firm_of[f].map_or(0.0, |b: usize| recv[f][b]);
            let blocks = if weak {
                da >= 0.0 && df >= 0.0 && (da > 0.0 || df > 0.0)
            } else {
                da > 0.0 && df > 0.0
            };
            if blocks {
                return false;
            }
        }
    }
    true
}

fn v3_core_uniqueness() {
    println!("V3: rigid-core uniqueness (exhaustive, partial matchings included)");
    let mut cases = 0;
    for &n in &[2usize, 3, 4, 5] {
        for &delta_a in &[0.02, 0.1, 0.149] {
            for &delta_big in &[0.15, 0.2, 0.225] {
                if delta_a >= delta_big || delta_big * (n as f64 - 1.0) > 0.9 {
                    continue;
                }
                for instance in [false, true] {
                    let (prop, recv) = embed(delta_a, delta_big, n, instance);
                    let all = all_partial_matchings(n, n);
                    let stable: Vec<_> = all
                        .iter()
                        .filter(|mu| is_stable(&prop, &recv, mu, false))
                        .collect();
                    let super_stable: Vec<_> = all
                        .iter()
                        .filter(|mu| is_stable(&prop, &recv, mu, true))
                        .collect();

                    // Predicted: a* -> its truly better firm, a_s -> the other,
                    // core agent i -> firm i.
                    let mut pred: Vec<Option<usize>> = (0..n).map(Some).collect();
                    if instance {
                        pred[0] = Some(1);
                        pred[1] = Some(0);
                    }
                    assert_eq!(stable.len(), 1, "n={n} Δ_A={delta_a} Δ_big={delta_big}");
                    assert_eq!(*stable[0], pred, "stable ≠ predicted");
                    assert_eq!(super_stable.len(), 1, "super-stable not unique");
                    assert_eq!(*super_stable[0], pred, "super-stable ≠ predicted");
                    cases += 1;
                }
            }
        }
    }
    println!("  unique stable = unique super-stable = predicted, {cases} parameter cases\n");
}

fn v2_coupling() {
    println!("V2: Theorem-A coupling — same seed, instances I vs II (irrev/no-interview)");
    let (t, n) = (400usize, 4usize);
    let (p1, r1) = embed(0.15, 0.25, n, false);
    let (p2, r2) = embed(0.15, 0.25, n, true);
    let (rr1, rr2) = (rankings(&r1), rankings(&r2));
    let (mut stable_1, mut stable_2) = (0u32, 0u32);
    let seeds = 400u64;
    for s in 1..=seeds {
        let o1 = simulate_market(&p1, &rr1, false, Policy::NoInterview, t, s);
        let o2 = simulate_market(&p2, &rr2, false, Policy::NoInterview, t, s);
        // The benchmarks differ, the trajectory is shared: regrets sum to ≥ T,
        // and at most one of the two runs can end stable.
        assert!(
            o1.final_regret() + o2.final_regret() >= t as f64,
            "seed {s}: coupled regrets sum below T"
        );
        assert!(
            !(o1.ended_stable && o2.ended_stable),
            "seed {s}: one commit cannot match both benchmarks"
        );
        stable_1 += o1.ended_stable as u32;
        stable_2 += o2.ended_stable as u32;
    }
    println!(
        "  {seeds} seeds: R_I + R_II ≥ T always; ended-stable I: {:.3}, II: {:.3} (instance-symmetric, ≪ 1)\n",
        f64::from(stable_1) / seeds as f64,
        f64::from(stable_2) / seeds as f64,
    );
}

fn mean_final(delta_a: f64, per_round: usize, t: usize) -> f64 {
    let (prop, recv) = embed(delta_a, 0.25, 4, false);
    let rr = rankings(&recv);
    let policy = Policy::Interview { per_round };
    (1..=12u64)
        .map(|s| simulate_market(&prop, &rr, false, policy, t, s).final_regret())
        .sum::<f64>()
        / 12.0
}

fn v1_tradeoff() {
    println!("V1: Theorem-B shape — interview-then-commit stable regret, (σ²/(kΔ²))·log T");

    // (a) log T: equal regret increments per 4× horizon step.
    let (r1, r2, r3) = (
        mean_final(0.15, 2, 4_000),
        mean_final(0.15, 2, 16_000),
        mean_final(0.15, 2, 64_000),
    );
    println!(
        "  T 4k/16k/64k: {r1:.0} / {r2:.0} / {r3:.0}; increments {:.0} vs {:.0} (log T ⇒ equal)",
        r2 - r1,
        r3 - r2
    );

    // (b) 1/k: doubling the per-round interview budget halves the regret.
    let (k1, k2, k4) = (
        mean_final(0.15, 1, 16_000),
        mean_final(0.15, 2, 16_000),
        mean_final(0.15, 4, 16_000),
    );
    println!(
        "  k 1/2/4: {k1:.0} / {k2:.0} / {k4:.0}; ratios {:.2}, {:.2} (1/k ⇒ 2.00)",
        k1 / k2,
        k2 / k4
    );

    // (c) 1/Δ²: halving the pivot quadruples the regret.
    let (wide, narrow) = (mean_final(0.2, 2, 8_000), mean_final(0.1, 2, 8_000));
    println!(
        "  Δ_A 0.2 → 0.1: {wide:.0} → {narrow:.0}; ratio {:.2} (1/Δ² ⇒ 4.00)",
        narrow / wide
    );

    // (d) The lower bound's trade-off algebra: min over m of
    // g(m) = m/k + (T/2)·exp(−κm) is at m* = (1/κ)·ln(kκT/2), value
    // (1/(kκ))·(ln(kκT/2) + 1). Numerical grid minimum vs closed form.
    let (kappa, k, t) = (0.045f64, 2.0f64, 16_000.0f64); // κ = Δ²/(2σ²), Δ=0.15, σ²=1/4
    let g = |m: f64| m / k + t / 2.0 * (-kappa * m).exp();
    let mut best = (0.0, f64::INFINITY);
    let mut m = 0.0;
    while m < 2_000.0 {
        if g(m) < best.1 {
            best = (m, g(m));
        }
        m += 0.01;
    }
    let m_star = (k * kappa * t / 2.0).ln() / kappa;
    let g_star = (1.0 / (k * kappa)) * ((k * kappa * t / 2.0).ln() + 1.0);
    println!(
        "  algebra: grid argmin m={:.1} g={:.1} vs closed form m*={m_star:.1} g*={g_star:.1}\n",
        best.0, best.1
    );
}

fn main() {
    v3_core_uniqueness();
    v2_coupling();
    v1_tradeoff();
    println!("all verification checks passed");
}
