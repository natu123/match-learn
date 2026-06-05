//! Cross-check the empirical regret growth against theory.
//!
//! Matching-bandit theory says a learner that stabilizes incurs *sublinear*
//! cumulative regret: `R(T) = o(T)`, growing like `O(sqrt T)` or `O(log T)`
//! rather than linearly. A clean, always-non-negative proxy for the same claim
//! is the cumulative number of rounds whose matching is *unstable* in the true
//! market, `U(T)`: if the market settles, `U(T)` grows sublinearly; a policy
//! that never learns leaves a constant per-round instability, so `U(T)` grows
//! linearly (slope 1 on a log-log plot).
//!
//! We run several horizons, fit the slope of `ln U` against `ln T`, and require
//! it to be clearly below 1 (sublinear) — while the no-learning baseline sits at
//! slope ≈ 1.

use match_learn::data::{correlated_market, prefs_from_util};
use match_learn::matching::{gale_shapley, is_stable};
use match_learn::{Market, Rng, simulate};

/// Least-squares slope of `y` on `x`.
fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        num += (x - mx) * (y - my);
        den += (x - mx) * (x - mx);
    }
    num / den
}

#[test]
fn unstable_count_grows_sublinearly() {
    let horizons = [1000usize, 2000, 4000, 8000];
    let max_t = *horizons.last().unwrap();
    let markets = 16;
    let n = 5;
    let noise = 0.2;

    // U(T) for the learner, and for a no-learning constant policy, averaged.
    let mut learn_u = vec![0.0f64; horizons.len()];
    let mut base_u = vec![0.0f64; horizons.len()];

    let mut seedgen = Rng::new(31337);
    for _ in 0..markets {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut g = Rng::new(seed);
        // Realistic, partly-correlated market.
        let (util_p, util_r) = correlated_market(&mut g, n, n, 0.5);
        let true_pp = prefs_from_util(&util_p);
        let true_rp = prefs_from_util(&util_r);

        // Receiver side is known here (one-sided unknown), so use util_r's
        // induced rankings as the fixed receiver preferences.
        let mut market = Market::with_thompson(
            util_p.clone(),
            true_rp.clone(),
            0.5,
            1.0,
            noise * noise,
            noise,
            seed ^ 0xABCD,
        );
        let rep = simulate(&mut market, max_t);

        // Prefix counts of unstable rounds at each horizon.
        for (h, &t) in horizons.iter().enumerate() {
            let unstable = rep.stable[..t].iter().filter(|&&s| !s).count();
            learn_u[h] += unstable as f64;
        }

        // No-learning baseline: proposers fixed at index order -> constant
        // matching; it is unstable in the true market for all T rounds, or none.
        let fixed: Vec<Vec<usize>> = (0..n).map(|_| (0..n).collect()).collect();
        let played = gale_shapley(&fixed, &true_rp);
        let unstable_round = !is_stable(&true_pp, &true_rp, &played);
        for (h, &t) in horizons.iter().enumerate() {
            base_u[h] += if unstable_round { t as f64 } else { 0.0 };
        }
    }

    // Average and guard against zeros before logging.
    let ln_t: Vec<f64> = horizons.iter().map(|&t| (t as f64).ln()).collect();
    let ln_learn: Vec<f64> = learn_u
        .iter()
        .map(|&u| (u / markets as f64).max(1.0).ln())
        .collect();
    let ln_base: Vec<f64> = base_u
        .iter()
        .map(|&u| (u / markets as f64).max(1.0).ln())
        .collect();

    let learn_slope = slope(&ln_t, &ln_learn);
    let base_slope = slope(&ln_t, &ln_base);

    println!("--- regret/instability scaling ({markets} correlated {n}x{n} markets) ---");
    for (h, &t) in horizons.iter().enumerate() {
        println!(
            "T={t:>5}  learn U={:>8.1}  base U={:>8.1}",
            learn_u[h] / markets as f64,
            base_u[h] / markets as f64
        );
    }
    println!("learn ln-ln slope = {learn_slope:.3}  (sublinear < 1)");
    println!("base  ln-ln slope = {base_slope:.3}  (linear ~ 1)");

    // The learner's instability grows clearly sublinearly...
    assert!(
        learn_slope < 0.85,
        "learner instability slope {learn_slope} is not clearly sublinear"
    );
    // ...while the no-learning baseline grows linearly (slope ~1).
    assert!(
        base_slope > 0.95,
        "no-learning baseline slope {base_slope} should be ~1 (linear)"
    );
}
