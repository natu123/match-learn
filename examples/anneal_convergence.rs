//! Numerical corroboration for the annealing-convergence theory
//! (`docs/theory-annealing.md`), at the sampling level where the mechanism is
//! clean.
//!
//! Near-tie *churn* is the case where a proposer's belief **means are already
//! correct** (so the mean/index matching is right) but one arm's posterior is
//! **wide and frozen** (an unmatched competitor that stopped being pulled), so
//! Thompson keeps re-sampling the two arms in different orders. We model exactly
//! that: fix two posteriors `a ~ N(m_a, s)` and `b ~ N(m_b, s)` with the correct
//! order `m_a > m_b` at a near-tie gap `δ = m_a − m_b ≪ s`, and each round draw
//! Thompson samples with std scaled by `α_t`. A "churn round" is one where `b`
//! samples above `a` (the order flips away from the truth).
//!
//! Theory: per-round churn probability is `Φ(−δ / (α_t · s√2))`. With no annealing
//! (`α_t = 1`) this is a constant, so cumulative churn is linear and the tail rate
//! is flat. Under `α_t = sqrt(tau/(tau+t))` the probability → 0, so cumulative
//! churn converges and the tail rate collapses. This isolates the sampling effect
//! from learning dynamics; the market-level payoff is in `anneal_study.rs`.
//!
//! ```text
//! cargo run --release --example anneal_convergence
//! ```

use match_learn::Rng;

const M_A: f64 = 0.50;
const M_B: f64 = 0.49; // correct order, near-tie gap δ = 0.01
const S: f64 = 0.10; // wide frozen posterior std
const HORIZON: usize = 40000;
const TAIL: usize = HORIZON / 10;

fn alpha(tau: Option<f64>, t: usize) -> f64 {
    match tau {
        None => 1.0,
        Some(tau) => (tau / (tau + t as f64)).sqrt(),
    }
}

/// (cumulative churn rounds, tail-window churn rounds) over the horizon.
fn churn(tau: Option<f64>, seed: u64) -> (usize, usize) {
    let mut rng = Rng::new(seed);
    let (mut cumulative, mut tail) = (0usize, 0usize);
    for t in 0..HORIZON {
        let a = alpha(tau, t);
        let xa = rng.normal(M_A, a * S);
        let xb = rng.normal(M_B, a * S);
        if xb > xa {
            cumulative += 1;
            if t >= HORIZON - TAIL {
                tail += 1;
            }
        }
    }
    (cumulative, tail)
}

fn avg(tau: Option<f64>) -> (f64, f64) {
    let seeds = 16;
    let (mut c, mut t) = (0.0, 0.0);
    for s in 0..seeds {
        let (cc, tt) = churn(tau, 1 + s as u64);
        c += cc as f64;
        t += tt as f64;
    }
    (c / seeds as f64, t / seeds as f64)
}

/// Standard normal CDF via the Abramowitz-Stegun 7.1.26 erf approximation.
fn phi(x: f64) -> f64 {
    let z = x / std::f64::consts::SQRT_2;
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let z = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-z * z).exp();
    0.5 * (1.0 + sign * y)
}

/// Predicted tail churn probability at the end of the horizon: Φ(−δ/(α_T·s·√2)).
fn predicted_tail(tau: Option<f64>) -> f64 {
    let a = alpha(tau, HORIZON);
    phi(-(M_A - M_B) / (a * S * std::f64::consts::SQRT_2))
}

fn main() {
    println!(
        "Annealing convergence (sampling level) — frozen near-tie posteriors m_a={M_A}, m_b={M_B} (δ={:.2}), s={S}, horizon {HORIZON}\n",
        M_A - M_B
    );
    println!("  schedule          cumulative churn   tail churn-rate   theory Φ(−δ/(α_T·s√2))");
    let report = |name: &str, tau: Option<f64>| {
        let (c, t) = avg(tau);
        println!(
            "  {name:<16}  {c:>16.1}   {:>14.6}   {:>14.6}",
            t / TAIL as f64,
            predicted_tail(tau),
        );
    };
    report("plain (tau=inf)", None);
    for tau in [8000.0, 2000.0, 500.0, 100.0, 20.0] {
        report(&format!("anneal tau={tau:.0}"), Some(tau));
    }
    println!(
        "\nTheory: per-round churn prob = Φ(−δ/(α_t·s√2)) (empirical tail matches the prediction).\nPlain (α=1) holds it constant → linear cumulative churn, flat tail. Cooling collapses it\nonce α_t·s < δ; the cost of fast cooling (small tau) is the lock-in risk (theory-annealing.md)."
    );
}
