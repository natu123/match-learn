//! A hand-designed 4×4 market with a **net** Θ(1) cascade floor
//! (`docs/theory-identifiability.md`, Prop. 2′). It witnesses the case the 3×3
//! `cascade_lower_bound.rs` could not: one near-indifferent proposer's mis-order
//! makes *every* proposer strictly worse off (no beneficiary), so the net
//! proposer regret — not just the victim's — is bounded below by a constant.
//!
//! Proposers p,q,r,s (indices 0..4) over receivers A,B,C,D (0..4).
//! True utilities (A,B near-tie for p with gap Δ_p):
//!   p: A=1.00  B=1.00−Δ_p  C=0.10  D=0.00     order A≻B≻C≻D
//!   q: B=0.90  C=0.40      D=0.05  A=0.00     order B≻C≻D≻A
//!   r: C=0.80  D=0.50      B=0.10  A=0.00     order C≻D≻B≻A
//!   s: D=0.70  A=0.30      B=0.20  C=0.10     order D≻A≻B≻C
//! Receiver prefs (known, exact):
//!   A: p≻q≻r≻s   B: p≻q≻r≻s   C: q≻r≻p≻s   D: r≻s≻p≻q
//!
//! In M* every proposer gets its rank-1 receiver: M* = {p-A, q-B, r-C, s-D}, so
//! M* is trivially the proposer-optimal stable matching. If p mis-orders its
//! near-tie pair (reports B≻A), Gale-Shapley produces a pure rejection chain
//! p→B (kicks q) → q→C (kicks r) → r→D (kicks s) → s→A, landing on
//! M' = {p-B, q-C, r-D, s-A}. Every proposer drops one rank:
//!   p: A→B (−Δ_p), q: B→C (−0.50), r: C→D (−0.30), s: D→A (−0.40).
//! Net regret = Δ_p + 1.20 → 1.20 = Θ(1) as Δ_p→0. No proposer gains, because the
//! freed receiver A is a *downgrade* even for the proposer (s) that ends up taking
//! it — the structural difference from the 3×3 redistribution.
//!
//! This runs Gale-Shapley directly on the two profiles (no learning needed): it
//! verifies the two hand-computed matchings and the net-regret arithmetic.
//!
//! ```text
//! cargo run --release --example net_floor_4x4
//! ```

use match_learn::gale_shapley;
use match_learn::matching::Matching;

const DP: f64 = 0.01; // p's near-tie gap (Δ_p); the proof takes Δ_p→0

fn util() -> Vec<Vec<f64>> {
    vec![
        vec![1.00, 1.00 - DP, 0.10, 0.00], // p
        vec![0.00, 0.90, 0.40, 0.05],      // q
        vec![0.00, 0.10, 0.80, 0.50],      // r
        vec![0.30, 0.20, 0.10, 0.70],      // s
    ]
}

fn receivers() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 2, 3], // A: p≻q≻r≻s
        vec![0, 1, 2, 3], // B: p≻q≻r≻s
        vec![1, 2, 0, 3], // C: q≻r≻p≻s
        vec![2, 3, 0, 1], // D: r≻s≻p≻q
    ]
}

/// Descending-by-utility ranking with index tie-break (true belief-free order).
fn true_rankings(util: &[Vec<f64>]) -> Vec<Vec<usize>> {
    util.iter()
        .map(|u| {
            let mut idx: Vec<usize> = (0..u.len()).collect();
            idx.sort_by(|&a, &b| u[b].partial_cmp(&u[a]).unwrap());
            idx
        })
        .collect()
}

fn proposer_regret(util: &[Vec<f64>], star: &Matching, m: &Matching) -> Vec<f64> {
    (0..util.len())
        .map(|p| {
            let b = star.proposer[p].map_or(0.0, |r| util[p][r]);
            let g = m.proposer[p].map_or(0.0, |r| util[p][r]);
            b - g
        })
        .collect()
}

fn fmt(m: &Matching, n: usize) -> String {
    let names = ["A", "B", "C", "D"];
    (0..n)
        .map(|p| format!("p{p}->{}", m.proposer[p].map_or("∅", |r| names[r])))
        .collect::<Vec<_>>()
        .join("  ")
}

fn main() {
    let util = util();
    let recv = receivers();
    let n = util.len();

    // True profile -> M* (proposer-optimal stable matching).
    let truth = true_rankings(&util);
    let star = gale_shapley(&truth, &recv);

    // p mis-orders its near-tie pair: report B(1) before A(0); others unchanged.
    let mut misreport = truth.clone();
    misreport[0] = vec![1, 0, 2, 3];
    let m_prime = gale_shapley(&misreport, &recv);

    println!("Net-floor 4×4 (Δ_p = {DP})\n");
    println!("M*  (true)      : {}", fmt(&star, n));
    println!("M'  (p mis-orders): {}", fmt(&m_prime, n));

    // Verify the hand-computed matchings.
    let star_ok = star.proposer == vec![Some(0), Some(1), Some(2), Some(3)];
    let mp_ok = m_prime.proposer == vec![Some(1), Some(2), Some(3), Some(0)];
    println!(
        "\nhand-proof check: M* {} {{p-A,q-B,r-C,s-D}},  M' {} {{p-B,q-C,r-D,s-A}}",
        if star_ok { "==" } else { "!=" },
        if mp_ok { "==" } else { "!=" },
    );
    assert!(star_ok, "M* mismatch");
    assert!(mp_ok, "M' mismatch");

    // Per-proposer and net regret of M' against M*.
    let reg = proposer_regret(&util, &star, &m_prime);
    let names = ["p", "q", "r", "s"];
    println!("\nper-proposer regret of M' vs M*:");
    for i in 0..n {
        println!("  {}: {:+.3}", names[i], reg[i]);
    }
    let net: f64 = reg.iter().sum();
    let min_loss = reg.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("\nnet proposer regret = {net:.3}   (min individual = {min_loss:+.3})");
    println!(
        "Every proposer is weakly worse (min loss ≥ 0): M' is M*-dominated, so the\n\
         cascade is NOT a redistribution — the net floor is {net:.2} = Θ(1), driven by\n\
         p's unresolvable Δ_p near-tie. Compare cascade_lower_bound.rs, where one\n\
         proposer gained and the net nearly cancelled."
    );
    assert!(
        min_loss >= -1e-12,
        "some proposer gained — not M*-dominated"
    );
    assert!(net > 1.0, "net floor should be ~1.20");
}
