//! A concrete cascade instance for the lower-bound theory
//! (`docs/theory-identifiability.md`, Prop. 2): a parametric 3×3 market where one
//! near-indifferent proposer's order swings another proposer's outcome by `Δ_q`.
//!
//! Proposers p0,p1,p2 over receivers ra,rb,rc (arm indices 0,1,2):
//! - p0 is near-indifferent between ra,rb (gap `Δ_p`), dislikes rc.
//! - p1 wants rb, falls back to rc (gap `Δ_q`); ra is its worst.
//! - p2 wants ra.
//! Receiver prefs (known): ra likes p0>p2>p1, rb likes p0>p1>p2.
//!
//! True optimal stable matching M* = {p0-ra, p1-rb, p2-rc}. If p0 (mis)ranks rb
//! first, Gale-Shapley gives {p0-rb, p1-rc, p2-ra}: the **victim p1** drops rb→rc
//! (loses Δ_q), while p2 rises rc→ra (gains). So the *net* proposer regret nearly
//! cancels (a redistribution), but the **victim's individual** regret is Θ(Δ_q).
//!
//! This measures: cumulative victim regret and cumulative net regret over the
//! horizon, for a sub-noise `Δ_p` (unresolvable → linear victim regret) vs a
//! resolvable `Δ_p` (sublinear). It corroborates that the cascade floor is an
//! *individual* floor, and connects to the ε-stability reframing (the wrong
//! matching is ε-stable: its only blocking pair, (p0,ra), has gain Δ_p ≪ ε).
//!
//! ```text
//! cargo run --release --example cascade_lower_bound
//! ```

use match_learn::Market;

const NOISE: f64 = 0.2;
const DQ: f64 = 0.6; // victim's rb→rc drop

fn instance(dp: f64) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let util_p = vec![
        vec![0.80, 0.80 - dp, 0.00], // p0: ra ≈ rb (gap dp), rc bad
        vec![0.05, 0.70, 0.70 - DQ], // p1: rb > rc (gap DQ), ra worst
        vec![0.90, 0.50, 0.30],      // p2: ra > rb > rc
    ];
    let recv = vec![
        vec![0, 2, 1], // ra: p0 > p2 > p1
        vec![0, 1, 2], // rb: p0 > p1 > p2
        vec![1, 2, 0], // rc: p1 > p2 > p0
    ];
    (util_p, recv)
}

/// (cumulative victim regret, cumulative net regret, fraction of rounds p0
/// mis-ranks rb above ra) over `rounds`, under greedy Thompson.
fn run(dp: f64, rounds: usize, seed: u64) -> (f64, f64, f64) {
    let (util_p, recv) = instance(dp);
    let mut market = Market::with_thompson(
        util_p.clone(),
        recv.clone(),
        0.5,
        1.0,
        NOISE * NOISE,
        NOISE,
        seed ^ 0xABCD,
    );
    let m_star = market.true_stable_matching();
    let star_u: Vec<f64> = (0..3)
        .map(|p| m_star.proposer[p].map_or(0.0, |r| util_p[p][r]))
        .collect();

    let mut victim = 0.0;
    let mut net = 0.0;
    let mut wrong = 0usize;
    for _ in 0..rounds {
        // record p0's belief order before the step
        let means = market.belief_means();
        if means[0][1] > means[0][0] {
            wrong += 1; // ranks rb (1) above ra (0)
        }
        let m = market.step();
        // victim is proposer 1; net is summed over all proposers
        let got1 = m.proposer[1].map_or(0.0, |r| util_p[1][r]);
        victim += star_u[1] - got1;
        for p in 0..3 {
            net += star_u[p] - m.proposer[p].map_or(0.0, |r| util_p[p][r]);
        }
    }
    (victim, net, wrong as f64 / rounds as f64)
}

fn avg(dp: f64, rounds: usize) -> (f64, f64, f64) {
    let seeds = 16;
    let (mut v, mut n, mut w) = (0.0, 0.0, 0.0);
    for s in 0..seeds {
        let (vv, nn, ww) = run(dp, rounds, 1 + s as u64);
        v += vv;
        n += nn;
        w += ww;
    }
    (v / seeds as f64, n / seeds as f64, w / seeds as f64)
}

fn main() {
    println!("Cascade lower-bound instance (3x3, Δ_q={DQ}, noise={NOISE})\n");
    for &dp in &[0.005, 0.15] {
        // Resolving the pair needs ~σ²/Δ_p² pulls of each arm (Lemma 1); compare
        // that to the horizon to judge whether p0 can learn its order in time.
        let n_resolve = (NOISE * NOISE) / (dp * dp);
        println!("Δ_p = {dp}  [needs ~{n_resolve:.0} pulls/arm to resolve]");
        println!("  rounds   victim regret   (per-round)   net regret   p0 wrong-order frac");
        for &t in &[2000usize, 8000, 32000] {
            let (v, n, w) = avg(dp, t);
            println!(
                "  {t:>6}   {v:>10.2}   {:>10.5}   {n:>10.2}   {w:>6.3}",
                v / t as f64
            );
        }
        println!();
    }
    println!(
        "Reading: for sub-noise Δ_p the victim's regret grows ~linearly (per-round rate stays\nbounded away from 0) — the individual cascade floor — while the *net* regret stays small\n(p2's gain offsets p1's loss). For resolvable Δ_p both fall off as p0 learns its order."
    );
}
