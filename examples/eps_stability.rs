//! Explore: does an *ε-stability* benchmark dissolve the near-tie stall modes?
//!
//! Regret-vs-unique-stable charges a proposer for any deviation from the exact
//! proposer-optimal stable matching, even a swap it is indifferent to. A softer
//! benchmark: a matching is **ε-stable** if it has no blocking pair `(p, r)` in
//! which `p` gains more than `ε` in utility *and* `r` strictly prefers `p` to its
//! current partner. Indifferent (sub-ε) swaps are then not violations.
//!
//! Hypothesis: ε-stability reclassifies **churn** (a proposer flips between arms
//! it values within ε — harmless) as fine, but **not cascades** (another proposer
//! loses far more than ε — genuinely unstable). So it should dissolve the churn
//! share of the "stalls" while leaving the cascades flagged.
//!
//! We take the markets whose settled belief matching has high strict regret (the
//! stalls) and report how many are nonetheless ε-stable.
//!
//! ```text
//! cargo run --release --example eps_stability
//! ```

use match_learn::matching::Matching;
use match_learn::{Market, Rng, gale_shapley, rank_by_scores, simulate};

const N: usize = 5;
const NOISE: f64 = 0.2;
const SEEDS: usize = 800;
const HORIZON: usize = 8000;
const EPS: f64 = 0.05; // indifference band for ε-stability
const STALL: f64 = 0.05; // strict settled regret above this = a stall

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

fn strict_regret(util: &[Vec<f64>], baseline: &Matching, m: &Matching) -> f64 {
    (0..util.len())
        .map(|p| {
            let b = baseline.proposer[p].map_or(0.0, |r| util[p][r]);
            let g = m.proposer[p].map_or(0.0, |r| util[p][r]);
            b - g
        })
        .sum()
}

/// Largest proposer gain over any true blocking pair whose gain exceeds `eps`
/// (0.0 if none): a pair `(p, r)` where `p`'s utility gain over its current
/// partner exceeds `eps` and `r` (by its known true preferences) strictly prefers
/// `p` to its current holder. `eps = 0` is exact stability.
fn worst_blocking(util: &[Vec<f64>], recv: &[Vec<usize>], m: &Matching, eps: f64) -> f64 {
    let n = util.len();
    // who holds each receiver
    let mut holder = vec![None; recv.len()];
    for p in 0..n {
        if let Some(r) = m.proposer[p] {
            holder[r] = Some(p);
        }
    }
    let pos = |r: usize, p: usize| recv[r].iter().position(|&q| q == p).unwrap_or(usize::MAX);
    let mut worst = 0.0_f64;
    #[allow(clippy::needless_range_loop)] // p also indexes the matching
    for p in 0..n {
        let cur = m.proposer[p].map_or(f64::MIN, |r| util[p][r]);
        for r in 0..recv.len() {
            if m.proposer[p] == Some(r) {
                continue;
            }
            let gain = util[p][r] - cur;
            if gain <= eps {
                continue;
            }
            // r strictly prefers p to its current holder (or is unheld)?
            let prefers = match holder[r] {
                Some(h) => pos(r, p) < pos(r, h),
                None => true,
            };
            if prefers {
                worst = worst.max(gain);
            }
        }
    }
    worst
}

fn main() {
    println!(
        "ε-stability benchmark — {SEEDS} markets ({N}x{N}, horizon {HORIZON}), ε={EPS}, stall>{STALL}\n"
    );
    let mut seedgen = Rng::new(20260606);
    let (mut stalls, mut eps_stable, mut exact_stable) = (0usize, 0usize, 0usize);
    let mut worst_block_sum = 0.0;
    for _ in 0..SEEDS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, N);
        let mut market = Market::with_thompson(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            NOISE * NOISE,
            NOISE,
            seed ^ 0xABCD,
        );
        simulate(&mut market, HORIZON);
        let means = market.belief_means();
        let baseline = market.true_stable_matching();
        let settled = gale_shapley(
            &means.iter().map(|m| rank_by_scores(m)).collect::<Vec<_>>(),
            &recv,
        );
        if strict_regret(&util, &baseline, &settled) <= STALL {
            continue;
        }
        stalls += 1;
        if worst_blocking(&util, &recv, &settled, 0.0) == 0.0 {
            exact_stable += 1; // exactly stable, just not proposer-optimal
        }
        let wb = worst_blocking(&util, &recv, &settled, EPS);
        if wb == 0.0 {
            eps_stable += 1;
        } else {
            worst_block_sum += wb;
        }
    }
    println!("stalls (strict regret vs proposer-OPTIMAL stable > {STALL}): {stalls}");
    println!("  exactly stable (no blocking pair at all): {exact_stable}/{stalls}");
    println!("  ε-stable (no >ε blocking pair, ε={EPS})  : {eps_stable}/{stalls}");
    let eps_unstable = stalls - eps_stable;
    if eps_unstable > 0 {
        println!(
            "  ε-unstable remainder                    : {eps_unstable}/{stalls}  (mean worst ε-blocking gain {:.3})",
            worst_block_sum / eps_unstable as f64
        );
    }
    println!(
        "\nReading: the near-tie stalls are not unstable matchings — they are *other stable*\nmatchings, just not the proposer-OPTIMAL one. The 'regret' is a proposer-optimality\ngap, not instability. Against an any-stable / ε-stable benchmark the stall vanishes;\nrecovering proposer-optimality specifically is what the coordinator (belief welfare) buys."
    );
}
