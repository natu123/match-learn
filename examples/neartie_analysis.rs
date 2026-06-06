//! Near-tie analysis: confirm, across many seeds, that genuine long-horizon
//! lock-ins are driven by *near-ties* in the true preferences — adjacent
//! receivers a proposer values almost equally, at a gap far below the noise floor
//! — rather than by frozen arms.
//!
//! For each random market we measure its tightest decision: the smallest gap, over
//! all proposers, between two receivers that are adjacent in that proposer's true
//! preference order. We then run greedy Thompson Sampling to a long horizon and
//! record whether the market is still paying regret in its tail. If small gaps
//! predict stalls, the lock-in is an identifiability limit (gap << noise), not a
//! learning failure.
//!
//! ```text
//! cargo run --release --example neartie_analysis
//! ```

use match_learn::{Market, Rng, gale_shapley, rank_by_scores, simulate};

const N: usize = 5;
const NOISE: f64 = 0.2;
const SEEDS: usize = 800;
const HORIZON: usize = 24000;
/// Tail mean regret rate above this counts the market as stalled.
const STALL_THRESHOLD: f64 = 0.05;

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

/// The smallest gap between two receivers adjacent in some proposer's true
/// preference order — the market's tightest "which do I prefer?" decision.
fn min_adjacent_gap(util: &[Vec<f64>]) -> f64 {
    let mut min_gap = f64::INFINITY;
    for row in util {
        let mut sorted = row.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for w in sorted.windows(2) {
            min_gap = min_gap.min(w[0] - w[1]);
        }
    }
    min_gap
}

fn main() {
    println!(
        "Near-tie analysis — {SEEDS} random {N}x{N} markets, Thompson, horizon {HORIZON}, noise {NOISE}\n"
    );

    let mut seedgen = Rng::new(20260606);
    let mut records: Vec<(f64, bool, f64)> = Vec::with_capacity(SEEDS); // (min_gap, stalled, tail_rate)
    let mut stalled_initiator_loss: Vec<f64> = Vec::new();
    for _ in 0..SEEDS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, N);
        let gap = min_adjacent_gap(&util);
        let mut market = Market::with_thompson(
            util.clone(),
            recv,
            0.5,
            1.0,
            NOISE * NOISE,
            NOISE,
            seed ^ 0xABCD,
        );
        let rep = simulate(&mut market, HORIZON);
        let tr = rep.tail_mean_regret(HORIZON / 5);
        let stalled = tr > STALL_THRESHOLD;
        records.push((gap, stalled, tr));

        if stalled {
            // The settled belief matching, and how much its *most indifferent
            // displaced* proposer actually loses by being off its true partner. A
            // near-tie cascade has a displaced proposer with ~zero own-loss: it is
            // indifferent, yet its swap cascades to hurt others.
            let stable = market.true_stable_matching();
            let rankings: Vec<Vec<usize>> = market
                .belief_means()
                .iter()
                .map(|m| rank_by_scores(m))
                .collect();
            let settled = gale_shapley(&rankings, market.receiver_prefs());
            let mut min_loss = f64::INFINITY;
            for p in 0..N {
                let (Some(star), Some(got)) = (stable.proposer[p], settled.proposer[p]) else {
                    continue;
                };
                if star != got {
                    min_loss = min_loss.min(util[p][star] - util[p][got]);
                }
            }
            if min_loss.is_finite() {
                stalled_initiator_loss.push(min_loss);
            }
        }
    }

    // Stall rate as a function of the market's tightest gap.
    let buckets = [
        (0.0, 0.005),
        (0.005, 0.02),
        (0.02, 0.05),
        (0.05, 0.1),
        (0.1, f64::INFINITY),
    ];
    println!("Stall rate by tightest true-preference gap (noise floor = {NOISE}):");
    println!("  gap bucket         markets  stalled  stall%   mean tail rate");
    for (lo, hi) in buckets {
        let in_b: Vec<&(f64, bool, f64)> = records
            .iter()
            .filter(|(g, _, _)| *g >= lo && *g < hi)
            .collect();
        if in_b.is_empty() {
            continue;
        }
        let stalled = in_b.iter().filter(|(_, s, _)| *s).count();
        let mean_tr = in_b.iter().map(|(_, _, t)| t).sum::<f64>() / in_b.len() as f64;
        let hi_s = if hi.is_infinite() {
            "inf ".to_string()
        } else {
            format!("{hi:.3}")
        };
        println!(
            "  [{lo:.3}, {hi_s})       {:>5}    {:>5}   {:>5.1}%   {:>8.4}",
            in_b.len(),
            stalled,
            100.0 * stalled as f64 / in_b.len() as f64,
            mean_tr,
        );
    }

    // Contrapositive: among the markets that stalled, how tight were their gaps?
    let mut stalled_gaps: Vec<f64> = records
        .iter()
        .filter(|(_, s, _)| *s)
        .map(|(g, _, _)| *g)
        .collect();
    let mut settled_gaps: Vec<f64> = records
        .iter()
        .filter(|(_, s, _)| !*s)
        .map(|(g, _, _)| *g)
        .collect();
    stalled_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    settled_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v[v.len() / 2]
        }
    };
    let below = |v: &[f64], t: f64| v.iter().filter(|&&g| g < t).count();
    println!(
        "\nStalled markets: {} (median tightest gap {:.4}, {} of them < noise {NOISE})",
        stalled_gaps.len(),
        median(&stalled_gaps),
        below(&stalled_gaps, NOISE),
    );
    println!(
        "Settled markets: {} (median tightest gap {:.4})",
        settled_gaps.len(),
        median(&settled_gaps),
    );
    // The decisive mechanism test: in a near-tie cascade, the displaced proposer
    // that *initiates* the wrong matching is itself near-indifferent (tiny own
    // loss), so its cheap swap externalizes large regret onto others.
    stalled_initiator_loss.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let below = |v: &[f64], t: f64| v.iter().filter(|&&g| g < t).count();
    if !stalled_initiator_loss.is_empty() {
        let n = stalled_initiator_loss.len();
        println!("\nCascade-initiator own-loss in stalled markets (min over displaced proposers):");
        println!(
            "  median {:.4}, {}/{} below 0.01, {}/{} below 0.05  (a near-indifferent proposer whose cheap swap cascades)",
            stalled_initiator_loss[n / 2],
            below(&stalled_initiator_loss, 0.01),
            n,
            below(&stalled_initiator_loss, 0.05),
            n,
        );
    }

    println!(
        "\nConclusion: if stall rate falls as the gap widens, stalled markets cluster at\nsub-noise gaps, and each stalled market has a near-indifferent displaced proposer,\nthen genuine lock-ins are a near-tie identifiability limit (gap << noise) amplified\nby Gale-Shapley's discontinuity — which more exploration cannot fix."
    );
}
