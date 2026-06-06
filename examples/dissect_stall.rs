//! Dissect a genuine stall: find a market where greedy Thompson Sampling locks
//! into a wrong stable matching even at a long horizon, then take it apart to see
//! *why* — is it a single agent's frozen arm, or a multi-agent coordination
//! failure?
//!
//! Phase 1 searches many seeds at a small, readable size for the worst
//! long-horizon lock-in. Phase 2 prints that market's true stable matching, the
//! matching the learners settle on, and each proposer's posterior beliefs versus
//! the truth — exposing the frozen underestimate — and classifies the cause.
//!
//! ```text
//! cargo run --release --example dissect_stall
//! ```

use match_learn::matching::Matching;
use match_learn::{Market, Rng, gale_shapley, rank_by_scores, simulate};

const N: usize = 4;
const NOISE: f64 = 0.2;
const SEARCH_SEEDS: u64 = 1500;
const SEARCH_H: usize = 6000;
const LONG_H: usize = 40000;

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

fn thompson(util: &[Vec<f64>], recv: &[Vec<usize>], seed: u64) -> Market {
    Market::with_thompson(
        util.to_vec(),
        recv.to_vec(),
        0.5,
        1.0,
        NOISE * NOISE,
        NOISE,
        seed ^ 0xABCD,
    )
}

/// The matching the learners have settled on: Gale-Shapley applied to the
/// *deterministic* belief-mean rankings (what the market plays once posterior
/// noise is negligible).
fn belief_matching(market: &Market) -> Matching {
    let rankings: Vec<Vec<usize>> = market
        .belief_means()
        .iter()
        .map(|means| rank_by_scores(means))
        .collect();
    gale_shapley(&rankings, market.receiver_prefs())
}

fn matching_str(m: &Matching, n: usize) -> String {
    (0..n)
        .map(|p| match m.proposer[p] {
            Some(r) => format!("p{p}->r{r}"),
            None => format!("p{p}->_"),
        })
        .collect::<Vec<_>>()
        .join("  ")
}

fn main() {
    // --- Phase 1: hunt for the worst long-horizon lock-in ---------------------
    let mut seedgen = Rng::new(20260606);
    let mut worst_seed = 0u64;
    let mut worst_tail = -1.0;
    for _ in 0..SEARCH_SEEDS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, N);
        let mut market = thompson(&util, &recv, seed);
        let rep = simulate(&mut market, SEARCH_H);
        let tail = rep.tail_mean_regret(SEARCH_H / 5);
        if tail > worst_tail {
            worst_tail = tail;
            worst_seed = seed;
        }
    }
    println!(
        "Worst Thompson lock-in over {SEARCH_SEEDS} seeds at {N}x{N}: seed {worst_seed}, tail regret rate {worst_tail:.4} (horizon {SEARCH_H})\n"
    );

    // --- Phase 2: dissect that market -----------------------------------------
    let mut mgen = Rng::new(worst_seed);
    let (util, recv) = random_market(&mut mgen, N);
    let mut market = thompson(&util, &recv, worst_seed);
    let rep = simulate(&mut market, LONG_H);

    let true_stable = market.true_stable_matching();
    let settled = belief_matching(&market);

    println!("True utilities (proposer x receiver):");
    for (p, row) in util.iter().enumerate() {
        let cells: Vec<String> = row.iter().map(|u| format!("{u:.3}")).collect();
        println!("  p{p}: [{}]", cells.join(", "));
    }
    println!("\nReceiver preferences (most preferred first):");
    for (r, pref) in recv.iter().enumerate() {
        println!("  r{r}: {pref:?}");
    }

    println!(
        "\nTrue stable matching:   {}",
        matching_str(&true_stable, N)
    );
    println!(
        "Settled (belief) match: {}   [tail regret rate at horizon {LONG_H}: {:.4}]",
        matching_str(&settled, N),
        rep.tail_mean_regret(LONG_H / 5),
    );

    println!("\nPer-proposer beliefs vs truth (★ = true stable partner, ◆ = settled partner):");
    let means = market.belief_means();
    for p in 0..N {
        let star = true_stable.proposer[p];
        let got = settled.proposer[p];
        let cells: Vec<String> = (0..N)
            .map(|r| {
                let mark = if Some(r) == star {
                    "★"
                } else if Some(r) == got {
                    "◆"
                } else {
                    " "
                };
                format!("r{r}{mark} true={:.3} est={:.3}", util[p][r], means[p][r])
            })
            .collect();
        println!("  p{p}: {}", cells.join(" | "));
    }

    // --- Classify the cause ---------------------------------------------------
    println!("\nDiagnosis:");
    for p in 0..N {
        let (Some(star), got) = (true_stable.proposer[p], settled.proposer[p]) else {
            continue;
        };
        if Some(star) == got {
            continue; // this proposer is fine
        }
        // p is not with its true stable partner r* = star. Two questions:
        //  (1) Does p *underestimate* r* (a frozen-arm symptom)?
        //  (2) Would r* actually take p, given who holds r* in the settled match?
        let underestimate = means[p][star] < util[p][star] - 0.1;
        // Who currently holds r* in the settled matching?
        let holder = (0..N).find(|&q| settled.proposer[q] == Some(star));
        let r_pref = &recv[star];
        let pos = |x: usize| r_pref.iter().position(|&y| y == x).unwrap_or(usize::MAX);
        let would_accept = match holder {
            Some(h) => pos(p) < pos(h), // r* prefers p to its current holder
            None => true,               // r* is free
        };
        let cause = match (underestimate, would_accept) {
            (true, true) => {
                "FROZEN underestimate of r*, and r* would accept p -> single-agent frozen arm"
            }
            (true, false) => {
                "underestimates r*, but r* prefers its holder -> blocked by competition (coordination)"
            }
            (false, true) => {
                "estimate of r* is fine, yet not proposing -> ranking/competition effect"
            }
            (false, false) => "r* prefers its holder -> genuine competition, not a frozen arm",
        };
        println!(
            "  p{p}: settled with {got:?}, true partner r{star} (est {:.3} vs true {:.3}); holder of r{star} = {holder:?}; {cause}",
            means[p][star], util[p][star]
        );
    }
}
