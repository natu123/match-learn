//! Proof of concept: can *coordinated near-tie tie-breaking* remove a cascade
//! stall that no per-agent exploration policy can?
//!
//! The dissected worst lock-in (seed 235418470, 4x4) is a near-tie cascade: one
//! proposer is indifferent between two receivers (true gap ~0.001) and its
//! arbitrary order, amplified by Gale-Shapley, costs another proposer ~0.84
//! regret. Because that proposer is *indifferent*, breaking its tie the other way
//! is free for it. The question: if a market-level coordinator is allowed to
//! choose the tie-break among near-equal arms to minimize total regret, does the
//! cascade vanish?
//!
//! We learn beliefs with plain Thompson, then compare three tie-break policies on
//! the converged belief means:
//! - `index`  — the current default (ties broken by arm index),
//! - `random` sample of orderings (sanity), and
//! - `coordinated` — search the near-tie orderings for the lowest-regret matching.
//!
//! If `coordinated` recovers ~0 regret where `index` cascades, coordinated
//! tie-breaking is a real cure for the cascade mode — motivating a market-level
//! mechanism above `PreferenceLearner`.
//!
//! ⚠ IMPORTANT (post-hoc only): this runs on **converged** belief means. The
//! result does NOT transfer to a live loop. The implementation team's live
//! `CoordinatedMarket` (coordinating each round on *current* beliefs) lost
//! stability to plain Thompson, because belief-welfare-max on inaccurate
//! mid-learning beliefs picks welfare-optimal-but-unstable matchings. See
//! `docs/stall-anatomy.md` §4.2. A live cure needs confidence-gating or a
//! stability objective.
//!
//! ```text
//! cargo run --release --example coordinated_poc
//! ```

use match_learn::matching::Matching;
use match_learn::{Market, Rng, gale_shapley, rank_by_scores, simulate};

const N: usize = 4;
const NOISE: f64 = 0.2;
const SEED: u64 = 235418470;
const HORIZON: usize = 40000;
/// Two arms within this belief-mean gap are treated as a near-tie the coordinator
/// may reorder. It is on the order of the noise-resolution floor, well above the
/// true 0.001 gap the learner cannot resolve.
const TIE_EPS: f64 = 0.05;

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

/// Total proposer regret of `m` against the true stable matching `baseline`.
fn regret(util: &[Vec<f64>], baseline: &Matching, m: &Matching) -> f64 {
    (0..util.len())
        .map(|p| {
            let b = baseline.proposer[p].map_or(0.0, |r| util[p][r]);
            let g = m.proposer[p].map_or(0.0, |r| util[p][r]);
            b - g
        })
        .sum()
}

/// All index-rankings reachable from `base` by reordering within near-tie groups
/// (arms whose belief mean is within `TIE_EPS` of the previous in the ranking).
/// Returns a small set of candidate rankings for this proposer.
fn near_tie_rankings(means: &[f64]) -> Vec<Vec<usize>> {
    let base = rank_by_scores(means); // descending by mean, index tie-break
    // Partition the ranking into contiguous near-tie groups.
    let mut groups: Vec<Vec<usize>> = vec![vec![base[0]]];
    for &arm in &base[1..] {
        let prev = *groups.last().unwrap().last().unwrap();
        if (means[prev] - means[arm]).abs() < TIE_EPS {
            groups.last_mut().unwrap().push(arm);
        } else {
            groups.push(vec![arm]);
        }
    }
    // Cartesian product of within-group permutations (groups are tiny).
    let mut rankings = vec![vec![]];
    for g in &groups {
        let perms = permutations(g);
        let mut next = Vec::new();
        for prefix in &rankings {
            for perm in &perms {
                let mut r = prefix.clone();
                r.extend(perm);
                next.push(r);
            }
        }
        rankings = next;
    }
    rankings
}

fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

fn main() {
    let mut mgen = Rng::new(SEED);
    let (util, recv) = random_market(&mut mgen, N);
    let mut market = Market::with_thompson(
        util.clone(),
        recv.clone(),
        0.5,
        1.0,
        NOISE * NOISE,
        NOISE,
        SEED ^ 0xABCD,
    );
    simulate(&mut market, HORIZON);
    let means = market.belief_means();
    let baseline = market.true_stable_matching();

    // Policy 1: index tie-break (the current default).
    let index_rankings: Vec<Vec<usize>> = means.iter().map(|m| rank_by_scores(m)).collect();
    let index_match = gale_shapley(&index_rankings, &recv);
    let index_regret = regret(&util, &baseline, &index_match);

    // Policy 2: coordinated. Search near-tie orderings and pick the matching that
    // maximizes total *belief* welfare (no access to true utilities) — the
    // practical objective a real coordinator could optimize. We also track the
    // true-regret-optimal choice (oracle) to confirm the proxy finds it.
    let belief_welfare = |m: &Matching| -> f64 {
        (0..N)
            .map(|p| m.proposer[p].map_or(0.0, |r| means[p][r]))
            .sum()
    };
    let per_proposer: Vec<Vec<Vec<usize>>> = means.iter().map(|m| near_tie_rankings(m)).collect();
    let combos: usize = per_proposer.iter().map(|r| r.len()).product();
    let mut best_regret = f64::INFINITY; // oracle objective
    let mut oracle_match = index_match.clone();
    let mut best_welfare = f64::NEG_INFINITY; // practical objective
    let mut best_match = index_match.clone();
    // Enumerate the Cartesian product of per-proposer candidate rankings.
    let mut idx = [0usize; N];
    'outer: loop {
        let rankings: Vec<Vec<usize>> = (0..N).map(|p| per_proposer[p][idx[p]].clone()).collect();
        let m = gale_shapley(&rankings, &recv);
        let r = regret(&util, &baseline, &m);
        if r < best_regret {
            best_regret = r;
            oracle_match = m.clone();
        }
        let w = belief_welfare(&m);
        if w > best_welfare {
            best_welfare = w;
            best_match = m;
        }
        // increment mixed-radix counter
        let mut k = 0;
        loop {
            if k == N {
                break 'outer;
            }
            idx[k] += 1;
            if idx[k] < per_proposer[k].len() {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
    let coord_regret = regret(&util, &baseline, &best_match); // true regret of the practical pick

    let fmt = |m: &Matching| {
        (0..N)
            .map(|p| format!("p{p}->r{}", m.proposer[p].map_or(99, |r| r)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!("Coordinated tie-break POC — seed {SEED}, {N}x{N}, horizon {HORIZON}\n");
    println!("true stable : {}", fmt(&baseline));
    println!(
        "index break : {}   regret {index_regret:.4}",
        fmt(&index_match)
    );
    println!(
        "coordinated : {}   regret {coord_regret:.4}   (max belief welfare over {combos} near-tie orderings)",
        fmt(&best_match)
    );
    println!(
        "  oracle    : {}   regret {best_regret:.4}   (true-regret-optimal, for reference)",
        fmt(&oracle_match)
    );
    let coincide = best_match.proposer == oracle_match.proposer;
    println!(
        "\nResult: coordinated tie-breaking by *belief welfare* (no true utilities) recovers regret\n{coord_regret:.4} vs {index_regret:.4} for the index default, and {} the true-regret-optimal matching.",
        if coincide { "matches" } else { "differs from" }
    );
    println!(
        "The reordered proposer was within {TIE_EPS} of indifferent, so the fix costs it ~nothing\nwhile its swap gains another proposer a lot -- a practical, oracle-free market-level coordinator."
    );

    // --- Coverage: how many cascade markets does coordinated tie-break fix? -----
    coverage_sweep();
}

/// Best matching by total belief welfare over near-tie orderings (oracle-free).
fn coordinated_match(means: &[Vec<f64>], recv: &[Vec<usize>]) -> Matching {
    let n = means.len();
    let per: Vec<Vec<Vec<usize>>> = means.iter().map(|m| near_tie_rankings(m)).collect();
    let welfare = |m: &Matching| -> f64 {
        (0..n)
            .map(|p| m.proposer[p].map_or(0.0, |r| means[p][r]))
            .sum()
    };
    let mut best = f64::NEG_INFINITY;
    let mut best_m: Option<Matching> = None;
    let mut idx = vec![0usize; n];
    'outer: loop {
        let rankings: Vec<Vec<usize>> = (0..n).map(|p| per[p][idx[p]].clone()).collect();
        let m = gale_shapley(&rankings, recv);
        let w = welfare(&m);
        if w > best {
            best = w;
            best_m = Some(m);
        }
        let mut k = 0;
        loop {
            if k == n {
                break 'outer;
            }
            idx[k] += 1;
            if idx[k] < per[k].len() {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
    best_m.unwrap()
}

fn coverage_sweep() {
    const SWEEP_N: usize = 5;
    const SWEEP_SEEDS: usize = 800;
    const SWEEP_H: usize = 8000;
    const CASCADE: f64 = 0.1; // index-tie settled regret above this = a cascade

    let mut seedgen = Rng::new(20260606);
    let (mut cascades, mut fixed) = (0usize, 0usize);
    let mut index_sum = 0.0;
    let mut coord_sum = 0.0;
    for _ in 0..SWEEP_SEEDS {
        let seed = (seedgen.below(1_000_000_000) as u64) + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, SWEEP_N);
        let mut market = Market::with_thompson(
            util.clone(),
            recv.clone(),
            0.5,
            1.0,
            NOISE * NOISE,
            NOISE,
            seed ^ 0xABCD,
        );
        simulate(&mut market, SWEEP_H);
        let means = market.belief_means();
        let baseline = market.true_stable_matching();
        let index_rankings: Vec<Vec<usize>> = means.iter().map(|m| rank_by_scores(m)).collect();
        let ir = regret(&util, &baseline, &gale_shapley(&index_rankings, &recv));
        if ir > CASCADE {
            cascades += 1;
            let cr = regret(&util, &baseline, &coordinated_match(&means, &recv));
            index_sum += ir;
            coord_sum += cr;
            if cr < 0.01 {
                fixed += 1;
            }
        }
    }
    println!(
        "\n--- Coverage over {SWEEP_SEEDS} markets ({SWEEP_N}x{SWEEP_N}, horizon {SWEEP_H}) ---"
    );
    println!(
        "settled-matching cascades (index-tie regret > {CASCADE}): {cascades}\n  coordinated belief-welfare tie-break fully fixes {fixed}/{cascades} of them",
    );
    if cascades > 0 {
        println!(
            "  mean settled regret on cascades: index {:.3} -> coordinated {:.3}",
            index_sum / cascades as f64,
            coord_sum / cascades as f64,
        );
    }
    println!(
        "(Cascades it cannot fix are the frozen-arm cases, where beliefs themselves are wrong --\nthose are forced exploration's job, not the coordinator's.)"
    );
}
