//! Explore: a *search-free* coordinator for the cascade mode, using the known
//! receiver preferences instead of an exponential belief-welfare search.
//!
//! Heuristic (derived from the dissected lock-in): when a proposer is indifferent
//! among a near-tie group of receivers, it should take the receiver that prefers
//! it **least**, freeing the receivers that prefer it *most* for proposers who
//! would otherwise be blocked. In the dissected case p0 is r1's top choice and
//! r3's last; taking r3 frees r1 for p3 — exactly the true stable matching.
//!
//! We compare three tie-break policies on the converged beliefs over many markets:
//! - `index`            — default (ties by arm index),
//! - `receiver-informed`— O(n log n), search-free (this heuristic),
//! - `belief-welfare`   — the exponential brute-force coordinator (POC, upper bar).
//!
//! ```text
//! cargo run --release --example receiver_informed
//! ```

use match_learn::matching::Matching;
use match_learn::{Market, Rng, gale_shapley, rank_by_scores, simulate};

const N: usize = 5;
const NOISE: f64 = 0.2;
const SEEDS: usize = 800;
const HORIZON: usize = 8000;
const TIE_EPS: f64 = 0.05;
const CASCADE: f64 = 0.1;

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let true_util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let receiver_prefs = (0..n).map(|_| rng.permutation(n)).collect();
    (true_util, receiver_prefs)
}

fn regret(util: &[Vec<f64>], baseline: &Matching, m: &Matching) -> f64 {
    (0..util.len())
        .map(|p| {
            let b = baseline.proposer[p].map_or(0.0, |r| util[p][r]);
            let g = m.proposer[p].map_or(0.0, |r| util[p][r]);
            b - g
        })
        .sum()
}

/// Contiguous near-tie groups of `means`' descending ranking.
fn tie_groups(means: &[f64]) -> Vec<Vec<usize>> {
    let base = rank_by_scores(means);
    let mut groups: Vec<Vec<usize>> = vec![vec![base[0]]];
    for &arm in &base[1..] {
        let prev = *groups.last().unwrap().last().unwrap();
        if (means[prev] - means[arm]).abs() < TIE_EPS {
            groups.last_mut().unwrap().push(arm);
        } else {
            groups.push(vec![arm]);
        }
    }
    groups
}

/// Receiver-informed ranking for proposer `p`: within each near-tie group, put the
/// receiver that ranks `p` *worst* (highest position index) first.
fn receiver_informed_ranking(p: usize, means: &[f64], recv: &[Vec<usize>]) -> Vec<usize> {
    let pos_of_p = |r: usize| recv[r].iter().position(|&q| q == p).unwrap_or(usize::MAX);
    let mut out = Vec::new();
    for mut g in tie_groups(means) {
        g.sort_by(|&x, &y| pos_of_p(y).cmp(&pos_of_p(x))); // worst-preferring receiver first
        out.extend(g);
    }
    out
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

/// Belief-welfare brute force (the POC coordinator), as the upper bar.
fn belief_welfare_match(means: &[Vec<f64>], recv: &[Vec<usize>]) -> Matching {
    let n = means.len();
    let per: Vec<Vec<Vec<usize>>> = means
        .iter()
        .map(|m| {
            let mut rs = vec![vec![]];
            for g in tie_groups(m) {
                let perms = permutations(&g);
                let mut next = Vec::new();
                for prefix in &rs {
                    for perm in &perms {
                        let mut r = prefix.clone();
                        r.extend(perm);
                        next.push(r);
                    }
                }
                rs = next;
            }
            rs
        })
        .collect();
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

fn main() {
    println!(
        "Receiver-informed tie-break — {SEEDS} markets ({N}x{N}, horizon {HORIZON}), cascade > {CASCADE}\n"
    );
    let mut seedgen = Rng::new(20260606);
    let (mut cascades, mut fix_recv, mut fix_bw) = (0usize, 0usize, 0usize);
    let (mut sum_idx, mut sum_recv, mut sum_bw) = (0.0, 0.0, 0.0);
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

        let index_rankings: Vec<Vec<usize>> = means.iter().map(|m| rank_by_scores(m)).collect();
        let ir = regret(&util, &baseline, &gale_shapley(&index_rankings, &recv));
        if ir <= CASCADE {
            continue;
        }
        cascades += 1;
        sum_idx += ir;

        let ri_rankings: Vec<Vec<usize>> = (0..N)
            .map(|p| receiver_informed_ranking(p, &means[p], &recv))
            .collect();
        let rr = regret(&util, &baseline, &gale_shapley(&ri_rankings, &recv));
        sum_recv += rr;
        if rr < 0.01 {
            fix_recv += 1;
        }

        let bw = regret(&util, &baseline, &belief_welfare_match(&means, &recv));
        sum_bw += bw;
        if bw < 0.01 {
            fix_bw += 1;
        }
    }
    println!("cascade markets: {cascades}");
    println!(
        "  index (default)        mean cascade regret {:.3}",
        sum_idx / cascades.max(1) as f64
    );
    println!(
        "  receiver-informed      fixes {fix_recv}/{cascades}   mean regret {:.3}   (search-free, O(n log n))",
        sum_recv / cascades.max(1) as f64
    );
    println!(
        "  belief-welfare (brute) fixes {fix_bw}/{cascades}   mean regret {:.3}   (exponential upper bar)",
        sum_bw / cascades.max(1) as f64
    );
    println!(
        "\n(If receiver-informed approaches the brute-force bar, the known receiver preferences\nalone resolve most cascades -- a cheap coordinator with no ordering search.)"
    );
}
