//! Decay rate of the stability coordinator's residual instability vs belief
//! accuracy (the open quantitative question of `theory-stability-objective.md`
//! Prop. 6). Companion to `stability_objective.rs`, which is the `η→0` (perfect
//! information) endpoint.
//!
//! We sweep a belief-noise level `η` (each belief mean = true utility plus
//! `N(0, η)`), and at each `η` measure, over many random markets, the TRUE
//! instability of the two coordinators' chosen matchings:
//!
//! - `welfare`: argmax Σ belief-mean[partner].
//! - `stability`: argmin estimated blocking pairs (belief proposer prefs vs the
//!   known-exact receiver prefs).
//!
//! The coordinator only ever sees beliefs; instability is scored against truth.
//!
//! Prop. 6 predicts: `stability` instability → 0 as `η → 0` (it is unbiased,
//! noise-limited), while `welfare` plateaus at its perfect-information bias floor
//! (~27%, cf. `stability_objective.rs`). The per-pair misjudgement probability is
//! `Φ(−g/(η√2))` for a true gap `g`, so with `η ≈ σ/√N` the stability instability
//! decays like `exp(−N·g²/(2σ²))` — the `Θ(σ²/g²)` resolution horizon per pair.
//!
//! ```text
//! cargo run --release --example decay_rate
//! ```

use match_learn::matching::Matching;
use match_learn::{Rng, gale_shapley, rank_by_scores};

const EPS: f64 = 0.05;
const GMAX: usize = 3;

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let recv = (0..n).map(|_| rng.permutation(n)).collect();
    (util, recv)
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

fn near_tie_groups(base: &[usize], means: &[f64]) -> Vec<Vec<usize>> {
    let mut groups = vec![vec![base[0]]];
    for &arm in &base[1..] {
        let prev = *groups.last().unwrap().last().unwrap();
        if (means[prev] - means[arm]).abs() < EPS {
            groups.last_mut().unwrap().push(arm);
        } else {
            groups.push(vec![arm]);
        }
    }
    groups
}

fn candidates(means: &[f64]) -> Vec<Vec<usize>> {
    let base = rank_by_scores(means);
    let groups = near_tie_groups(&base, means);
    let mut rankings = vec![vec![]];
    for g in &groups {
        let perms = if (2..=GMAX).contains(&g.len()) {
            permutations(g)
        } else {
            vec![g.clone()]
        };
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

/// Blocking-pair count of `m` under (proposer prefs, receiver prefs).
fn count_blocking(prop: &[Vec<usize>], recv: &[Vec<usize>], m: &Matching) -> usize {
    let pos = |rank: &[usize], x: usize| rank.iter().position(|&y| y == x).unwrap();
    let mut blocks = 0;
    for (p, prefs) in prop.iter().enumerate() {
        let cur = m.proposer[p];
        for &r in prefs {
            let wants = cur.is_none_or(|c| pos(prefs, r) < pos(prefs, c));
            if !wants {
                continue;
            }
            let r_wants = m.receiver[r].is_none_or(|h| pos(&recv[r], p) < pos(&recv[r], h));
            if r_wants {
                blocks += 1;
            }
        }
    }
    blocks
}

/// Pick (welfare-max, blocking-min) matchings. Welfare and blocking are both
/// scored from BELIEFS (means / belief rankings); receiver prefs are exact.
fn coordinate(
    per: &[Vec<Vec<usize>>],
    means: &[Vec<f64>],
    belief_rank: &[Vec<usize>],
    recv: &[Vec<usize>],
) -> (Matching, Matching) {
    let n = per.len();
    let welfare = |m: &Matching| -> f64 {
        (0..n)
            .map(|p| m.proposer[p].map_or(0.0, |r| means[p][r]))
            .sum()
    };
    let mut idx = vec![0usize; n];
    let (mut best_w, mut best_b) = (f64::NEG_INFINITY, usize::MAX);
    let (mut wm, mut bm): (Option<Matching>, Option<Matching>) = (None, None);
    'outer: loop {
        let rankings: Vec<Vec<usize>> = (0..n).map(|p| per[p][idx[p]].clone()).collect();
        let m = gale_shapley(&rankings, recv);
        let w = welfare(&m);
        if w > best_w {
            best_w = w;
            wm = Some(m.clone());
        }
        let b = count_blocking(belief_rank, recv, &m); // estimated from beliefs
        if b < best_b {
            best_b = b;
            bm = Some(m);
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
    (wm.unwrap(), bm.unwrap())
}

fn main() {
    const SWEEP: usize = 3000;
    const N: usize = 5;
    let etas = [0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.0];
    println!("Stability vs welfare coordinator: TRUE instability vs belief noise η");
    println!("({SWEEP} random {N}x{N} markets per η, ε={EPS}; coordinator sees beliefs only)\n");
    println!("    η      welfare unstable   stability unstable");

    for &eta in &etas {
        let mut seedgen = Rng::new(20260607);
        let (mut w_un, mut s_un) = (0usize, 0usize);
        for _ in 0..SWEEP {
            let seed = seedgen.below(1_000_000_000) as u64 + 1;
            let mut mgen = Rng::new(seed);
            let (util, recv) = random_market(&mut mgen, N);
            let true_rank: Vec<Vec<usize>> = util.iter().map(|u| rank_by_scores(u)).collect();

            // Beliefs = true utilities + N(0, eta).
            let mut bgen = Rng::new(seed ^ 0xBE11E5);
            let means: Vec<Vec<f64>> = util
                .iter()
                .map(|u| u.iter().map(|&x| x + bgen.normal(0.0, eta)).collect())
                .collect();
            let belief_rank: Vec<Vec<usize>> = means.iter().map(|m| rank_by_scores(m)).collect();

            let per: Vec<Vec<Vec<usize>>> = means.iter().map(|m| candidates(m)).collect();
            let (wm, sm) = coordinate(&per, &means, &belief_rank, &recv);
            if count_blocking(&true_rank, &recv, &wm) > 0 {
                w_un += 1;
            }
            if count_blocking(&true_rank, &recv, &sm) > 0 {
                s_un += 1;
            }
        }
        let pct = |x: usize| 100.0 * x as f64 / SWEEP as f64;
        println!(
            "  {eta:>5.2}        {:>6.1}%             {:>6.1}%",
            pct(w_un),
            pct(s_un)
        );
    }
    println!(
        "\nReading: as η→0 the stability coordinator's TRUE instability decays toward 0\n\
         (unbiased, noise-limited — Prop. 6(1)), while the welfare coordinator flattens at\n\
         its perfect-information BIAS floor (~27%, Prop. 5). With η ≈ σ/√N the stability\n\
         decay is the per-pair Θ(σ²/g²) resolution rate; welfare's floor never decays."
    );
}
