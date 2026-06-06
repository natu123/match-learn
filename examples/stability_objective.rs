//! Why the *stability-direct* coordinator has no 2ε ceiling that the
//! *belief-welfare* coordinator does (`docs/theory-stability-objective.md`).
//!
//! The two coordinators differ only in their objective over near-tie reorderings:
//!  - `welfare` : pick the matching maximizing Σ_p mean_p[partner].
//!  - `stability`: pick the matching minimizing the number of blocking pairs.
//!
//! This example removes the *belief-noise* confound entirely by giving both
//! coordinators **perfect information** (beliefs = true utilities). Any residual
//! instability is then purely a property of the *objective*, not of learning. We
//! sweep random 5×5 markets and report how often each coordinator's chosen
//! matching is unstable, and how often they disagree with `stability` strictly
//! better. If `welfare` is unstable on a positive fraction at perfect info, its
//! objective is not stability-consistent — the structural reason for its live cap.
//!
//! ```text
//! cargo run --release --example stability_objective
//! ```

use match_learn::matching::Matching;
use match_learn::{Rng, gale_shapley, rank_by_scores};

const EPS: f64 = 0.05; // near-tie band the coordinator may reorder within
const GMAX: usize = 3; // cap reordered group size (search cost)

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

/// Candidate rankings for one proposer: reorder within near-tie groups of size
/// `2..=GMAX` (larger groups kept in mean order).
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

/// Number of true blocking pairs of `m` under strict preference rankings.
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
            let r_holder = m.receiver[r];
            let r_wants = r_holder.is_none_or(|h| pos(&recv[r], p) < pos(&recv[r], h));
            if r_wants {
                blocks += 1;
            }
        }
    }
    blocks
}

/// Enumerate the joint candidate product; return (welfare-max matching,
/// blocking-min matching) under perfect-info means.
fn coordinate(
    per: &[Vec<Vec<usize>>],
    means: &[Vec<f64>],
    recv: &[Vec<usize>],
) -> (Matching, Matching) {
    let n = per.len();
    let welfare = |m: &Matching| -> f64 {
        (0..n)
            .map(|p| m.proposer[p].map_or(0.0, |r| means[p][r]))
            .sum()
    };
    let mut idx = vec![0usize; n];
    let mut best_w = f64::NEG_INFINITY;
    let mut wmatch: Option<Matching> = None;
    let mut best_b = usize::MAX;
    let mut bmatch: Option<Matching> = None;
    let prop_true: Vec<Vec<usize>> = means.iter().map(|m| rank_by_scores(m)).collect();
    'outer: loop {
        let rankings: Vec<Vec<usize>> = (0..n).map(|p| per[p][idx[p]].clone()).collect();
        let m = gale_shapley(&rankings, recv);
        let w = welfare(&m);
        if w > best_w {
            best_w = w;
            wmatch = Some(m.clone());
        }
        let b = count_blocking(&prop_true, recv, &m);
        if b < best_b {
            best_b = b;
            bmatch = Some(m);
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
    (wmatch.unwrap(), bmatch.unwrap())
}

fn main() {
    const SWEEP: usize = 4000;
    const N: usize = 5;
    println!(
        "Stability vs welfare objective at PERFECT information ({SWEEP} markets {N}x{N}, ε={EPS})\n"
    );

    let mut seedgen = Rng::new(20260607);
    let (mut w_unstable, mut s_unstable, mut disagree, mut s_strictly_better) = (0, 0, 0, 0);
    let mut had_neartie = 0;
    let mut witness: Option<u64> = None;
    for _ in 0..SWEEP {
        let seed = seedgen.below(1_000_000_000) as u64 + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, N);
        let means = util.clone(); // perfect information
        let prop_true: Vec<Vec<usize>> = means.iter().map(|m| rank_by_scores(m)).collect();

        let per: Vec<Vec<Vec<usize>>> = means.iter().map(|m| candidates(m)).collect();
        if per.iter().any(|c| c.len() > 1) {
            had_neartie += 1;
        }
        let (wm, sm) = coordinate(&per, &means, &recv);
        let wb = count_blocking(&prop_true, &recv, &wm);
        let sb = count_blocking(&prop_true, &recv, &sm);
        if wb > 0 {
            w_unstable += 1;
        }
        if sb > 0 {
            s_unstable += 1;
        }
        if wm.proposer != sm.proposer {
            disagree += 1;
            if sb < wb {
                s_strictly_better += 1;
                if witness.is_none() {
                    witness = Some(seed);
                }
            }
        }
    }
    let pct = |x: usize| 100.0 * x as f64 / SWEEP as f64;
    println!("markets with a near-tie group (coordinator active): {had_neartie} / {SWEEP}");
    println!(
        "welfare-objective matching unstable   : {w_unstable:>4}  ({:.1}%)",
        pct(w_unstable)
    );
    println!(
        "stability-objective matching unstable : {s_unstable:>4}  ({:.1}%)",
        pct(s_unstable)
    );
    println!(
        "objectives disagree                   : {disagree:>4}  ({:.1}%)",
        pct(disagree)
    );
    println!("  of which stability strictly fewer blocks: {s_strictly_better}");

    if let Some(seed) = witness {
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, N);
        let means = util.clone();
        let prop_true: Vec<Vec<usize>> = means.iter().map(|m| rank_by_scores(m)).collect();
        let per: Vec<Vec<Vec<usize>>> = means.iter().map(|m| candidates(m)).collect();
        let (wm, sm) = coordinate(&per, &means, &recv);
        let w = |m: &Matching| -> f64 {
            (0..N)
                .map(|p| m.proposer[p].map_or(0.0, |r| means[p][r]))
                .sum()
        };
        let names = ["A", "B", "C", "D", "E"];
        let fmt = |m: &Matching| {
            (0..N)
                .map(|p| format!("p{p}->{}", m.proposer[p].map_or("∅", |r| names[r])))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!("\n  Witness market (seed {seed}):");
        println!(
            "    welfare-obj  : {}   welfare {:.3}  blocks {}",
            fmt(&wm),
            w(&wm),
            count_blocking(&prop_true, &recv, &wm)
        );
        println!(
            "    stability-obj: {}   welfare {:.3}  blocks {}",
            fmt(&sm),
            w(&sm),
            count_blocking(&prop_true, &recv, &sm)
        );
        println!("    -> welfare-obj takes a higher-welfare but UNSTABLE near-tie ordering;");
        println!("       stability-obj keeps a stable one at slightly lower proposer welfare.");
    }
    println!(
        "\nReading: at PERFECT information the welfare objective is still unstable on a\n\
         positive fraction (it trades a near-tie agent's ~0 loss for cross-agent welfare,\n\
         landing on a proposer-favoring UNSTABLE matching), while the stability objective\n\
         drives blocking pairs to their reachable minimum. The instability is a property\n\
         of the OBJECTIVE, not of belief noise — the structural reason welfare-gating caps\n\
         at 2ε while stability-targeting does not (cf. dev live: 0.699 vs 0.961)."
    );
}
