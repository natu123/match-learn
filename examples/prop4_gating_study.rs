//! Controlled **same-belief-stream** A/B of Prop. 4 confidence-gating
//! (`docs/theory-identifiability.md` §4a). Complements the dev side's *live*
//! product validation (`coordinated_validation.rs`): here a single Thompson loop
//! generates the belief trajectory, and each measured round the three
//! coordination decisions are formed **on identical beliefs** — isolating the
//! gate's effect on the decision from the learning-feedback confound.
//!
//! On the current posterior means/stds we form three matchings:
//!
//! - `plain`: Gale-Shapley on the belief-mean ranking (no coordination).
//! - `ungated`: reorder every near-tie group (mean-gap < ε) to maximize belief
//!   welfare (the rule that fails live).
//! - `gated`: reorder a group only once it certifies as ε-tied,
//!   `|m̂_a−m̂_b| + z·√(s_a²+s_b²) ≤ ε` for each consecutive pair (Lemma 2).
//!   Un-certified groups keep the mean order.
//!
//! Each is scored by true exact-stability and true regret vs `M*`, split into an
//! early (coarse-belief) and a late (accurate-belief) regime.
//!
//! Findings: `ungated` loses exact-stability vs `plain` in BOTH regimes (it
//! reorders sub-resolution near-ties into proposer-favoring but unstable
//! matchings, so its regret even goes negative); `gated` restores most of the
//! stability (Prop. 4(1) safety) and `≈ plain` at the tight band ε=0.02, with a
//! reorder-rate rising early→late (the Prop. 4(2) activation curve). Honest limit:
//! belief-welfare-max stays unstable even with accurate beliefs, so the gate caps
//! the damage but does not make the welfare objective correct — evidence for
//! optimizing stability directly (the §4a alternative).
//!
//! The welfare search is exponential in the largest near-tie group, so groups
//! above `GMAX` are left unre-ordered and any round whose candidate product
//! exceeds `COMBO_CAP` falls back to `plain` (both counted and reported — no
//! silent truncation).
//!
//! ```text
//! cargo run --release --example prop4_gating_study
//! ```

use match_learn::matching::Matching;
use match_learn::{
    GaussianThompson, PreferenceLearner, Rng, gale_shapley, is_stable, rank_by_scores,
};

const NOISE: f64 = 0.2;
const Z: f64 = 1.64; // η ≈ 0.05 per-pair certification confidence
const FORCE_C: f64 = 0.5; // light forcing ε_t = min(1, c/t) drives concentration
const GMAX: usize = 3; // max near-tie group size the coordinator will reorder
const COMBO_CAP: usize = 30_000; // max candidate matchings searched per round
const MEAS_EVERY: usize = 20; // measure the A/B every this many rounds

fn random_market(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let util = (0..n)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let recv = (0..n).map(|_| rng.permutation(n)).collect();
    (util, recv)
}

fn true_rankings(util: &[Vec<f64>]) -> Vec<Vec<usize>> {
    util.iter().map(|u| rank_by_scores(u)).collect()
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

/// Contiguous near-tie groups of `base` (adjacent means within `eps`).
fn near_tie_groups(base: &[usize], means: &[f64], eps: f64) -> Vec<Vec<usize>> {
    let mut groups = vec![vec![base[0]]];
    for &arm in &base[1..] {
        let prev = *groups.last().unwrap().last().unwrap();
        if (means[prev] - means[arm]).abs() < eps {
            groups.last_mut().unwrap().push(arm);
        } else {
            groups.push(vec![arm]);
        }
    }
    groups
}

/// Group certifies (Lemma 2) iff every consecutive pair has
/// `|m̂_a−m̂_b| + z·√(s_a²+s_b²) ≤ ε`.
fn group_certified(g: &[usize], means: &[f64], stds: &[f64], eps: f64) -> bool {
    g.windows(2).all(|w| {
        let (a, b) = (w[0], w[1]);
        (means[a] - means[b]).abs() + Z * (stds[a] * stds[a] + stds[b] * stds[b]).sqrt() <= eps
    })
}

/// Candidate rankings for one proposer: reorder within near-tie groups of size in
/// `2..=GMAX` (all when `gated == false`, only certified ones when `gated`).
/// Larger groups are left in mean order (search-cost cap).
fn candidates(means: &[f64], stds: &[f64], eps: f64, gated: bool) -> Vec<Vec<usize>> {
    let base = rank_by_scores(means);
    let groups = near_tie_groups(&base, means, eps);
    let mut rankings = vec![vec![]];
    for g in &groups {
        let reorderable =
            (2..=GMAX).contains(&g.len()) && (!gated || group_certified(g, means, stds, eps));
        let perms = if reorderable {
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

/// Matching maximizing total belief welfare over the per-proposer candidates, or
/// `None` if the candidate product exceeds `COMBO_CAP` (caller falls back).
fn coordinate(
    per: &[Vec<Vec<usize>>],
    means: &[Vec<f64>],
    recv: &[Vec<usize>],
) -> Option<Matching> {
    let n = per.len();
    let combos: usize = per.iter().map(|c| c.len()).product();
    if combos > COMBO_CAP {
        return None;
    }
    let welfare = |m: &Matching| -> f64 {
        (0..n)
            .map(|p| m.proposer[p].map_or(0.0, |r| means[p][r]))
            .sum()
    };
    let mut idx = vec![0usize; n];
    let mut best = f64::NEG_INFINITY;
    let mut best_m: Option<Matching> = None;
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
    best_m
}

fn regret(util: &[Vec<f64>], star: &Matching, m: &Matching) -> f64 {
    (0..util.len())
        .map(|p| {
            let b = star.proposer[p].map_or(0.0, |r| util[p][r]);
            let g = m.proposer[p].map_or(0.0, |r| util[p][r]);
            b - g
        })
        .sum()
}

/// One regime's running tallies for a policy.
#[derive(Default, Clone, Copy)]
struct Tally {
    rounds: f64,
    stable: f64,
    regret: f64,
    reorders: f64, // measured rounds whose matching differs from plain
    skips: f64,    // rounds the search was capped (fell back to plain)
}
impl Tally {
    fn add(&mut self, stable: bool, reg: f64, reordered: bool, skipped: bool) {
        self.rounds += 1.0;
        self.stable += stable as u8 as f64;
        self.regret += reg;
        self.reorders += reordered as u8 as f64;
        self.skips += skipped as u8 as f64;
    }
}

/// Run ONE plain-Thompson live loop; on measured rounds form the three
/// coordination decisions on the *same* beliefs. `acc[regime][policy]`,
/// regime 0 = early, 1 = late.
fn run_market(
    util: &[Vec<f64>],
    recv: &[Vec<usize>],
    eps: f64,
    horizon: usize,
    seed: u64,
    acc: &mut [[Tally; 3]; 2],
) {
    let n = util.len();
    let truth = true_rankings(util);
    let star = gale_shapley(&truth, recv);
    let mut learners: Vec<GaussianThompson> = (0..n)
        .map(|p| {
            GaussianThompson::new(n, 0.5, 1.0, NOISE * NOISE, seed ^ (0x9E37 * (p as u64 + 1)))
        })
        .collect();
    let mut rng = Rng::new(seed ^ 0xC0DE);

    for t in 1..=horizon {
        let means: Vec<Vec<f64>> = learners.iter().map(|l| l.means()).collect();
        // master exposes per-arm posterior std via the PreferenceLearner::stds
        // trait method (the research branch's temporary posterior_std converges
        // to this on integration).
        let stds: Vec<Vec<f64>> = learners.iter().map(|l| l.stds()).collect();

        // Early window skips the trivial prior-dominated prefix [1, H/10].
        let early = t > horizon / 10 && t <= horizon / 3;
        let late = t > 2 * horizon / 3;
        if t % MEAS_EVERY == 0 && (early || late) {
            let reg = if early { 0 } else { 1 };
            let plain = {
                let rk: Vec<Vec<usize>> = means.iter().map(|m| rank_by_scores(m)).collect();
                gale_shapley(&rk, recv)
            };
            let mk = |gated: bool| -> (Matching, bool) {
                let per: Vec<Vec<Vec<usize>>> = means
                    .iter()
                    .zip(&stds)
                    .map(|(m, s)| candidates(m, s, eps, gated))
                    .collect();
                match coordinate(&per, &means, recv) {
                    Some(m) => (m, false),
                    None => (plain.clone(), true),
                }
            };
            let (ungated, u_skip) = mk(false);
            let (gated, g_skip) = mk(true);
            acc[reg][0].add(
                is_stable(&truth, recv, &plain),
                regret(util, &star, &plain),
                false,
                false,
            );
            acc[reg][1].add(
                is_stable(&truth, recv, &ungated),
                regret(util, &star, &ungated),
                ungated.proposer != plain.proposer,
                u_skip,
            );
            acc[reg][2].add(
                is_stable(&truth, recv, &gated),
                regret(util, &star, &gated),
                gated.proposer != plain.proposer,
                g_skip,
            );
        }

        // Drive learning with a forced-Thompson matching (exploration so beliefs
        // concentrate and the gate can open); this is what gets pulled.
        let force_p = (FORCE_C / t as f64).min(1.0);
        let drive_rk: Vec<Vec<usize>> = (0..n)
            .map(|p| {
                let mut base = rank_by_scores(&learners[p].scores());
                if rng.uniform() < force_p {
                    let forced = (0..n)
                        .max_by(|&a, &b| stds[p][a].partial_cmp(&stds[p][b]).unwrap())
                        .unwrap();
                    base.retain(|&x| x != forced);
                    base.insert(0, forced);
                }
                base
            })
            .collect();
        let realized = gale_shapley(&drive_rk, recv);
        for p in 0..n {
            if let Some(r) = realized.proposer[p] {
                let reward = rng.normal(util[p][r], NOISE);
                learners[p].update(r, reward);
            }
        }
    }
}

fn report(eps: f64) {
    const SWEEP: usize = 300;
    const HORIZON: usize = 3000;
    const N: usize = 5;
    let mut seedgen = Rng::new(20260607);
    let mut acc = [[Tally::default(); 3]; 2];
    for _ in 0..SWEEP {
        let seed = seedgen.below(1_000_000_000) as u64 + 1;
        let mut mgen = Rng::new(seed);
        let (util, recv) = random_market(&mut mgen, N);
        run_market(&util, &recv, eps, HORIZON, seed, &mut acc);
    }

    let names = ["plain  ", "ungated", "gated  "];
    println!("=== ε = {eps}  ({SWEEP} markets {N}x{N}, horizon {HORIZON}) ===");
    for (reg, label) in [
        (0usize, "EARLY (coarse beliefs)"),
        (1, "LATE (accurate beliefs)"),
    ] {
        println!("  {label}:");
        println!("    policy    exact-stable   mean regret   reorder-rate   capped");
        for p in 0..3 {
            let t = acc[reg][p];
            println!(
                "    {}       {:>6.3}        {:>8.4}      {:>6.3}        {:>6.3}",
                names[p],
                t.stable / t.rounds,
                t.regret / t.rounds,
                t.reorders / t.rounds,
                t.skips / t.rounds,
            );
        }
    }
    println!();
}

fn main() {
    println!("Prop. 4 confidence-gating — same-belief-stream A/B\n");
    report(0.05);
    report(0.02);
}
