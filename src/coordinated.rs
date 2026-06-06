//! Coordinated near-tie tie-breaking, the research-track cure for the cascade
//! stall (the implementation delegated by the stall study).
//!
//! When two arms are a *near-tie* in an agent's beliefs, the order it reports
//! between them is essentially arbitrary — yet Gale-Shapley amplifies that
//! arbitrary choice, and a single indifferent agent can push others off the
//! stable matching (the **cascade** mode). No amount of per-agent exploration
//! fixes it (see `docs/theory-identifiability.md`); a market-level *coordinator*
//! does: each round it searches the within-near-tie orderings and picks the
//! matching that maximizes total *belief* welfare `Σ_p mean_p[partner(p)]` — an
//! oracle-free objective. Frozen arms are still handled by vanishing forced
//! exploration.
//!
//! The search is exponential in the largest near-tie group and in the number of
//! tied agents, so it is capped (`max_group`, and a total-combination limit);
//! beyond the cap it falls back to the plain mean-greedy matching for the round.
//!
//! **Two coordinators (validated by `examples/coordinated_validation.rs`).**
//! [`CoordinatedMarket`] is the *ungated* version and carries a negative finding:
//! maximizing belief welfare every round with imperfect mid-learning beliefs
//! raises proposer welfare (regret goes negative) but is *much less* stable than
//! plain Thompson (≈0.70 vs 0.92 tail-stable). The post-hoc cascade cure does not
//! transfer naively to the live loop. [`GatedCoordinatedMarket`] is the Prop-4
//! cure: it coordinates a near-tie only once the pair's posterior is certified
//! tight (see [`near_tie_rankings_certified`]), so it never reorders an
//! un-converged pair. With a tight band it recovers nearly all the lost stability
//! (≈0.91 vs 0.92) at slightly better welfare, and the band `eps` tunes a
//! *bounded* welfare/stability tradeoff. Prop 4 guarantees `2·eps`-stability, not
//! strict stability, so a small eps-controlled gap to plain Thompson remains by
//! design.

use crate::eval::LearningMarket;
use crate::learner::GaussianThompson;
use crate::learner::PreferenceLearner;
use crate::matching::{Matching, gale_shapley};
use crate::prefs::rank_by_scores;
use crate::rng::Rng;

/// Cap on the number of joint orderings searched per round.
const MAX_COMBOS: usize = 4096;

/// All permutations of `items` (small slices only).
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

/// Candidate rankings for one agent: the mean-greedy order, with every
/// within-near-tie-group ordering as an alternative. Groups larger than
/// `max_group` are left in base order (capped, not permuted).
pub fn near_tie_rankings(means: &[f64], eps: f64, max_group: usize) -> Vec<Vec<usize>> {
    let base = rank_by_scores(means); // descending by mean, index tie-break
    // Partition into contiguous near-tie groups.
    let mut groups: Vec<Vec<usize>> = vec![vec![base[0]]];
    for &arm in &base[1..] {
        let prev = *groups.last().unwrap().last().unwrap();
        if (means[prev] - means[arm]).abs() < eps {
            groups.last_mut().unwrap().push(arm);
        } else {
            groups.push(vec![arm]);
        }
    }
    // Cartesian product of within-group permutations.
    let mut rankings = vec![vec![]];
    for g in &groups {
        let perms = if g.len() <= max_group {
            permutations(g)
        } else {
            vec![g.clone()] // cap: do not permute oversized groups
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

/// Like [`near_tie_rankings`] but the *base order* comes from `order` (e.g. a
/// Thompson sample, preserving exploration) while the near-tie grouping is
/// decided by `means` — so only arms that are a genuine near-tie in the means
/// are permuted, and exploration elsewhere is untouched.
fn near_tie_rankings_masked(
    order: &[f64],
    means: &[f64],
    eps: f64,
    max_group: usize,
) -> Vec<Vec<usize>> {
    let base = rank_by_scores(order);
    let mut groups: Vec<Vec<usize>> = vec![vec![base[0]]];
    for &arm in &base[1..] {
        let prev = *groups.last().unwrap().last().unwrap();
        if (means[prev] - means[arm]).abs() < eps {
            groups.last_mut().unwrap().push(arm);
        } else {
            groups.push(vec![arm]);
        }
    }
    let mut rankings = vec![vec![]];
    for g in &groups {
        let perms = if g.len() <= max_group {
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

/// Like [`near_tie_rankings_masked`], but a pair of arms is grouped (and so
/// permuted by the coordinator) only when it passes the **Prop-4 certification
/// test**: the posterior credible band around the belief gap fits inside the
/// near-tie band `eps`,
///
/// `|mean_a − mean_b| + z·√(std_a² + std_b²) ≤ eps`.
///
/// where `z = Φ⁻¹(1−η)` sets the confidence `1−η`. Early on, when the posterior
/// stds are large, no pair certifies and the report is exactly the Thompson
/// sample (so coordination never reorders an un-converged pair — the safety
/// property). Forced exploration drives every `std → 0`, so genuine near-ties
/// eventually certify and get coordinated.
fn near_tie_rankings_certified(
    order: &[f64],
    means: &[f64],
    stds: &[f64],
    eps: f64,
    z: f64,
    max_group: usize,
) -> Vec<Vec<usize>> {
    let base = rank_by_scores(order);
    let certified = |a: usize, b: usize| -> bool {
        (means[a] - means[b]).abs() + z * (stds[a].powi(2) + stds[b].powi(2)).sqrt() <= eps
    };
    let mut groups: Vec<Vec<usize>> = vec![vec![base[0]]];
    for &arm in &base[1..] {
        let prev = *groups.last().unwrap().last().unwrap();
        if certified(prev, arm) {
            groups.last_mut().unwrap().push(arm);
        } else {
            groups.push(vec![arm]);
        }
    }
    let mut rankings = vec![vec![]];
    for g in &groups {
        let perms = if g.len() <= max_group {
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

/// Pick the Gale-Shapley matching maximizing total belief welfare over the joint
/// near-tie orderings. `candidates[p]` is agent `p`'s candidate rankings; falls
/// back to each agent's first candidate if the joint search would exceed the cap.
fn coordinated_match(
    candidates: &[Vec<Vec<usize>>],
    receiver_prefs: &[Vec<usize>],
    means: &[Vec<f64>],
) -> Matching {
    let n = candidates.len();
    let total: usize = candidates
        .iter()
        .map(|c| c.len())
        .try_fold(1usize, |acc, l| acc.checked_mul(l))
        .unwrap_or(usize::MAX);

    let base_rankings: Vec<Vec<usize>> = candidates.iter().map(|c| c[0].clone()).collect();
    if total > MAX_COMBOS {
        return gale_shapley(&base_rankings, receiver_prefs); // capped: no search
    }

    let welfare = |m: &Matching| -> f64 {
        (0..n)
            .map(|p| m.proposer[p].map_or(0.0, |r| means[p][r]))
            .sum()
    };

    let mut best_m = gale_shapley(&base_rankings, receiver_prefs);
    let mut best_w = welfare(&best_m);
    let mut idx = vec![0usize; n];
    loop {
        let rankings: Vec<Vec<usize>> = (0..n).map(|p| candidates[p][idx[p]].clone()).collect();
        let m = gale_shapley(&rankings, receiver_prefs);
        let w = welfare(&m);
        if w > best_w {
            best_w = w;
            best_m = m;
        }
        // Mixed-radix increment over the candidate indices.
        let mut k = 0;
        loop {
            if k == n {
                return best_m;
            }
            idx[k] += 1;
            if idx[k] < candidates[k].len() {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// A learning market that coordinates near-tie tie-breaks (cascade cure) and
/// forces probes of frozen arms (frozen cure).
pub struct CoordinatedMarket {
    true_util: Vec<Vec<f64>>,
    receiver_prefs: Vec<Vec<usize>>,
    learners: Vec<GaussianThompson>,
    counts: Vec<Vec<f64>>,
    eps: f64,
    max_group: usize,
    force_c: f64,
    noise: f64,
    rng: Rng,
    round: usize,
}

impl CoordinatedMarket {
    /// Build a coordinated market. `eps` is the near-tie band; `force_c` the
    /// vanishing-forced-exploration constant (`eps_t = min(1, force_c / t)`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        true_util: Vec<Vec<f64>>,
        receiver_prefs: Vec<Vec<usize>>,
        prior_mean: f64,
        prior_var: f64,
        obs_var: f64,
        eps: f64,
        force_c: f64,
        noise: f64,
        seed: u64,
    ) -> Self {
        let n_p = true_util.len();
        let n_r = receiver_prefs.len();
        let learners = (0..n_p)
            .map(|p| {
                GaussianThompson::new(
                    n_r,
                    prior_mean,
                    prior_var,
                    obs_var,
                    seed ^ (0x2000 + p as u64),
                )
            })
            .collect();
        Self {
            true_util,
            receiver_prefs,
            learners,
            counts: vec![vec![0.0; n_r]; n_p],
            eps,
            max_group: 6,
            force_c,
            noise,
            rng: Rng::new(seed),
            round: 0,
        }
    }

    /// The least-sampled arm for proposer `p` (ties by lowest index).
    fn least_sampled(&self, p: usize) -> usize {
        let c = &self.counts[p];
        (0..c.len())
            .min_by(|&a, &b| c[a].partial_cmp(&c[b]).unwrap())
            .unwrap_or(0)
    }

    /// Play one round and return the realized matching.
    pub fn step(&mut self) -> Matching {
        let n_p = self.true_util.len();
        let n_r = self.receiver_prefs.len();
        // Beliefs: means drive the welfare objective and near-tie grouping;
        // Thompson samples drive exploration (so the loop explores like plain
        // Thompson, which coordination then refines rather than replaces).
        let means: Vec<Vec<f64>> = self.learners.iter().map(|l| l.means()).collect();
        let samples: Vec<Vec<f64>> = self.learners.iter_mut().map(|l| l.scores()).collect();

        let eps_t = (self.force_c / (self.round as f64 + 1.0)).min(1.0);
        let candidates: Vec<Vec<Vec<usize>>> = (0..n_p)
            .map(|p| {
                if self.force_c > 0.0 && self.rng.uniform() < eps_t {
                    // Forced round for p: frozen arm first, rest by sampled order.
                    let frozen = self.least_sampled(p);
                    let mut rest = rank_by_scores(&samples[p]);
                    rest.retain(|&a| a != frozen);
                    let mut ranking = vec![frozen];
                    ranking.extend(rest);
                    vec![ranking]
                } else {
                    // Explore via the Thompson sample, but coordinate the arms
                    // that are a genuine near-tie *in the means* (the cascade
                    // trigger): permute only those, keeping the sampled order
                    // elsewhere.
                    near_tie_rankings_masked(&samples[p], &means[p], self.eps, self.max_group)
                }
            })
            .collect();

        let matching = coordinated_match(&candidates, &self.receiver_prefs, &means);

        for (p, &slot) in matching.proposer.iter().enumerate().take(n_p) {
            if let Some(r) = slot {
                debug_assert!(r < n_r);
                let reward = self.rng.normal(self.true_util[p][r], self.noise);
                self.learners[p].update(r, reward);
                self.counts[p][r] += 1.0;
            }
        }
        self.round += 1;
        matching
    }
}

impl LearningMarket for CoordinatedMarket {
    fn step(&mut self) -> Matching {
        CoordinatedMarket::step(self)
    }
    fn n_proposers(&self) -> usize {
        self.true_util.len()
    }
    fn proposer_util(&self, p: usize, r: usize) -> f64 {
        self.true_util[p][r]
    }
    fn true_proposer_prefs(&self) -> Vec<Vec<usize>> {
        self.true_util
            .iter()
            .map(|row| rank_by_scores(row))
            .collect()
    }
    fn true_receiver_prefs(&self) -> Vec<Vec<usize>> {
        self.receiver_prefs.clone()
    }
}

/// The **Prop-4 gated** coordinated market: the live cure that resolves
/// [`CoordinatedMarket`]'s negative finding.
///
/// Identical to [`CoordinatedMarket`] except a near-tie pair is coordinated only
/// after it passes the confidence-certification test (see
/// [`near_tie_rankings_certified`]). The proof (research track,
/// `docs/theory-identifiability.md` Prop 4):
///
/// - **safe** — an un-converged pair is never reordered by belief welfare, so the
///   market is never worse than plain forced-exploration Thompson (the failure
///   the ungated coordinator showed);
/// - **eventually active** — forcing drives every posterior `std → 0`, so genuine
///   near-ties certify in finite time `Θ(σ²/ε²)` and then get coordinated;
/// - **optimal once active** — among certified orderings it picks the
///   belief-welfare-maximizing stable matching.
///
/// `z` is the certification multiplier `Φ⁻¹(1−η)` for the target confidence
/// `1−η` (e.g. `z ≈ 1.96` for 97.5%).
pub struct GatedCoordinatedMarket {
    true_util: Vec<Vec<f64>>,
    receiver_prefs: Vec<Vec<usize>>,
    learners: Vec<GaussianThompson>,
    counts: Vec<Vec<f64>>,
    eps: f64,
    z: f64,
    max_group: usize,
    force_c: f64,
    noise: f64,
    rng: Rng,
    round: usize,
}

impl GatedCoordinatedMarket {
    /// Build a gated coordinated market. `eps` is the near-tie band, `z` the
    /// certification confidence multiplier, `force_c` the vanishing-forced-
    /// exploration constant.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        true_util: Vec<Vec<f64>>,
        receiver_prefs: Vec<Vec<usize>>,
        prior_mean: f64,
        prior_var: f64,
        obs_var: f64,
        eps: f64,
        z: f64,
        force_c: f64,
        noise: f64,
        seed: u64,
    ) -> Self {
        let n_p = true_util.len();
        let n_r = receiver_prefs.len();
        let learners = (0..n_p)
            .map(|p| {
                GaussianThompson::new(
                    n_r,
                    prior_mean,
                    prior_var,
                    obs_var,
                    seed ^ (0x3000 + p as u64),
                )
            })
            .collect();
        Self {
            true_util,
            receiver_prefs,
            learners,
            counts: vec![vec![0.0; n_r]; n_p],
            eps,
            z,
            max_group: 6,
            force_c,
            noise,
            rng: Rng::new(seed),
            round: 0,
        }
    }

    /// The least-sampled arm for proposer `p` (ties by lowest index).
    fn least_sampled(&self, p: usize) -> usize {
        let c = &self.counts[p];
        (0..c.len())
            .min_by(|&a, &b| c[a].partial_cmp(&c[b]).unwrap())
            .unwrap_or(0)
    }

    /// Play one round and return the realized matching.
    pub fn step(&mut self) -> Matching {
        let n_p = self.true_util.len();
        let n_r = self.receiver_prefs.len();
        let means: Vec<Vec<f64>> = self.learners.iter().map(|l| l.means()).collect();
        let stds: Vec<Vec<f64>> = self.learners.iter().map(|l| l.stds()).collect();
        let samples: Vec<Vec<f64>> = self.learners.iter_mut().map(|l| l.scores()).collect();

        let eps_t = (self.force_c / (self.round as f64 + 1.0)).min(1.0);
        let candidates: Vec<Vec<Vec<usize>>> = (0..n_p)
            .map(|p| {
                if self.force_c > 0.0 && self.rng.uniform() < eps_t {
                    // Forced round for p: frozen arm first, rest by sampled order.
                    let frozen = self.least_sampled(p);
                    let mut rest = rank_by_scores(&samples[p]);
                    rest.retain(|&a| a != frozen);
                    let mut ranking = vec![frozen];
                    ranking.extend(rest);
                    vec![ranking]
                } else {
                    // Coordinate only the *certified* near-ties; everything else
                    // keeps the Thompson sample order (the safety property).
                    near_tie_rankings_certified(
                        &samples[p],
                        &means[p],
                        &stds[p],
                        self.eps,
                        self.z,
                        self.max_group,
                    )
                }
            })
            .collect();

        let matching = coordinated_match(&candidates, &self.receiver_prefs, &means);

        for (p, &slot) in matching.proposer.iter().enumerate().take(n_p) {
            if let Some(r) = slot {
                debug_assert!(r < n_r);
                let reward = self.rng.normal(self.true_util[p][r], self.noise);
                self.learners[p].update(r, reward);
                self.counts[p][r] += 1.0;
            }
        }
        self.round += 1;
        matching
    }
}

impl LearningMarket for GatedCoordinatedMarket {
    fn step(&mut self) -> Matching {
        GatedCoordinatedMarket::step(self)
    }
    fn n_proposers(&self) -> usize {
        self.true_util.len()
    }
    fn proposer_util(&self, p: usize, r: usize) -> f64 {
        self.true_util[p][r]
    }
    fn true_proposer_prefs(&self) -> Vec<Vec<usize>> {
        self.true_util
            .iter()
            .map(|row| rank_by_scores(row))
            .collect()
    }
    fn true_receiver_prefs(&self) -> Vec<Vec<usize>> {
        self.receiver_prefs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::simulate;
    use crate::matching::gale_shapley;

    fn aligned() -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
        let util = vec![
            vec![1.0, 0.4, 0.1],
            vec![0.2, 1.0, 0.5],
            vec![0.1, 0.3, 1.0],
        ];
        let recv = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 1, 0]];
        (util, recv)
    }

    #[test]
    fn near_tie_rankings_enumerate_tied_orders() {
        // Two near-tied arms (0.50, 0.49) give both orderings; the third is apart.
        let means = [0.50, 0.49, 0.10];
        let r = near_tie_rankings(&means, 0.05, 6);
        assert_eq!(r.len(), 2);
        assert!(r.contains(&vec![0, 1, 2]));
        assert!(r.contains(&vec![1, 0, 2]));
    }

    #[test]
    fn coordinated_market_converges() {
        let (util, recv) = aligned();
        let mut m = CoordinatedMarket::new(util, recv, 0.5, 1.0, 0.04, 0.05, 0.5, 0.2, 42);
        let rep = simulate(&mut m, 3000);
        assert!(
            rep.tail_stable_fraction(600) > 0.9,
            "tail stable {}",
            rep.tail_stable_fraction(600)
        );
        assert!(rep.tail_mean_regret(600).abs() < 0.02);
    }

    #[test]
    fn coordination_maximizes_belief_welfare_over_index_tiebreak() {
        // After learning, the coordinated matching's belief welfare is at least
        // that of the plain index tie-break on the same beliefs.
        let (util, recv) = aligned();
        let mut m = CoordinatedMarket::new(util, recv.clone(), 0.5, 1.0, 0.04, 0.1, 0.5, 0.2, 7);
        for _ in 0..1000 {
            m.step();
        }
        let means: Vec<Vec<f64>> = m.learners.iter().map(|l| l.means()).collect();
        let index: Vec<Vec<usize>> = means.iter().map(|x| rank_by_scores(x)).collect();
        let index_m = gale_shapley(&index, &recv);
        let cand: Vec<Vec<Vec<usize>>> =
            means.iter().map(|x| near_tie_rankings(x, 0.1, 6)).collect();
        let coord_m = coordinated_match(&cand, &recv, &means);
        let welfare = |mm: &Matching| -> f64 {
            (0..means.len())
                .map(|p| mm.proposer[p].map_or(0.0, |r| means[p][r]))
                .sum()
        };
        assert!(welfare(&coord_m) >= welfare(&index_m) - 1e-9);
    }

    #[test]
    fn certification_gates_on_confidence() {
        // Two arms a near-tie in the means (0.50, 0.49).
        let order = [0.50, 0.49, 0.10];
        let means = [0.50, 0.49, 0.10];
        // Wide posterior: the credible band exceeds eps, so nothing certifies and
        // the report is a single ranking (no coordination of the un-converged pair).
        let wide = [0.20, 0.20, 0.20];
        let r = near_tie_rankings_certified(&order, &means, &wide, 0.05, 1.96, 6);
        assert_eq!(r.len(), 1, "uncertain pair must not be coordinated");
        // Tight posterior: the band fits inside eps, so the near-tie certifies and
        // both orderings are offered.
        let tight = [0.001, 0.001, 0.001];
        let r = near_tie_rankings_certified(&order, &means, &tight, 0.05, 1.96, 6);
        assert_eq!(r.len(), 2, "certified near-tie must be coordinated");
    }

    #[test]
    fn gated_market_is_safe_and_converges() {
        // The metric the ungated coordinator could fail: the gated market reaches
        // a stable matching on most late rounds (it never reorders un-converged
        // pairs, so it is no worse than plain forced-exploration Thompson).
        let (util, recv) = aligned();
        let mut m = GatedCoordinatedMarket::new(util, recv, 0.5, 1.0, 0.04, 0.1, 1.96, 0.5, 0.2, 9);
        let rep = simulate(&mut m, 3000);
        assert!(
            rep.tail_stable_fraction(600) > 0.9,
            "tail stable {}",
            rep.tail_stable_fraction(600)
        );
    }
}
